"""novaad

LUT-based Analog/Mixed-Signal IC Design Tool using Gm/Id Methodology.

Examples:
  * Forward usage of Gm/Id design model:
    Size a device from a DC-OP and a channel 
    length (aliasing a intrinsic gain spec.).
    

  * Backward usage of Gm/Id design model:
    Obtain the electrical parameters of a 
    device from its W/L sizing and DC-OP.
    

Usage:
  novaad (device | moscap | switch ) -i=INPUT_FILE [-o=OUTPUT_FILE] [--gui] [--verbose]
  novaad --gui --type=TYPE --vds=VSB --vbs=VSB [--lch-plot LCH_PLOT ...]
  novaad (-h | --help)
  novaad --version
  novaad COMMAND_FILE
  novaad --setup-config=CONFIG_FILE

Options:
  -h --help                   Show this screen.
  --version                   Show version.
  -i, --input=INPUT_FILE      Input file.
  -o, --output=OUTPUT_FILE    Output file.
  COMMAND_FILE                File with commands to run.
  --gui                       Launch GUI.
  --vds=VDS                   Drain-to-Source Voltage [default: LUT mean].
  --vsb=VSB                   Bulk-to-Source Voltage [default: LUT min].
  --type=TYPE                 Device Type [default: 'nch'].
  --lch-plot                  Channel lengths to include in plot [default: 'all'].
  --verbose                   Verbose Output.
"""

from docopt import docopt, DocoptExit
from warnings import warn
from typing import Union, TypeAlias, Optional
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

from yaml import safe_load, safe_dump
from toml import load as toml_load

from numpy import log10, array
from pandas import DataFrame, concat

from novaad import Device, DeviceSizingSpecification, MoscapSizingSpecification, SwitchSizingSpecification, DcOp, Sizing, GuiApp, ElectricModel, BaseEnum, DeviceType

import pdb

global __REF_CWD__
__REF_CWD__ = str(Path(__file__).resolve().cwd())

global __DEFAULT_CFG_PATH__
__DEFAULT_CFG_PATH__ = str(Path(__REF_CWD__+'/cfg.yml').resolve())
global CFG_PATH
CFG_PATH = __DEFAULT_CFG_PATH__

class InstanceConfig(BaseEnum):
  """Device Configuration."""
  MOSCAP = 'moscap'
  DEVICE = 'device'
  SWITCH = 'switch'

class InstanceObjective(BaseEnum):
  """Instance Objective."""
  SIZING = "sizing"
  CHARACTERIZATION = "characterization"

@dataclass
class SpecInput:
  id: str
  objective: InstanceObjective
  device_type: DeviceType
  sizing: Sizing
  sizing_spec: Union[DeviceSizingSpecification, MoscapSizingSpecification, SwitchSizingSpecification]

def validate_input_file(input_data: dict) -> dict:
  """ Checks if all data parsed are floats """
  for device_id in input_data:
    device_data = input_data[device_id]
    for key in device_data:
      if key == 'type':
        continue
      if type(device_data[key]) is int:
        device_data[key] = float(device_data[key])
      if type(device_data[key]) is not float:
        raise ValueError(f'Invalid input data type for {key} in instance {device_id}.')
    input_data[device_id] = device_data
  return input_data

def parse_toml_device_input(input_file: Union[str, Path]) -> dict[str, SpecInput]:
  """ Parse input file in TOML format into a DeviceSizingSpecification object.
  Args:
      input_file (InputPath): _description_

  Returns:
      dict[str, SpecInput]: Dictionary of instance specs to obtain sizing and electrical parameters.
  """
  specs: dict[str, SpecInput] = {}
  with open(input_file, 'r') as fp:
    data = toml_load(fp)
    data = data['device']
  
    data = validate_input_file(data)
    
    for device_id in data:
      device_data = data[device_id]
      specs[device_id] = SpecInput(
        id=device_id,
        objective=InstanceObjective.SIZING if 'wch' not in device_data else InstanceObjective.CHARACTERIZATION,
        device_type=DeviceType(device_data['type']),
        sizing_spec=DeviceSizingSpecification(
          vds=[device_data['vds']],
          vsb=[device_data['vsb']],
          vgs=[device_data['vgs']],
          lch=[device_data['lch']],
          gmoverid=[device_data['gmid']] if 'gmid' in device_data else None,
          gm=[device_data['gm']] if 'gm' in device_data else None,
          ids=[device_data['ids']] if 'ids' in device_data else None,
        ),
        sizing=Sizing(
          wch=[device_data['wch']] if 'wch' in device_data else None,
          lch=[device_data['lch']]
        )
      )
  return specs

def get_device_instance_results(
  config: dict, 
  spec_input: dict[str, SpecInput],
  ) -> DataFrame:
  """ Get each instance DCOP, Sizing and Electrical Parameters. """
  result = DataFrame(columns=[
    'id', 'type', 'objective', 'vgs', 
    'vds', 'vsb', 'ids', 'gmoverid', 
    'wch', 'lch', 'gm', 'gds', 'cgg', 
    'cgs', 'cgd', 'csb', 'cdb', 'av', 
    'ft', 'fom_bw', 'fom_nbw'
  ])
  devices = {}
  device_types = [dt for dt in ['nch', 'pch'] if dt in config and 'lut-path' in config[dt]]
  
  for device_type in device_types:
    lut_varmap=config[device_type]['varmap']
    bsim4params_varmap=config[device_type]['bsim4-params-varmap']
    devices[device_type] = Device(
      lut_path=config[device_type]['lut-path'], 
      lut_varmap={v:k for k,v in lut_varmap.items()},
      device_type=DeviceType(device_type),
      bsim4params_path=config[device_type]['bsim4-params-path'],
      bsim4params_varmap={v:k for k,v in bsim4params_varmap.items()},
      ref_width=float(config[device_type]['ref-width']),
    )
  for id in spec_input:
    spec = spec_input[id]
    pprint(spec)
    device_type = spec.device_type
    if spec.device_type.value not in devices:
      warn(f'No configuration found for {spec.device_type}.')
      continue
    device = devices[spec.device_type.value]
    sizing = spec.sizing
    dcop = DcOp(
      vds=spec.sizing_spec.vds,
      vsb=spec.sizing_spec.vsb,
      vgs=spec.sizing_spec.vgs,
      ids=spec.sizing_spec.ids
    )
    if spec.objective == InstanceObjective.SIZING:
      dcop, sizing = device.sizing(spec.sizing_spec, return_dcop=True)
    electric_model: ElectricModel = device.electric_model(dcop, sizing)
    result = concat([result, DataFrame(data={
      'id': id,
      'type': spec.device_type.value,
      'objective': spec.objective,
      'vgs': dcop.vgs,
      'vds': dcop.vds,
      'vsb': dcop.vsb,
      'ids': dcop.ids,
      'gmoverid': (array(electric_model.gm)/array(dcop.ids)).tolist(),
      'wch': sizing.wch,
      'lch': sizing.lch,
      'gm': electric_model.gm,
      'gds': electric_model.gds,
      'cgg': electric_model.cgg,
      'cgs': electric_model.cgs,
      'cgd': electric_model.cgd,
      'csb': electric_model.csb,
      'cdb': electric_model.cdb,
      'av': (array(electric_model.gm)/array(electric_model.gds)).tolist(),
      'ft': (array(electric_model.gm)/(2*3.14159*array(electric_model.cgg))).tolist(),
      'fom_bw': (array(electric_model.av)*array(electric_model.gm)/(2*3.14159*array(electric_model.cgg))).tolist(),
      'fom_nbw': ((array(electric_model.gm)/(2*3.14159*array(electric_model.cgg)))*(array(electric_model.gm)/array(dcop.ids))).tolist(),
    })])
  
  return result

def parse_toml_moscap_input(input_file: Union[str, Path]) -> dict[str, SpecInput]:
  """ Parse input file in TOML format into a DeviceSizingSpecification object.
  Args:
      input_file (InputPath): _description_

  Returns:
      dict[str, SpecInput]: Dictionary of instance specs to obtain sizing and electrical parameters.
  """
  specs: dict[str, SpecInput] = {}
  with open(input_file, 'r') as fp:
    data = toml_load(fp)
    data = data['moscap']
  
    data = validate_input_file(data)

    for device_id in data:
      device_data = data[device_id]
      specs[device_id] = SpecInput(
        id=device_id,
        objective=InstanceObjective.SIZING if 'wch' in device_data else InstanceObjective.CHARACTERIZATION,
        device_type=DeviceType(device_data['device_type']),
        sizing_spec=MoscapSizingSpecification(
          vds=[device_data['vds']],
          vsb=[device_data['vsb']],
          vgs=[device_data['vgs']],
          lch=[device_data['lch']],
          cgg=[device_data['cgg']] if 'cgg' in device_data else None,
        ),
        sizing=Sizing(
          wch=[device_data['wch']] if 'wch' in device_data else None,
          lch=[device_data['lch']]
        )
      )
  return specs

def get_moscap_instance_results(
  config: dict,
  spec_input: dict[str, SpecInput],
  ) -> DataFrame:
  """ Get each instance DCOP, Sizing and Electrical Parameters. """
  result = DataFrame(columns=[
    'id', 'type', 'objective', 'vgs', 
    'vds', 'vsb', 'wch', 
    'lch', 'cgs', 'cgd', 'cgg'
  ])
  
  devices = {}
  device_types = [dt for dt in ['nch', 'pch'] if dt in config]
  for device_type in device_types:
    devices[device_type] = MoscapDevice(
      lut_path=config[device_type], 
      lut_varmap=config[device_type]['varmap'],
      device_type=DeviceType(device_type)
    )
  raise NotImplementedError("App not implemented.")

def parse_toml_switch_input(input_file: Union[str, Path]) -> dict[str, SpecInput]:
  """ Parse input file in TOML format into a DeviceSizingSpecification object.
  Args:
      input_file (InputPath): _description_

  Returns:
      dict[str, SpecInput]: Dictionary of instance specs to obtain sizing and electrical parameters.
  """
  specs: dict[str, SpecInput] = {}
  with open(input_file, 'r') as fp:
    data = toml_load(fp)
    data = data['switch']
  
    data = validate_input_file(data)

    for device_id in data:
      device_data = data[device_id]
      specs[device_id] = SpecInput(
        id=device_id,
        objective=InstanceObjective.SIZING if 'wch' in device_data else InstanceObjective.CHARACTERIZATION,
        device_type=DeviceType(device_data['device_type']),
        sizing_spec=SwitchSizingSpecification(
          vsb=[device_data['vsb']],
          vgs=[device_data['vgs']],
          lch=[device_data['lch']],
          ron=[device_data['ron']] if 'ron' in device_data else None,
        ),
        sizing=Sizing(
          wch=[device_data['wch']] if 'wch' in device_data else None,
          lch=[device_data['lch']]
        )
      )
  return specs

def get_switch_instance_results(
  config: dict,
  spec_input: dict[str, SpecInput],
  ) -> DataFrame:
  """ Get each instance DCOP, Sizing and Electrical Parameters. """
  result = DataFrame(columns=[
    'id', 'type', 'objective', 'vgs', 
    'vsb', 'wch', 'lch', 'ron'
  ])
  
  devices = {}
  device_types = [dt for dt in ['nch', 'pch'] if dt in config]
  for device_type in device_types:
    devices[device_type] = SwitchDevice(
      lut_path=config[device_type], 
      lut_varmap=config[device_type]['varmap'],
      device_type=DeviceType(device_type)
    )
  raise NotImplementedError("App not implemented.")

def format_results_dataframe(results: DataFrame, instance_config:InstanceConfig) -> DataFrame:
  """ Format results DataFrame to human-readable format. """
  
  formatted_results = DataFrame(data={
    "Type": results['type'],
    "ID": results['id'],
    "Vgs [V]": results['vgs'].apply(lambda x: f"{x:.2f}"),
    "Vds [V]": results['vds'].apply(lambda x: f"{x:.2f}"),
    "Vsb [V]": results['vsb'].apply(lambda x: f"{x:.2f}"),
    "Wch [um]": results['wch'].apply(lambda x: f"{x/1e-6:.4f}"),
    "Lch [um]": results['lch'].apply(lambda x: f"{x/1e-6:.4f}"),
    "Cgg [fF]": results['cgg'].apply(lambda x: f"{x/1e-15:.4f}"),
    "Cdb [fF]": results['cdb'].apply(lambda x: f"{x/1e-15:.4f}" if x is not None else None),
    "Cgs [fF]": results['cgs'].apply(lambda x: f"{x/1e-15:.4f}" if x is not None else None),
    "Cgd [fF]": results['cgd'].apply(lambda x: f"{x/1e-15:.4f}" if x is not None else None),
    "Csb [fF]": results['csb'].apply(lambda x: f"{x/1e-15:.4f}" if x is not None else None),
    "Objective": results['objective'],
  })

    
  if instance_config is InstanceConfig.DEVICE:
    formatted_results["Ids [uA]"] = results['ids'].apply(lambda x: f"{x/1e-6:.4e}")
    formatted_results["Gm/Id"] = results['gmoverid'].apply(lambda x: f"{x:.4f}")
    formatted_results["Gm [uS]"] = results['gm'].apply(lambda x: f"{x/1e-6:.4f}")
    formatted_results["Gds [uS]"] = results['gds'].apply(lambda x: f"{x/1e-6:.4f}")
    formatted_results["Av [V/V]"] = results['av'].apply(lambda x: f"{x:.4f}")
    formatted_results["Av [dB]"] = results['av'].apply(lambda x: f"{20*log10(x):.4f}")
    formatted_results["Ft [GHz]"] = results['ft'].apply(lambda x: f"{x/1e9:.4f}")
    formatted_results["FOM Av*Ft [GHz]"] = results['fom_bw'].apply(lambda x: f"{x/1e9:.4f}")
    formatted_results["FOM NBW [GHz/V]"] = results['fom_nbw'].apply(lambda x: f"{x/1e9:.4f}")
  if instance_config is InstanceConfig.SWITCH:
    if 'ron' in results.columns:
      formatted_results["Ron [Ohm]"] = results['ron'].apply(lambda x: f"{x:.4f}")
    else: 
      ron = 1/results['gds'].values
      formatted_results["Ron [Ohm]"] = [f"{x:.4e}" for x in ron]
  return formatted_results

def app(args: dict, cfg: dict):
  if args['--verbose']:
    print('Arguments:')
    pprint(args)
    
    print('Configuration:')
    pprint(cfg)
  
  if args['device']:
    specs = parse_toml_device_input(args['--input'])
    results = get_device_instance_results(cfg, specs)
    formatted_results = format_results_dataframe(results, InstanceConfig.DEVICE)
    if args['--gui']:
      gui = GuiApp(cfg)
      gui.show_device_results_table(results)
    if args['--output']:
      formatted_results.to_csv(args['--output'])
    else:
      print(formatted_results)
  elif args['moscap']:
    raise NotImplementedError("Moscap sizing not implemented.")
  
  elif args['switch']:
    raise NotImplementedError("Switch sizing not implemented.")

def config(args):
  cfg = None
  with open(__DEFAULT_CFG_PATH__, 'r') as f:
    cfg = safe_load(f)
  if cfg is None:
    raise ValueError("Configuration file not found.")
  if cfg.get('cfg-path', '') != '':
    CFG_PATH = str(Path(cfg.get('cfg-path', None)).resolve())
    with open(CFG_PATH, 'r') as f:
      cfg = safe_load(f)
  if args["--setup-config"]:
    CFG_PATH = Path(args["<cfg-path>"]).resolve()
    new_cfg = None
    with open(CFG_PATH, 'r') as f:
      new_cfg = safe_load(f)
    if new_cfg is not None:
      with open(__DEFAULT_CFG_PATH__, 'w') as f:
        cfg['cfg-path'] = str(CFG_PATH)
        f.write(safe_dump(cfg, default_flow_style=False, ident=2, width=80)) 
        cfg = new_cfg
  return cfg

def main():  
  args = docopt(__doc__, version='novaad 0.1')
  cfg = config(args)
  if not args["COMMAND_FILE"]:
    app(args, cfg)
  else:
    fp = Path(args["COMMAND_FILE"])
    assert fp.exists(), "Command file does not exist."
    assert fp.is_file(), "Command file must be a file."
    ext = fp.suffixes[0] if len(fp.suffixes) > 0 else ''
    assert ext in ['.txt', ''], "Command file must be a text file."
    with open(args["COMMAND_FILE"], 'r') as f:
      for line in f.readlines():
        line = line if len(line) > 0 else '-h'
        argv = line.split()
        try:
          args = docopt(__doc__, argv=argv, version='novaad 0.1')
          app(args, cfg)
        except DocoptExit:
          continue
        except Exception as e:
          print(f"Input:\t{line}\tOutput: {e}")
          
if __name__ == '__main__':
  main()