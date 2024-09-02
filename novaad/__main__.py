"""novaad

LUT-based Analog/Mixed-Signal IC Design Tool using Gm/Id Methodology.

Examples:
  * Forward usage of Gm/Id design model:
    Size a device from a DC-OP and a channel 
    length (aliasing a intrinsic gain spec.).
    
    novaad --nch --vgs=0.8 --vds=0.5 --vsb=0.0 --lch=180e-9 --gmid=10 --gm=1e-3

  * Backward usage of Gm/Id design model:
    Obtain the electrical parameters of a 
    device from its W/L sizing and DC-OP.
    
    novaad --pch --vgs=0.8 --vds=0.5 --vsb=0.0 --ids=500e-6 --lch=180e-9 --wch=18e-6

Usage:
  novaad (--nch | --pch) [--gui] [ --lch-plot <lch-plot> ... ] [--vgs <vgs> ... --vds <vds> ... --vsb <vsb> ... --lch <lch> ... ( --wch <wch> ... |  --gmid <gmid> ... (--ids <ids> ... | --gm <gm> ...) | --ron <ron> ... | --cgg <cgg> ...)] [--verbose]
  novaad COMMAND_FILE
  novaad (-h | --help)
  novaad --version
  novaad --setup-config <cfg-path>

Options:
  -h --help       Show this screen.
  --version       Show version.
  --vgs           Gate-Source Voltage [default: 0.8].
  --vds           Drain-Source Voltage [default: 0.8].
  --vsb           Substrate Voltage [default: 0.0].
  --lch           Channel Length [default: 180e-9].
  --gmid          Transconductance Efficiency [default: 10.0].
  --ids           Drain Current [default: 100e-6].
  --wch           Channel Width [default: 10e-6].
  --cgg           Gate-Source Capacitance [default: 10e-15].
  --ron           On-Resistance [default: 10.0].
  COMMAND_FILE    Input Command File.
  --gui           Launch GUI.
  --lch-plot      Channel lengths to include in plot [default: all].
  --verbose       Verbose Output.
  --setup-config  Set Configuration File Path.
"""

from docopt import docopt, DocoptExit
from warnings import warn
from pathlib import Path

from pprint import pprint

from yaml import safe_load, safe_dump

from novaad import Device, SizingSpecification, DcOp, Sizing, GuiApp

import pdb

global __REF_CWD__
__REF_CWD__ = str(Path(__file__).resolve().cwd())

global __DEFAULT_CFG_PATH__
__DEFAULT_CFG_PATH__ = str(Path(__REF_CWD__+'/cfg.yml').resolve())
global CFG_PATH
CFG_PATH = __DEFAULT_CFG_PATH__
  
def device_sizing(args, cfg) -> bool:
  verbose = 0
  if args['--verbose']:
    verbose = 1
  device_type = None
  if args['--nch']:
    device_type = 'nch'
  elif args['--pch']:
    device_type = 'pch'
  else: raise ValueError("Device type not found in arguments.")
  device = None
  device_cfg = cfg.get(device_type, None)
  assert device_cfg is not None, "Device ('nch' | 'pch') not found in configuration."
  lut_path = device_cfg.get('lut-path', None)
  assert lut_path is not None, "Device 'lut-path' not found in configuration."
  lut_path = Path(lut_path).resolve()
  assert lut_path is not None, "Device 'lut-path' was not resolved."
  lut_varmap = device_cfg.get('varmap', None)
  if lut_varmap is not None:
    lut_varmap = {v: k for k, v in lut_varmap.items()}
  
  bsim4_params_path = Path(device_cfg.get('bsim4-params-path', None)).resolve()
  bsim4_params_varmap = device_cfg.get('bsim4-params-varmap', None)
  if bsim4_params_varmap is not None:
    bsim4_params_varmap = {k: v for k, v in bsim4_params_varmap.items()}
  reference_width = cfg.get('ref-width', None)
  if reference_width is None:
    warn("Reference width not found in configuration.")
    warn("Using default reference width of 10 um.")
    reference_width = 10e-6
  reference_width = float(reference_width)
  device = Device(
    lut_path, 
    lut_varmap=lut_varmap, 
    bsim4params_path=bsim4_params_path, 
    bsim4params_varmap=bsim4_params_varmap, 
    ref_width=reference_width,
    device_type=device_type
  )
  
  sizing_spec = SizingSpecification(
    vgs=float(args['--vgs']),
    vds=float(args['--vds']),
    vsb=float(args['--vsb']),
    lch=float(args['--lch']),
    gmoverid=float(args['--gmid'])
  )
  if args['--ids']:
    sizing_spec.ids = float(args['--ids'])
  if args['--gm']:
    sizing_spec.ids = None
    sizing_spec.gm = float(args['--gm'])
  dcop, sizing = device.sizing(sizing_spec, return_dcop=True)
  print()
  print("Summary:")
  print('DC-OP:')
  print(dcop.to_df())
  print()
  print('Sizing:')
  print(sizing.to_df())
  return True
  
def moscap_sizing(args, cfg):
  raise NotImplementedError("MOSCAP sizing not implemented.")

def switch_sizing(args, cfg):
  raise NotImplementedError("Switch sizing not implemented.")

def electrical_params_from_sizing_dcop(args, cfg):
  raise NotImplementedError("Electrical parameters from sizing and DC-OP not implemented.")


def app(args, cfg) -> bool:
  if args['--verbose']:
    print('Arguments:')
    pprint(args)
    
    print('Configuration:')
    pprint(cfg)
    
  if args['--gui']:
    gui = GuiApp(cfg)
    gui.run(args, verbose=1 if args['--verbose'] else 0, tol=1e-2)
  if args['--gmid']:
    return device_sizing(args, cfg)
  if args['--ron']:
    return switch_sizing(args, cfg)
  if args['--cgg']:
    return moscap_sizing(args, cfg)
  if args['--wch']:
    return electrical_params_from_sizing_dcop(args, cfg)
  
  return False

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
  if not args["<command-file>"]:
    app(args, cfg)
  else:
    fp = Path(args["<command-file>"])
    assert fp.exists(), "Command file does not exist."
    assert fp.is_file(), "Command file must be a file."
    ext = fp.suffixes[0] if len(fp.suffixes) > 0 else ''
    assert ext in ['.txt', ''], "Command file must be a text file."
    with open(args["<command-file>"], 'r') as f:
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