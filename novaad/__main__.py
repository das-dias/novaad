"""novaad

LUT-based Analog/Mixed-Signal IC Design Tool using Gm/Id Methodology.

Examples:
  * Forward usage of Gm/Id design model:
    Size a device from a DC-OP and a channel 
    length (aliasing a intrinsic gain spec.).
    
    novaad --nch --vgs 0.8 --vds 0.5 --vsb 0.0 --lch 180e-9 --gmid 10

  * Backward usage of Gm/Id design model:
    Obtain the electrical parameters of a 
    device from its W/L sizing and DC-OP.
    
    novaad --pch 0.8 --vds 0.5 --vsb 0.0 --ids 1e-3 --lch 180e-9 --wch 1.8e-6

Usage:
  novaad (--nch | --pch) --vgs=<vgs> --vds=<vds> --vsb=<vsb> --lch=<lch> ( --wch=<wch> |  --gmid=<gmid> (--ids=<ids> | --gm=<gm>) | --ron=<ron> | --cgg=<cgg> )
  novaad <command-file>
  novaad (-h | --help)
  novaad --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --vgs <vgs>    Gate-Source Voltage.
  --vds <vds>    Drain-Source Voltage.
  --vsb <vsb>    Substrate Voltage.
  --lch <lch>    Channel Length.
  --gmid <gmid>  Transconductance Efficiency.
  --ids <ids>    Drain Current.
  --wch <wch>    Channel Width.
  --cgg <cgg>    Gate-Source Capacitance.
  --ron <ron>    On-Resistance.
  --typ <typ>    Device Type.
  <command-file> Input Command File.
"""

from docopt import docopt
from warnings import warn
from pathlib import Path

from yaml import safe_load

from novaad import Device, SizingSpecification, DcOp, Sizing

__cfg_path__ = '../cfg.yml'

def device_sizing(args, cfg) -> bool:
  device_type = args['--typ']
  device = None
  device_cfg = cfg.get(device_type, None)
  assert device_cfg is not None, "Device ('nch' | 'pch') not found in configuration."
  lut_path = device_cfg.get('lut-path', None)
  assert lut_path is not None, "Device 'lut-path' not found in configuration."
  lut_varmap = lut_varmap.get('varmap', None)
  if lut_varmap is not None:
    lut_varmap = {k: v for k, v in lut_varmap.items()}
  bsim4_params_path = device_cfg.get('bsim4-params-path', None)
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
    gmid=float(args['--gmid'])
  )
  if args['--ids']:
    sizing_spec.ids = float(args['--ids'])
  elif args['--gm']:
    sizing_spec.gm = float(args['--gm'])
  sizing = device.size(sizing_spec)
  print(sizing)
  return True
  
def moscap_sizing(args, cfg):
  raise NotImplementedError("MOSCAP sizing not implemented.")

def switch_sizing(args, cfg):
  raise NotImplementedError("Switch sizing not implemented.")

def electrical_params_from_sizing_dcop(args, cfg):
  raise NotImplementedError("Electrical parameters from sizing and DC-OP not implemented.")

  

def app(args, cfg) -> bool:
  if args['--gmid']:
    return device_sizing(args, cfg)
  elif args['--ron']:
    return switch_sizing(args, cfg)
  elif args['--cgg']:
    return moscap_sizing(args, cfg)
  elif args['--wch']:
    return electrical_params_from_sizing_dcop(args, cfg)
  return False

def main():
  cfg = None
  with open(__cfg_path__, 'r') as f:
    cfg = safe_load(f)
  args = docopt(__doc__, version='novaad 0.1')
  if not args["<command-file>"]:
    app(args, cfg)
  else:
    fp = Path(args["<command-file>"])
    assert fp.exists(), "Command file does not exist."
    assert fp.is_file(), "Command file must be a file."
    path, ext = fp.suffixes
    assert ext in ['.txt', ''], "Command file must be a text file."
    with open(args["<command-file>"], 'r') as f:
      for line in f:
        args = docopt(line, version='novaad 0.1')
        app(args, cfg)

if __name__ == '__main__':
  main()