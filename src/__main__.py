"""novaad

LUT-based Analog/Mixed-Signal IC Design Tool using Gm/Id Methodology.

Examples:
  * Forward usage of Gm/Id design model:
    Size a device from a DC-OP and a channel 
    length (aliasing a intrinsic gain spec.).
    
    novaad --typ nch --vgs 0.8 --vds 0.5 --vsb 0.0 --lch 180e-9 --gmid 10

  * Backward usage of Gm/Id design model:
    Obtain the electrical parameters of a 
    device from its W/L sizing and DC-OP.
    
    novaad --typ nch --vgs 0.8 --vds 0.5 --vsb 0.0 --ids 1e-3 --lch 180e-9 --wch 1.8e-6

Usage:
  novaad --typ [ nch | pch ] --vgs <vgs> --vds <vds> --vsb <vsb> --lch <lch> --gmid <gmid>
  novaad --typ [ nch | pch ] --vgs <vgs> --vds <vds> --vsb <vsb> --ids <ids> -lch <lch> --wch <wch>
  
  novaad --typ [ ncap | pcap ] --vgs <vgs> --vsb <vsb> --lch <lch> --cgg <cgg>
  novaad --typ [ ncap | pcap ] --vgs <vgs> --vsb <vsb> --lch <lch> --wch <wch>
  
  novaad --typ [ nsw | psw ] --vgs <vgs> --vsb <vsb> --ron <ron>
  novaad --typ [ nsw | psw ] --vgs <vgs> --vsb <vsb> --lch <lch> --wch <wch>
  
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
"""

from docopt import docopt
from novaad import Device, SizingSpecification, DcOp, Sizing

__cfg_path__ = '../cfg.yml'

def device(args, cfg):
  pass

def moscap(args, cfg):
  pass

def switch(args, cfg):
  pass

def main():
  args = docopt(__doc__, version='novaad 0.1')
  device_type = args['--typ']
  if device_type in ['nch', 'pch']:
    pass
  elif device_type in ['ncap', 'pcap']:
    pass
  elif device_type in ['nsw', 'psw']:
    pass
  raise ValueError(f"Device type '{device_type}' not supported.")

if __name__ == '__main__':
  main()