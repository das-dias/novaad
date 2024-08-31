"""novaad

LUT-based Analog/Mixed-Signal IC Design Tool using Gm/Id Methodology.

Examples:
  * Forward usage of Gm/Id design model:
    Size a device from a DC-OP and a channel 
    length (aliasing a intrinsic gain spec.).
    
    novaad -vgs 0.8 -vds 0.5 -vsb 0.0 -l 180e-9 -gmid 10

  * Backward usage of Gm/Id design model:
    Obtain the electrical parameters of a 
    device from its W/L sizing and DC-OP.
    
    novaad -vgs 0.8 -vds 0.5 -vsb 0.0 -ids 1e-3 -l 180e-9 -w 1.8e-6

Usage:
  novaad -typ [ nch | pch ] -vgs <vgs> -vds <vds> -vsb <vsb> -l <l> -gmid <gmid>
  novaad -typ [ nch | pch ] -vgs <vgs> -vds <vds> -vsb <vsb> -ids <ids> -l <l> -w <w>
  
  novaad -typ [ ncap | pcap ] -vgs <vgs> -vsb <vsb> -l <l> -c <cgg>
  novaad -typ [ ncap | pcap ] -vgs <vgs> -vsb <vsb> -l <l> -w <w>
  
  novaad -typ [ nsw | psw ] -vgs <vgs> -vsb <vsb> -ron <ron>
  novaad -typ [ nsw | psw ] -vgs <vgs> -vsb <vsb> -l <l> -w <w>
  
  novaad (-h | --help)
  novaad --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  -vgs <vgs>    Gate-Source Voltage.
  -vds <vds>    Drain-Source Voltage.
  -vsb <vbs>    Source-Body Voltage.
  -l <l>        Channel Length.
  -gmid <gmid>  Transconductance Efficiency.
  -w <w>        Channel Width.
  --gui         Launch GUI.
  -typ <typ>    Device Type.
  -ids <ids>    Drain-Source Current.
  -c <cgg>      Total Gate Capacitance.
  -ron <ron>    Triode Region On-Resistance.
"""

from docopt import docopt

__cfg_path__ = '../cfg.yml'

def main():
  args = docopt(__doc__, version='novaad 0.1')
  print(args)

if __name__ == '__main__':
  main()