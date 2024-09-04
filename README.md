# novaad
### Gm/Id Analog/Mixed-Signal IC Design Tool

This tool is inspired in [Prof. B. Mumman's Gm/Id CAD tool/book](https://github.com/bmurmann/Book-on-gm-ID-design/tree/main/starter_kit), also providing support for interpolation upon look-up. This tool extends the former by being deployed as a Python library, providing an API as well as a CLI that can both be used to perform analog IC design.

<div>
<img align=center width=500 src=./docs/figs/eqs.png>
</img>
</div>

The tool can be used with the Gm/Id method in a forward modelling problem:
  - Given: 
    - A DC operating point (e.g., vgs, vds, vsb, ids),
    - A target drain current (ids) or transconductance (gm),
    - A target Gm/Id ratio (encoding inversion region information), and a channel length (lch) (encoding intirnsic gain (Gm/Gds) and maximum operating frequency information (Gm/Cgg))
  - Optain the device width (w) that satisfies the target Gm/Id ratio for the given DC-OP.
If Vgs is not provided, it will also be targetted as an output design parameter to achieve the target Gm/Id ratio.

The tool can also be used in a reverse modelling problem:
  - Given:
    - A DC operating point (vgs, vds, vsb, ids),
    - A device width (w),
    - A channel length (lch),
  - Obtain the electrical characteristics of the device (e.g., gm, gds, cgg, cgs, cgd, etc.)

Note that this tool doesn't take into account the effect of number of fingers right now. This is a feature that will be added in the future.

## 1. Installation

Install as a normal Python package from PyPI:

```bash
pip install novaad
```

## 2. Usage

There are a sequence of steps required to use the tool:
1. Extract the Look-Up Table (LUT) for each NMOS and PMOS device in the technology of interest. 
   1. Example Cadence Ocean scripts are provided in my['cadence-scripts'](https://github.com/das-dias/cadence_workflow/tree/master/cadence-scripts/gmid) directory to extract the LUTs.

2. Make sure the LUTs are provided in ```.csv``` format.
   1. The LUTs should follow the same exact naming convention used in the Cadence Ocean scripts pointed above but without the device type extension (e.g., ```nmos: _n``` or ```pmos: _p```).
   
   2. For both NMOS and PMOS devices, Gate-Source Voltage (Vgs), Drain-Source Voltage (Vds), Source-Bulk (Vsb) and Drain-Source Current (Ids) should be provided in the LUTs with the naming: ```vgs, vds, vsb, ids```.

### *2.1. API*

The CLI is supported by a flexible Python library that can be used from a Python script. The following example shows how to use the library to perform the Gm/Id design:

```python
# Create a Device instance object
device = Device(
        "./test/data/test_nch_lut_renamed.csv",
        device_type="nch",
)

# Perform a customized LUT lookup:

output_cols = ["av", "jd", "ft"]
target = { # all lists must have the same length
    "vgs": [device.lut["vgs"].mean()],
    "vds": [0.9],
    "vsb": [0.0],
    "lch": [device.lut["lch"].min() * 1],
    "gmoverid": [10.0],
}

kwargs = { # interp_method: "pchip" (default), see Pandas.DataFrame.interpolate for more options
    "interp_method": "nearest",
    "interp_mode": "default",
}
row = device.look_up(output_cols,targetreturn_xy=True, **kwargs)

print(row)
""" row ( as Pandas.DataFrame )
Output: 

       jd           lch            ft  gmoverid  vgs  vsb  vds       av
1  8.0358  1.800000e-07  2.770000e+09      10.0  0.9  0.0  0.9  175.469
"""

# Perform a Gm/Id sizing operation for a given device:

sizing_spec = DeviceSizingSpecification(
    vgs=0.5, vds=0.6, vsb=0.0, lch=device.lut["lch"].min(), gmoverid=10.0, gm=1e-3
)
sizing = device.sizing(sizing_spec)
print(sizing.to_df())
""" sizing ( as Pandas.DataFrame )
Output:
            lch       wch
0  1.800000e-07  0.000013
"""
```

This API effectively enables the user to fit the tool into an optimization loop to perform automatic sizing of whole circuits like OTA's, filters, buffers, etc.

### *2.2. CLI*



After extracting the LUT's, create a configuration file using ```.yaml``` format with the following structure:
  

```yaml
# example_config.yaml

nch:
  lut_path: path/to/nmos_lut.csv
  ref_width: 10e-6 # Constant width used to extract the LUT

pch:
  lut_path: path/to/pmos_lut.csv
  ref_width: 10e-6 # Constant width used to extract the LUT

```

After creating the configuration file and creating the LUTs, you are finally able to use the CLI to perform the Gm/Id design.

* **Help**

```bash
# Input:
$ novaad --help
```

```bash
# Output:
novaad

LUT-based Analog/Mixed-Signal IC Design Tool using Gm/Id Methodology.
    
Usage:
  novaad (device | moscap | switch ) -i=INPUT_FILE [-o=OUTPUT_FILE] [--noise] [--gui] [--verbose] [--config=CONFIG_FILE]
  novaad --gui --type=TYPE [--vds=VSB --vsb=VSB --lch-plot LCH_PLOT ...] [--config=CONFIG_FILE]
  novaad (-h | --help)
  novaad --version
  novaad COMMAND_FILE

Options:
  -h --help                   Show this screen.
  --version                   Show version.
  -i, --input=INPUT_FILE      Input file.
  -o, --output=OUTPUT_FILE    Output file.
  COMMAND_FILE                File with commands to run.
  --gui                       Launch GUI.
  --vds=VDS                   Drain-to-Source Voltage. Default value is LUT mean.
  --vsb=VSB                   Bulk-to-Source Voltage. Default value is LUT minimum.
  --type=TYPE                 Device Type [default: 'nch'].
  --lch-plot                  Channel lengths to include in plot [default: 'all'].
  --noise                     Include noise summary in the analysis.
  --verbose                   Verbose Output.
  --config=CONFIG_FILE        Configuration file [default: 'cfg.yml'].

```

* **Input file:**


To use ```novaad```'s design capabilities, you must specify your design targets in an input ```.toml``` file. An example of such file is shown below:

```toml
# example_input.toml
[device]
m0 = { type = "nch", vds = 0.5, vsb = 0.0, lch = 180e-9, gmid = 26, gm = 1e-3 }
m1 = { type = "pch", vgs = 0.8, vds = 1.3, vsb = 0.0, lch = 180e-9, gmid = 23, gm = 1e-3 } # No output generated for this device; there is no LUT
m2 = { type = "nch" , vgs = 0.8, vds = 0.5, vsb = 0.0, lch = 180e-9, gmid = 10, ids = 500e-6 }
m3 = { type = "nch" , vgs = 0.8, vds = 0.5, vsb = 0.0, lch = 1e-6, wch = 12e-6, ids = 100e-6 }

[noise]
m0 = {t_celsius = 36, noise_fmax = 100e6, flicker_corner_freq=30e3}

# the current test lut doesn't support vsb above 0.0
[moscap]
m4 = { type = "nch", vgs = 0.8, vds = 0.5, vsb = 0.0, lch = 180e-9, cgg = 1.2e-15 } # vds is ignored and set to minimum in LUT table
m5 = { type = "pch", vgs = 0.8, vds = 1.3, vsb = 0.0, lch = 180e-9, cgg = 1.2e-15 } # No output due to no LUT!
m6 = { type = "nch", vgs = 0.8, vsb = 0.0, lch = 180e-9, cgg = 1e-14 }
m7 = { type = "nch", vgs = 0.8, vsb = 0.0, lch = 1e-6, cgg = 1e-14 }
m8 = { type = "nch", vgs = 0.8, vsb = 0.0, lch = 1e-6, wch = 12e-6 } # if wch is parsed, cgg is ignored

[switch]
m9 = { type = "nch", vgs = 1.8, vds = 0.5, vsb = 0.0, lch = 180e-9, ron = 10 } # vds is ignored and set to minimum in LUT table
m10 = { type = "pch", vgs = 0.8, vds = 1.3, vsb = 0.0, lch = 180e-9, ron = 10 } # No output due to no LUT!
m11 = { type = "nch", vgs = 0.8, vsb = 0.0, lch = 180e-9, ron = 10 }
m12 = { type = "nch", vgs = 0.8, vsb = 0.0, lch = 180e-9, ron = 10 }
m13 = { type = "nch", vgs = 0.8, vsb = 0.0, lch = 1e-6, ron = 10 }
m14 = { type = "nch", vgs = 0.8, vsb = 0.0, lch = 1e-6, wch = 12e-6 } # if wch is parsed, ron is ignored
```

The input file is separated in four sections:
1. **Device:** This section is used to specify the design targets for the Gm/Id design of transconductor devices. The following parameters are required:
   - ```type```: Device type (```nch``` or ```pch```).
   - ```vgs```: Gate-to-Source Voltage.
   - ```vds```: Drain-to-Source Voltage.
   - ```vsb```: Bulk-to-Source Voltage.
   - ```lch```: Channel Length.
   - ```gmid```: Target Gm/Id ratio.
   - ```gm```: Target transconductance.
   - ```ids```: Target drain current.
   - ```wch```: Channel Width (optional).

2. **Noise:** This section is used to specify the parameters to perform a simple model-based computation of the steady-state thermal and flicker noise for ```device``` instances (**ONLY DEVICE**). The following parameters are required:
   - ```t_celsius```: Temperature in Celsius.
   - ```noise_fmax```: Maximum frequency for noise analysis.
   - ```flicker_corner_freq```: Flicker noise corner frequency.

NOTE: To ignore flicker just set ```flicker_corner_freq = 0.0```.

3. **Moscap:** This section is used to specify the design targets for the Gm/Id design of MOSCAP devices. The following parameters are required:
   - ```type```: Device type (```nch``` or ```pch```).
   - ```vgs```: Gate-to-Source Voltage.
   - ```vds```: Drain-to-Source Voltage.
   - ```vsb```: Source-to-Bulk Voltage.
   - ```lch```: Channel Length.
   - ```cgg```: Target total-gate capacitance.

4. **Switch:** This section is used to specify the design targets for the Gm/Id design of Switch devices. The following parameters are required:
    - ```type```: Device type (```nch``` or ```pch```).
    - ```vgs```: Gate-to-Source Voltage.
    - ```vds```: Drain-to-Source Voltage.
    - ```vsb```: Source-to-Bulk Voltage.
    - ```lch```: Channel Length.
    - ```ron```: Target on-resistance.

NOTE 2: The ```vds``` parameter is ignored for MOSCAP and SWITCH devices and set to the minimum value in the LUT table (ideally, will be set to ```0.0 V``` if such simulation space is included in the LUTs).

By specifing ```device```, ```moscap``` or ```switch``` in the CLI, the tool will perform the Gm/Id design for the corresponding instances in each section of the input file. 

An example:

```bash
# Input:
$ novaad device -i=path/to/example_input.toml --config=path/to/example_config.yaml
```

```bash
# Output:
(novaad-py3.11) ➜  novaad git:(master) ✗ novaad device --gui -i/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_input.toml
.../novaad/novaad/__main__.py:166: UserWarning: No configuration found for pch.
  warn(f'No configuration found for {spec.device_type.value}.')

Device Sizing Results:
  Type  ID Vgs [V] Vds [V] Vsb [V] Wch [um] Lch [um]  Cgg [fF]  ...    Gm/Id    Gm [uS] Gds [uS]  Av [V/V]  Av [dB] Ft [GHz] FOM Av*Ft [GHz] FOM NBW [GHz/V]
0  nch  m0    0.80    0.50    0.00  33.8682   0.1800  109.9022  ...  23.6611   411.4982   6.0861  143.0626  43.1105   0.9885        141.4173         23.3890
0  nch  m2    0.80    0.50    0.00  65.8263   0.1800  314.6496  ...   9.9727  4986.3410  50.6533  107.3693  40.6176   2.5250        271.1076         25.1810
0  nch  m3    0.80    0.50    0.00  12.0000   1.0000   68.4000  ...   8.9520   895.2000   7.6440  126.4365  42.0374   2.2800        288.2752         20.4106

[3 rows x 22 columns]
```


* **GUI**
  
The GUI can be launched in two modes of operation:

1. Giving the ```--gui``` flag and specifying the type of device (```nch``` or ```pch```) to the graphs required to perform the Gm/Id design.
   1. The channel lengths to include in the plot can also be specified using the ```--lch-plot``` flag.

Example:

```bash
# Input:
$ novaad --gui --type=nch --config=path/to/example_config.yaml
```
<div>
<img align=center width=700 src=./docs/figs/gui_mode1.png>
</img>
</div>



2. Giving the ```--gui``` flag and specifying the input file with the design targets. This will open a GUI page containing the result tables for the design targets specified in the input file.

Example:

```bash
# Input:
$ novaad device --gui -i=path/to/example_input.toml --config=path/to/example_config.yaml
```

<div>
<img align=center width=700 src=./docs/figs/device.png>
</img>
</div>

## 3. Dependencies

This project was developed using a Python 3.11 virtual environment. The following dependencies are required:

```bash
# requirements.txt
numpy:    pip install numpy
pandas:   pip install pandas
pyyaml:   pip install pyyaml
plotly:   pip install plotly
docopt:   pip install docopt
scipy:    pip install scipy
pydantic: pip install pydantic
confz:    pip install confz
toml:     pip install toml
```

I recommend the creation of a [Python virtual environment](https://docs.python.org/3/library/venv.html) in the folder in which you want to [setup](https://python.land/virtual-environments/virtualenv) this tool.

```bash
# Create a virtual environment on the current directory 
# inside a folder named 'venv'

$ python -m venv venv

# Activate the virtual environment
$ source ./venv/bin/activate

# Install tool
$ pip install novaad
```

If you are using Windows 10 or above, consider using the [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install) to run the tool. It will make your life easier as no dedicated Windows support is provided for now.

## 4. License

This project is licensed under the [BSD 2-Clause License](https://opensource.org/license/bsd-2-clause) - see the [LICENSE](./LICENSE) file for details.

## 5. Contributing

For now, the only maintainer of this project is the author (me). If you want to contribute, please contact me at ```das.dias@campus.fct.unl.pt``` or ```ddias@tudelft.nl```.

## 6. Acknowledgements

* Prof. João Goes - Reverse usage mode of the Gm/Id method for getting the electrical characteristics of the device from sizing and DC-OP information.

* [Positive Feedback web page](https://positivefb.com/gm-id-methodology/)