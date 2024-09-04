"""novaad

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
"""

from docopt import docopt, DocoptExit

from warnings import warn
from typing import Union ,Optional
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

from yaml import safe_load, safe_dump
from toml import load as toml_load
from confz import BaseConfig, FileSource, validate_all_configs
from pydantic import AnyUrl, FilePath

from numpy import log10, array
from pandas import DataFrame, concat

from novaad import (
    Device,
    Moscap,
    Switch,
    DeviceSizingSpecification,
    MoscapSizingSpecification,
    SwitchSizingSpecification,
    DcOp,
    Sizing,
    GuiApp,
    ElectricModel,
    BaseEnum,
    DeviceType,
)

"""
global __REF_CWD__
__REF_CWD__ = str(Path(__file__).resolve().cwd())

global __DEFAULT_CFG_PATH__
__DEFAULT_CFG_PATH__ = str(Path(__REF_CWD__ + "/cfg.yml").resolve())

global CFG_PATH
CFG_PATH = __DEFAULT_CFG_PATH__
"""

class LutConfig(BaseConfig):
    lut_path: Union[str, AnyUrl, FilePath] # required
    ref_width: float # required
    bsim4_params_path: Optional[Union[str, AnyUrl, FilePath]] = None


class Config(BaseConfig):
    nch: Optional[LutConfig] = None
    pch: Optional[LutConfig] = None
    out_log_path: Optional[Union[FilePath, AnyUrl, str]] = None

    CONFIG_SOURCES = FileSource(
        folder=str(Path(__file__).resolve().cwd()),
        file="novaad.cfg.yml",
        file_from_env="NOVAAD_ENV",
        file_from_cl="--config",
    )


class InstanceConfig(BaseEnum):
    """Device Configuration."""

    MOSCAP = "moscap"
    DEVICE = "device"
    SWITCH = "switch"


class InstanceObjective(BaseEnum):
    """Instance Objective."""

    SIZING = "sizing"
    CHARACTERIZATION = "characterization"

@dataclass
class NoiseSpecInput:
    flicker_corner_freq: float
    noise_fmax: float
    t_celsius: float

@dataclass
class SpecInput:
    id: str
    objective: InstanceObjective
    device_type: DeviceType
    sizing: Sizing
    sizing_spec: Union[
        DeviceSizingSpecification, MoscapSizingSpecification, SwitchSizingSpecification
    ]
    noise: Optional[NoiseSpecInput] = None

# TODO: Change data structures to Pydantic BaseModels to automatically 
# validate input data types and values.
def validate_input_file(input_data: dict) -> dict:
    """Checks if all data parsed are floats"""
    for device_id in input_data:
        device_data = input_data[device_id]
        for key in device_data:
            if key == "type":
                continue
            if type(device_data[key]) is int:
                device_data[key] = float(device_data[key])
            if type(device_data[key]) is not float:
                raise ValueError(
                    f"Invalid input data type for {key} in instance {device_id}."
                )
        input_data[device_id] = device_data
    return input_data


def parse_toml_device_input(input_file: Union[str, Path]) -> dict[str, SpecInput]:
    """Parse input file in TOML format into a DeviceSizingSpecification object.
    Args:
        input_file (InputPath): _description_

    Returns:
        dict[str, SpecInput]: Dictionary of instance specs to obtain sizing and electrical parameters.
    """
    specs: dict[str, SpecInput] = {}
    with open(input_file, "r") as fp:
        data = toml_load(fp)
        noise_data = data.get("noise", {})
        noise_data = validate_input_file(noise_data)
        data = data["device"]
        data = validate_input_file(data)
        for device_id in data:
            device_noise_data = noise_data.get(device_id, None)
            device_data = data[device_id]
            specs[device_id] = SpecInput(
                id=device_id,
                objective=(
                    InstanceObjective.SIZING
                    if "wch" not in device_data
                    else InstanceObjective.CHARACTERIZATION
                ),
                device_type=DeviceType(device_data["type"]),
                sizing_spec=DeviceSizingSpecification(
                    vgs=[device_data["vgs"]] if "vgs" in device_data else None,
                    vds=[device_data["vds"]],
                    vsb=[device_data["vsb"]],
                    lch=[device_data["lch"]],
                    gmoverid=[device_data["gmid"]] if "gmid" in device_data else None,
                    gm=[device_data["gm"]] if "gm" in device_data else None,
                    ids=[device_data["ids"]] if "ids" in device_data else None,
                ),
                sizing=Sizing(
                    wch=[device_data["wch"]] if "wch" in device_data else None,
                    lch=[device_data["lch"]],
                ),
                noise = NoiseSpecInput(
                    flicker_corner_freq=device_noise_data.get("flicker_corner_freq", 40e3),
                    noise_fmax=device_noise_data.get("noise_fmax", 100e6),
                    t_celsius=device_noise_data.get("t_celsius", 27)
                ) if device_noise_data else None
            )
    return specs


def get_device_instance_results(
    config: dict,
    spec_input: dict[str, SpecInput],
    return_noise: bool = False,
    **kwargs
) -> DataFrame:
    """Get each instance DCOP, Sizing and Electrical Parameters."""
    result = DataFrame(
        columns=[
            "id",
            "type",
            "objective",
            "vgs",
            "vds",
            "vsb",
            "ids",
            "gmoverid",
            "wch",
            "lch",
            "gm",
            "gds",
            "cgg",
            "cgs",
            "cgd",
            "csb",
            "cdb",
            "av",
            "ft",
            "fom_bw",
            "fom_nbw",
            "vng_rms"
        ]
    )
    devices = {}
    device_types = [
        dt for dt in ["nch", "pch"] if dt in config and "lut_path" in config[dt]
    ]

    for device_type in device_types:
        devices[device_type] = Device(
            lut_path=config[device_type]["lut_path"],
            device_type=DeviceType(device_type),
            ref_width=float(config[device_type]["ref_width"]),
        )
    for id in spec_input:
        spec = spec_input[id]
        if spec.device_type.value not in devices:
            warn(f"No configuration found for {spec.device_type.value}.")
            continue
        device = devices[spec.device_type.value]
        sizing = spec.sizing
        dcop = DcOp(
            vds=spec.sizing_spec.vds,
            vsb=spec.sizing_spec.vsb,
            vgs=spec.sizing_spec.vgs,
            ids=spec.sizing_spec.ids,
        )
        if spec.objective == InstanceObjective.SIZING:
            dcop, sizing = device.sizing(spec.sizing_spec, return_dcop=True)
        electric_model: ElectricModel = device.electric_model(dcop, sizing)
        vng_rms = None
        if return_noise and (spec.noise is not None):
          t_celsius = spec.noise.t_celsius
          flicker_corner_freq = spec.noise.flicker_corner_freq
          noise_fmax = spec.noise.noise_fmax
          vng_rms = device.total_gate_referred_noise(dcop,sizing,t_celsius,flicker_corner_freq,noise_fmax)
        result = concat(
            [
                result,
                DataFrame(
                    data={
                        "id": id,
                        "type": spec.device_type.value,
                        "objective": spec.objective,
                        "vgs": dcop.vgs,
                        "vds": dcop.vds,
                        "vsb": dcop.vsb,
                        "ids": dcop.ids,
                        "gmoverid": (
                            array(electric_model.gm) / array(dcop.ids)
                        ).tolist(),
                        "wch": sizing.wch,
                        "lch": sizing.lch,
                        "gm": electric_model.gm,
                        "gds": electric_model.gds,
                        "cgg": electric_model.cgg,
                        "cgs": electric_model.cgs,
                        "cgd": electric_model.cgd,
                        "csb": electric_model.csb,
                        "cdb": electric_model.cdb,
                        "av": electric_model.av,
                        "ft": electric_model.ft,
                        "fom_bw": (
                            array(electric_model.av) * array(electric_model.ft)
                        ).tolist(),
                        "fom_nbw": (
                            array(electric_model.ft)
                            * (array(electric_model.gm) / array(dcop.ids))
                        ).tolist(),
                        "vng_rms": vng_rms
                    }
                ),
            ]
        )

        current_gmid = result[result["id"] == id]["gmoverid"].values
        max_gmid = device.lut["gmoverid"].max()
        if any(current_gmid > max_gmid):
            warn(
                f"Gm/Id ratio {current_gmid} for {id} is higher than the maximum value in the LUT {[max_gmid]}. Consider decreasing target Gm/Id or Vgs."
            )
    # fill missing values
    result["ron"] = None
    return result


def parse_toml_moscap_input(input_file: Union[str, Path]) -> dict[str, SpecInput]:
    """Parse input file in TOML format into a DeviceSizingSpecification object.
    Args:
        input_file (InputPath): _description_

    Returns:
        dict[str, SpecInput]: Dictionary of instance specs to obtain sizing and electrical parameters.
    """
    specs: dict[str, SpecInput] = {}
    with open(input_file, "r") as fp:
        data = toml_load(fp)
        data = data["moscap"]

        data = validate_input_file(data)

        for device_id in data:
            device_data = data[device_id]
            specs[device_id] = SpecInput(
                id=device_id,
                objective=(
                    InstanceObjective.SIZING
                    if "wch" not in device_data
                    else InstanceObjective.CHARACTERIZATION
                ),
                device_type=DeviceType(device_data["type"]),
                sizing_spec=MoscapSizingSpecification(
                    vsb=[device_data["vsb"]],
                    vgs=[device_data["vgs"]],
                    lch=[device_data["lch"]],
                    cgg=[device_data["cgg"]] if "cgg" in device_data else None,
                ),
                sizing=Sizing(
                    wch=[device_data["wch"]] if "wch" in device_data else None,
                    lch=[device_data["lch"]],
                ),
            )
    return specs


def get_moscap_instance_results(
    config: dict,
    spec_input: dict[str, SpecInput],
) -> DataFrame:
    """Get each instance DCOP, Sizing and Electrical Parameters."""
    result = DataFrame(
        columns=[
            "id",
            "type",
            "objective",
            "vgs",
            "vds",
            "vsb",
            "wch",
            "lch",
            "cgs",
            "cgd",
            "cgg",
            "cdb",
            "csb",
        ]
    )

    devices = {}
    device_types = [
        dt for dt in ["nch", "pch"] if dt in config and "lut_path" in config[dt]
    ]

    for device_type in device_types:
        devices[device_type] = Moscap(
            lut_path=config[device_type]["lut_path"],
            device_type=DeviceType(device_type),
            ref_width=float(config[device_type]["ref_width"]),
        )

    for id in spec_input:
        spec = spec_input[id]
        if spec.device_type.value not in devices:
            warn(f"No configuration found for {spec.device_type.value}.")
            continue
        device = devices[spec.device_type.value]
        sizing = spec.sizing
        spec.sizing_spec.vds = [device.lut["vds"].min()]
        dcop = DcOp(
            vds=[device.lut["vds"].min()],
            vsb=spec.sizing_spec.vsb,
            vgs=spec.sizing_spec.vgs,
        )
        if spec.objective == InstanceObjective.SIZING:
            dcop, sizing = device.sizing(spec.sizing_spec, return_dcop=True)
        electric_model: ElectricModel = device.electric_model(dcop, sizing)
        result = concat(
            [
                result,
                DataFrame(
                    data={
                        "id": id,
                        "type": spec.device_type.value,
                        "objective": spec.objective.value,
                        "vgs": dcop.vgs,
                        "vds": dcop.vds,
                        "vsb": dcop.vsb,
                        "wch": sizing.wch,
                        "lch": sizing.lch,
                        "cgs": electric_model.cgs,
                        "cgd": electric_model.cgd,
                        "cgg": electric_model.cgg,
                        "cdb": electric_model.cdb,
                        "csb": electric_model.csb,
                    }
                ),
            ]
        )

    # fill missing values
    result["ids"] = None
    result["gm"] = None
    result["gds"] = None
    result["av"] = None
    result["ft"] = None
    result["fom_bw"] = None
    result["fom_nbw"] = None
    result["gmoverid"] = None
    result["ron"] = None
    result["vng_rms"] = None
    return result


def parse_toml_switch_input(input_file: Union[str, Path]) -> dict[str, SpecInput]:
    """Parse input file in TOML format into a DeviceSizingSpecification object.
    Args:
        input_file (InputPath): _description_

    Returns:
        dict[str, SpecInput]: Dictionary of instance specs to obtain sizing and electrical parameters.
    """
    specs: dict[str, SpecInput] = {}
    with open(input_file, "r") as fp:
        data = toml_load(fp)
        data = data["switch"]

        data = validate_input_file(data)

        for device_id in data:
            device_data = data[device_id]
            specs[device_id] = SpecInput(
                id=device_id,
                objective=(
                    InstanceObjective.SIZING
                    if "wch" not in device_data
                    else InstanceObjective.CHARACTERIZATION
                ),
                device_type=DeviceType(device_data["type"]),
                sizing_spec=SwitchSizingSpecification(
                    vsb=[device_data["vsb"]],
                    vgs=[device_data["vgs"]],
                    lch=[device_data["lch"]],
                    ron=[device_data["ron"]] if "ron" in device_data else None,
                ),
                sizing=Sizing(
                    wch=[device_data["wch"]] if "wch" in device_data else None,
                    lch=[device_data["lch"]],
                ),
            )
    return specs


def get_switch_instance_results(
    config: dict,
    spec_input: dict[str, SpecInput],
) -> DataFrame:
    """Get each instance DCOP, Sizing and Electrical Parameters."""
    result = DataFrame(
        columns=[
            "id",
            "type",
            "objective",
            "vgs",
            "vsb",
            "wch",
            "lch",
            "ron",
            "csb",
            "cdb",
            "cgs",
            "cgd",
        ]
    )

    devices = {}
    device_types = [
        dt for dt in ["nch", "pch"] if dt in config and "lut_path" in config[dt]
    ]
    for device_type in device_types:
        devices[device_type] = Switch(
            lut_path=config[device_type]["lut_path"],
            device_type=DeviceType(device_type),
            ref_width=float(config[device_type]["ref_width"]),
        )

    for id in spec_input:
        spec = spec_input[id]
        if spec.device_type.value not in devices:
            warn(f"No configuration found for {spec.device_type}.")
            continue
        device = devices[spec.device_type.value]
        sizing = spec.sizing
        spec.sizing_spec.vds = [device.lut["vds"].min()]
        dcop = DcOp(
            vsb=spec.sizing_spec.vsb,
            vgs=spec.sizing_spec.vgs,
            vds=[device.lut["vds"].min()],
        )
        if spec.objective == InstanceObjective.SIZING:
            dcop, sizing = device.sizing(spec.sizing_spec, return_dcop=True)
        electric_model: ElectricModel = device.electric_model(dcop, sizing)
        result = concat(
            [
                result,
                DataFrame(
                    data={
                        "id": id,
                        "type": spec.device_type.value,
                        "objective": spec.objective.value,
                        "vgs": dcop.vgs,
                        "vds": dcop.vds,
                        "vsb": dcop.vsb,
                        "wch": sizing.wch,
                        "lch": sizing.lch,
                        "ron": electric_model.ron,
                        "cgs": electric_model.cgs,
                        "cgd": electric_model.cgd,
                        "cdb": electric_model.cdb,
                        "csb": electric_model.csb,
                    }
                ),
            ]
        )

    # fill missing values
    result["ids"] = None
    result["gm"] = None
    result["gds"] = None
    result["av"] = None
    result["ft"] = None
    result["fom_bw"] = None
    result["fom_nbw"] = None
    result["gmoverid"] = None
    result["cgg"] = None
    result["vng_rms"] = None
    return result


def format_results_dataframe(
    results: DataFrame, instance_config: InstanceConfig
) -> DataFrame:
    """Format results DataFrame to human-readable format."""

    formatted_results = DataFrame(
        data={
            "Type": results["type"],
            "ID": results["id"],
            "Vng,rms [uV]": results["vng_rms"].apply(
              lambda x: f"{x/1e-6:.4f}" if x is not None else None
            ),
            "Vgs [V]": results["vgs"].apply(lambda x: f"{x:.2f}"),
            "Vds [V]": results["vds"].apply(lambda x: f"{x:.2f}"),
            "Vsb [V]": results["vsb"].apply(lambda x: f"{x:.2f}"),
            "Wch [um]": results["wch"].apply(lambda x: f"{x/1e-6:.4f}"),
            "Lch [um]": results["lch"].apply(lambda x: f"{x/1e-6:.4f}"),
            "Cgg [fF]": results["cgg"].apply(
                lambda x: f"{x/1e-15:.4f}" if x is not None else None
            ),
            "Cdb [fF]": results["cdb"].apply(
                lambda x: f"{x/1e-15:.4f}" if x is not None else None
            ),
            "Cgs [fF]": results["cgs"].apply(
                lambda x: f"{x/1e-15:.4f}" if x is not None else None
            ),
            "Cgd [fF]": results["cgd"].apply(
                lambda x: f"{x/1e-15:.4f}" if x is not None else None
            ),
            "Csb [fF]": results["csb"].apply(
                lambda x: f"{x/1e-15:.4f}" if x is not None else None
            ),
            "Objective": results["objective"],
        }
    )

    if instance_config is InstanceConfig.DEVICE:
        formatted_results["Ids [uA]"] = results["ids"].apply(lambda x: f"{x/1e-6:.4e}")
        formatted_results["Gm/Id"] = results["gmoverid"].apply(lambda x: f"{x:.4f}")
        formatted_results["Gm [uS]"] = results["gm"].apply(lambda x: f"{x/1e-6:.4f}")
        formatted_results["Gds [uS]"] = results["gds"].apply(lambda x: f"{x/1e-6:.4f}")
        formatted_results["Av [V/V]"] = results["av"].apply(lambda x: f"{x:.4f}")
        formatted_results["Av [dB]"] = results["av"].apply(
            lambda x: f"{20*log10(x):.4f}"
        )
        formatted_results["Ft [GHz]"] = results["ft"].apply(lambda x: f"{x/1e9:.4f}")
        formatted_results["FOM Av*Ft [GHz]"] = results["fom_bw"].apply(
            lambda x: f"{x/1e9:.4f}"
        )
        formatted_results["FOM Ft*(Gm/Id) [GHz/V]"] = results["fom_nbw"].apply(
            lambda x: f"{x/1e9:.4f}"
        )
    if instance_config is InstanceConfig.SWITCH:
        if "ron" in results.columns:
            formatted_results["Ron [Ω]"] = results["ron"].apply(
                lambda x: f"{x:.4f}" if x is not None else None
            )
    return formatted_results


def app(args: dict, cfg: dict):
    if args["--verbose"]:
        print("Arguments:")
        pprint(args)

        print("Configuration:")
        pprint(cfg)

    if args["device"]:
        specs = parse_toml_device_input(args["--input"])
        results = get_device_instance_results(cfg, specs, return_noise=args["--noise"])
        formatted_results = format_results_dataframe(results, InstanceConfig.DEVICE)
        if args["--gui"]:
            gui = GuiApp(cfg)
            gui.show_results_table(results, title="Device Sizing Results Table")
        if args["--output"]:
            formatted_results.to_csv(args["--output"])
        else:
            print()
            print("Device Sizing Results:")
            print(formatted_results)
        return True
    elif args["moscap"]:
        if args["--noise"]: warn("Noise analysis is only available for 'device'.")
        specs = parse_toml_moscap_input(args["--input"])
        results = get_moscap_instance_results(cfg, specs)
        formatted_results = format_results_dataframe(results, InstanceConfig.MOSCAP)
        if args["--gui"]:
            gui = GuiApp(cfg)
            gui.show_results_table(results, title="Moscap Sizing Results Table")
        if args["--output"]:
            formatted_results.to_csv(args["--output"])
        else:
            print()
            print("Moscap Sizing Results:")
            print(formatted_results)
        return True
    elif args["switch"]:
        if args["--noise"]: warn("Noise analysis is only available for 'device'.")
        specs = parse_toml_switch_input(args["--input"])
        results = get_switch_instance_results(cfg, specs)
        formatted_results = format_results_dataframe(results, InstanceConfig.SWITCH)
        if args["--gui"]:
            gui = GuiApp(cfg)
            gui.show_results_table(results, title="Switch Sizing Results Table")
        if args["--output"]:
            formatted_results.to_csv(args["--output"])
        else:
            print()
            print("Switch Sizing Results:")
            print(formatted_results)
        return True

    if args["--gui"]:
        gui = GuiApp(cfg)
        gui.show_device_graphs(args)


def config(args):
    validate_all_configs()
    cfg: Config = Config()
    if args["--config"]:
        Config.CONFIG_SOURCES = FileSource(file=args["--config"])
    cfg: dict = cfg.model_dump()
    cfg = {k:v for k,v in cfg.items() if v is not None}
    return cfg


def main():
    args = docopt(__doc__, version="novaad 0.1")
    cfg = config(args)
    if not args["COMMAND_FILE"]:
        app(args, cfg)
    else:
        fp = Path(args["COMMAND_FILE"])
        assert fp.exists(), "Command file does not exist."
        assert fp.is_file(), "Command file must be a file."
        ext = fp.suffixes[0] if len(fp.suffixes) > 0 else ""
        assert ext in [".txt", ""], "Command file must be a text file."
        with open(args["COMMAND_FILE"], "r") as f:
            for line in f.readlines():
                line = line if len(line) > 0 else "-h"
                argv = line.split()
                try:
                    args = docopt(__doc__, argv=argv, version="novaad 0.1")
                    app(args, cfg)
                except DocoptExit:
                    continue
                except Exception as e:
                    print(f"Input:\t{line}\tOutput: {e}")


if __name__ == "__main__":
    main()
