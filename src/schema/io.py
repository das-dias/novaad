"""Convert device LUTs between DataFrames / CSV and the compact protobuf format.

A protobuf LUT file is a zlib-compressed, serialized `CmosDevice` (see protos/lut.proto) holding the
device type, the reference width and the LUT itself, stored column by column as packed float32 arrays.
That is about 6x smaller than the CSV, with a relative error of about 1e-7; grid values round-trip exactly.

    python -m schema.io luts/sg13_lv_nch.csv luts/sg13_lv_nch.pb --type nch --ref-width 10e-6
    python -m schema.io luts/sg13_lv_nch.pb luts/sg13_lv_nch.csv
"""

import argparse
import zlib
from pathlib import Path

import numpy as np
from pandas import DataFrame, read_csv

from schema.lut_pb2 import CmosDevice, DeviceType, Lut

COLUMNS = [f.name for f in Lut.DESCRIPTOR.fields]
DEVICE_TYPES = {"nch": DeviceType.NCH, "pch": DeviceType.PCH}


def lut_to_proto(lut: DataFrame) -> Lut:
    """A LUT DataFrame as a columnar `Lut` message; columns missing from the DataFrame are left empty."""
    unknown = set(lut.columns) - set(COLUMNS)
    if unknown:
        raise ValueError(f"LUT columns not in the protobuf schema: {sorted(unknown)}")
    msg = Lut()
    for name in lut.columns:
        getattr(msg, name).extend(lut[name].to_numpy(dtype=np.float32))
    return msg


def lut_from_proto(msg: Lut) -> DataFrame:
    """The non-empty columns of a `Lut` message, as float64, in schema order.

    Each float32 is widened through its shortest decimal form, so grid values come back exact
    (0.4875, not 0.48749999701976776): lookups that match the vgs/vds grid keep working.
    """
    return DataFrame({name: np.asarray(getattr(msg, name), dtype=np.float32).astype(str).astype(np.float64)
                      for name in COLUMNS if len(getattr(msg, name))})


def to_bytes(lut: DataFrame, device_type: str = "nch", ref_width: float = 10e-6) -> bytes:
    device = CmosDevice(type=DEVICE_TYPES[device_type], ref_width=ref_width, lut=lut_to_proto(lut))
    return zlib.compress(device.SerializeToString(), 9)


def from_bytes(data: bytes) -> tuple[DataFrame, str, float]:
    """(LUT, device type "nch"/"pch", reference width) of a compressed `CmosDevice`."""
    device = CmosDevice.FromString(zlib.decompress(data))
    if device.WhichOneof("lut_source") != "lut":
        raise ValueError("CmosDevice has no inline LUT")
    device_type = next(k for k, v in DEVICE_TYPES.items() if v == device.type)
    return lut_from_proto(device.lut), device_type, device.ref_width


def write_proto(path, lut: DataFrame, device_type: str = "nch", ref_width: float = 10e-6) -> None:
    Path(path).write_bytes(to_bytes(lut, device_type, ref_width))


def read_proto(path) -> tuple[DataFrame, str, float]:
    return from_bytes(Path(path).read_bytes())


def csv_to_proto(csv_path, pb_path, device_type: str = "nch", ref_width: float = 10e-6) -> None:
    write_proto(pb_path, read_csv(csv_path), device_type, ref_width)


def proto_to_csv(pb_path, csv_path) -> None:
    lut, _, _ = read_proto(pb_path)
    lut.to_csv(csv_path, index=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Convert a device LUT between CSV and protobuf (.pb).")
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--type", choices=sorted(DEVICE_TYPES), default="nch", help="device type (CSV -> .pb)")
    parser.add_argument("--ref-width", type=float, default=10e-6, help="LUT reference width [m] (CSV -> .pb)")
    args = parser.parse_args(argv)
    suffixes = (args.src.suffix, args.dst.suffix)
    if suffixes == (".csv", ".pb"):
        csv_to_proto(args.src, args.dst, args.type, args.ref_width)
    elif suffixes == (".pb", ".csv"):
        proto_to_csv(args.src, args.dst)
    else:
        parser.error("convert .csv -> .pb or .pb -> .csv")
    print(f"{args.src} ({args.src.stat().st_size / 1e6:.2f} MB) -> {args.dst} ({args.dst.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
