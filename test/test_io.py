from pathlib import Path

import numpy as np
import pytest
from pandas import DataFrame, read_csv

from novaad import Device
from schema.io import csv_to_proto, from_bytes, proto_to_csv, read_proto, to_bytes

CSV = Path(__file__).parent / "data" / "test_nch_lut_renamed.csv"


def test_csv_proto_round_trip(tmp_path):
    pb = tmp_path / "nch.pb"
    csv_to_proto(CSV, pb, "pch", 5e-6)
    lut, device_type, ref_width = read_proto(pb)
    ref = read_csv(CSV)
    assert (device_type, ref_width) == ("pch", 5e-6)
    assert set(lut.columns) == set(ref.columns)
    assert np.allclose(lut[ref.columns], ref, rtol=1e-6, atol=0)
    assert pb.stat().st_size < CSV.stat().st_size

    back = tmp_path / "back.csv"
    proto_to_csv(pb, back)
    assert np.allclose(read_csv(back)[ref.columns], ref, rtol=1e-6, atol=0)


def test_device_reads_proto(tmp_path):
    pb = tmp_path / "nch.pb"
    csv_to_proto(CSV, pb)
    from_csv, from_pb = Device(CSV, "nch"), Device(pb, "nch")
    assert np.allclose(from_pb.lut[from_csv.lut.columns], from_csv.lut, rtol=1e-6, atol=0)


def test_unknown_columns_raise():
    with pytest.raises(ValueError, match="not in the protobuf schema"):
        to_bytes(DataFrame({"vgs": [0.1], "bogus": [1.0]}))


def test_partial_columns():
    lut, _, _ = from_bytes(to_bytes(DataFrame({"vgs": [0.1, 0.2], "ids": [1e-6, 2e-6]})))
    assert list(lut.columns) == ["vgs", "ids"]


def test_grid_values_exact():
    lut, _, _ = from_bytes(to_bytes(DataFrame({"vgs": [0.4875, 0.025], "lch": [1.3e-7, 5e-7]})))
    assert lut["vgs"].tolist() == [0.4875, 0.025] and lut["lch"].tolist() == [1.3e-7, 5e-7]
