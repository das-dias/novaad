from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DeviceType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEVICE_TYPE_UNSPECIFIED: _ClassVar[DeviceType]
    NCH: _ClassVar[DeviceType]
    PCH: _ClassVar[DeviceType]
DEVICE_TYPE_UNSPECIFIED: DeviceType
NCH: DeviceType
PCH: DeviceType

class Lut(_message.Message):
    __slots__ = ("vgs", "vds", "vsb", "lch", "ids", "gm", "gds", "cgg", "cgs", "cgd", "cdb", "csb", "gmoverid", "jd", "ft", "av", "ron", "fom_bw")
    VGS_FIELD_NUMBER: _ClassVar[int]
    VDS_FIELD_NUMBER: _ClassVar[int]
    VSB_FIELD_NUMBER: _ClassVar[int]
    LCH_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    GM_FIELD_NUMBER: _ClassVar[int]
    GDS_FIELD_NUMBER: _ClassVar[int]
    CGG_FIELD_NUMBER: _ClassVar[int]
    CGS_FIELD_NUMBER: _ClassVar[int]
    CGD_FIELD_NUMBER: _ClassVar[int]
    CDB_FIELD_NUMBER: _ClassVar[int]
    CSB_FIELD_NUMBER: _ClassVar[int]
    GMOVERID_FIELD_NUMBER: _ClassVar[int]
    JD_FIELD_NUMBER: _ClassVar[int]
    FT_FIELD_NUMBER: _ClassVar[int]
    AV_FIELD_NUMBER: _ClassVar[int]
    RON_FIELD_NUMBER: _ClassVar[int]
    FOM_BW_FIELD_NUMBER: _ClassVar[int]
    vgs: _containers.RepeatedScalarFieldContainer[float]
    vds: _containers.RepeatedScalarFieldContainer[float]
    vsb: _containers.RepeatedScalarFieldContainer[float]
    lch: _containers.RepeatedScalarFieldContainer[float]
    ids: _containers.RepeatedScalarFieldContainer[float]
    gm: _containers.RepeatedScalarFieldContainer[float]
    gds: _containers.RepeatedScalarFieldContainer[float]
    cgg: _containers.RepeatedScalarFieldContainer[float]
    cgs: _containers.RepeatedScalarFieldContainer[float]
    cgd: _containers.RepeatedScalarFieldContainer[float]
    cdb: _containers.RepeatedScalarFieldContainer[float]
    csb: _containers.RepeatedScalarFieldContainer[float]
    gmoverid: _containers.RepeatedScalarFieldContainer[float]
    jd: _containers.RepeatedScalarFieldContainer[float]
    ft: _containers.RepeatedScalarFieldContainer[float]
    av: _containers.RepeatedScalarFieldContainer[float]
    ron: _containers.RepeatedScalarFieldContainer[float]
    fom_bw: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, vgs: _Optional[_Iterable[float]] = ..., vds: _Optional[_Iterable[float]] = ..., vsb: _Optional[_Iterable[float]] = ..., lch: _Optional[_Iterable[float]] = ..., ids: _Optional[_Iterable[float]] = ..., gm: _Optional[_Iterable[float]] = ..., gds: _Optional[_Iterable[float]] = ..., cgg: _Optional[_Iterable[float]] = ..., cgs: _Optional[_Iterable[float]] = ..., cgd: _Optional[_Iterable[float]] = ..., cdb: _Optional[_Iterable[float]] = ..., csb: _Optional[_Iterable[float]] = ..., gmoverid: _Optional[_Iterable[float]] = ..., jd: _Optional[_Iterable[float]] = ..., ft: _Optional[_Iterable[float]] = ..., av: _Optional[_Iterable[float]] = ..., ron: _Optional[_Iterable[float]] = ..., fom_bw: _Optional[_Iterable[float]] = ...) -> None: ...

class CmosDevice(_message.Message):
    __slots__ = ("type", "ref_width", "lut_path", "lut")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    REF_WIDTH_FIELD_NUMBER: _ClassVar[int]
    LUT_PATH_FIELD_NUMBER: _ClassVar[int]
    LUT_FIELD_NUMBER: _ClassVar[int]
    type: DeviceType
    ref_width: float
    lut_path: str
    lut: Lut
    def __init__(self, type: _Optional[_Union[DeviceType, str]] = ..., ref_width: _Optional[float] = ..., lut_path: _Optional[str] = ..., lut: _Optional[_Union[Lut, _Mapping]] = ...) -> None: ...

class Sizing(_message.Message):
    __slots__ = ("lch", "wch")
    LCH_FIELD_NUMBER: _ClassVar[int]
    WCH_FIELD_NUMBER: _ClassVar[int]
    lch: _containers.RepeatedScalarFieldContainer[float]
    wch: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, lch: _Optional[_Iterable[float]] = ..., wch: _Optional[_Iterable[float]] = ...) -> None: ...

class DcOp(_message.Message):
    __slots__ = ("vgs", "vds", "vsb", "ids")
    VGS_FIELD_NUMBER: _ClassVar[int]
    VDS_FIELD_NUMBER: _ClassVar[int]
    VSB_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    vgs: _containers.RepeatedScalarFieldContainer[float]
    vds: _containers.RepeatedScalarFieldContainer[float]
    vsb: _containers.RepeatedScalarFieldContainer[float]
    ids: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, vgs: _Optional[_Iterable[float]] = ..., vds: _Optional[_Iterable[float]] = ..., vsb: _Optional[_Iterable[float]] = ..., ids: _Optional[_Iterable[float]] = ...) -> None: ...

class DeviceSizingSpecification(_message.Message):
    __slots__ = ("vds", "vsb", "lch", "vgs", "gmoverid", "ids", "gm")
    VDS_FIELD_NUMBER: _ClassVar[int]
    VSB_FIELD_NUMBER: _ClassVar[int]
    LCH_FIELD_NUMBER: _ClassVar[int]
    VGS_FIELD_NUMBER: _ClassVar[int]
    GMOVERID_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    GM_FIELD_NUMBER: _ClassVar[int]
    vds: _containers.RepeatedScalarFieldContainer[float]
    vsb: _containers.RepeatedScalarFieldContainer[float]
    lch: _containers.RepeatedScalarFieldContainer[float]
    vgs: _containers.RepeatedScalarFieldContainer[float]
    gmoverid: _containers.RepeatedScalarFieldContainer[float]
    ids: _containers.RepeatedScalarFieldContainer[float]
    gm: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, vds: _Optional[_Iterable[float]] = ..., vsb: _Optional[_Iterable[float]] = ..., lch: _Optional[_Iterable[float]] = ..., vgs: _Optional[_Iterable[float]] = ..., gmoverid: _Optional[_Iterable[float]] = ..., ids: _Optional[_Iterable[float]] = ..., gm: _Optional[_Iterable[float]] = ...) -> None: ...

class MoscapSizingSpecification(_message.Message):
    __slots__ = ("vgs", "vsb", "lch", "cgg", "vds")
    VGS_FIELD_NUMBER: _ClassVar[int]
    VSB_FIELD_NUMBER: _ClassVar[int]
    LCH_FIELD_NUMBER: _ClassVar[int]
    CGG_FIELD_NUMBER: _ClassVar[int]
    VDS_FIELD_NUMBER: _ClassVar[int]
    vgs: _containers.RepeatedScalarFieldContainer[float]
    vsb: _containers.RepeatedScalarFieldContainer[float]
    lch: _containers.RepeatedScalarFieldContainer[float]
    cgg: _containers.RepeatedScalarFieldContainer[float]
    vds: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, vgs: _Optional[_Iterable[float]] = ..., vsb: _Optional[_Iterable[float]] = ..., lch: _Optional[_Iterable[float]] = ..., cgg: _Optional[_Iterable[float]] = ..., vds: _Optional[_Iterable[float]] = ...) -> None: ...

class SwitchSizingSpecification(_message.Message):
    __slots__ = ("vgs", "vsb", "lch", "ron", "vds")
    VGS_FIELD_NUMBER: _ClassVar[int]
    VSB_FIELD_NUMBER: _ClassVar[int]
    LCH_FIELD_NUMBER: _ClassVar[int]
    RON_FIELD_NUMBER: _ClassVar[int]
    VDS_FIELD_NUMBER: _ClassVar[int]
    vgs: _containers.RepeatedScalarFieldContainer[float]
    vsb: _containers.RepeatedScalarFieldContainer[float]
    lch: _containers.RepeatedScalarFieldContainer[float]
    ron: _containers.RepeatedScalarFieldContainer[float]
    vds: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, vgs: _Optional[_Iterable[float]] = ..., vsb: _Optional[_Iterable[float]] = ..., lch: _Optional[_Iterable[float]] = ..., ron: _Optional[_Iterable[float]] = ..., vds: _Optional[_Iterable[float]] = ...) -> None: ...

class ElectricModel(_message.Message):
    __slots__ = ("jd", "ids", "gm", "gds", "cgg", "cgs", "cgd", "cdb", "csb", "ft", "av", "ron")
    JD_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    GM_FIELD_NUMBER: _ClassVar[int]
    GDS_FIELD_NUMBER: _ClassVar[int]
    CGG_FIELD_NUMBER: _ClassVar[int]
    CGS_FIELD_NUMBER: _ClassVar[int]
    CGD_FIELD_NUMBER: _ClassVar[int]
    CDB_FIELD_NUMBER: _ClassVar[int]
    CSB_FIELD_NUMBER: _ClassVar[int]
    FT_FIELD_NUMBER: _ClassVar[int]
    AV_FIELD_NUMBER: _ClassVar[int]
    RON_FIELD_NUMBER: _ClassVar[int]
    jd: _containers.RepeatedScalarFieldContainer[float]
    ids: _containers.RepeatedScalarFieldContainer[float]
    gm: _containers.RepeatedScalarFieldContainer[float]
    gds: _containers.RepeatedScalarFieldContainer[float]
    cgg: _containers.RepeatedScalarFieldContainer[float]
    cgs: _containers.RepeatedScalarFieldContainer[float]
    cgd: _containers.RepeatedScalarFieldContainer[float]
    cdb: _containers.RepeatedScalarFieldContainer[float]
    csb: _containers.RepeatedScalarFieldContainer[float]
    ft: _containers.RepeatedScalarFieldContainer[float]
    av: _containers.RepeatedScalarFieldContainer[float]
    ron: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, jd: _Optional[_Iterable[float]] = ..., ids: _Optional[_Iterable[float]] = ..., gm: _Optional[_Iterable[float]] = ..., gds: _Optional[_Iterable[float]] = ..., cgg: _Optional[_Iterable[float]] = ..., cgs: _Optional[_Iterable[float]] = ..., cgd: _Optional[_Iterable[float]] = ..., cdb: _Optional[_Iterable[float]] = ..., csb: _Optional[_Iterable[float]] = ..., ft: _Optional[_Iterable[float]] = ..., av: _Optional[_Iterable[float]] = ..., ron: _Optional[_Iterable[float]] = ...) -> None: ...
