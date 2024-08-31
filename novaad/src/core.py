#from pydantic import BaseModel

import warnings


from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from enum import Enum, EnumMeta

warnings.simplefilter(action='ignore', category=DeprecationWarning)
from numpy import ndarray, array, abs, squeeze
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

from pprint import pprint

import pdb

warnings.simplefilter(action='ignore', category=FutureWarning)
from pandas import DataFrame, read_csv, concat
from pandas.core.reshape.util import cartesian_product

__cfg = "../novaad/cfg.yml"

def input_referred_flicker_noise_psd(): pass

def input_referred_thermal_noise_psd(): pass
  

DeviceLutPath = Union[str, Path]

class MetaEnum(EnumMeta):
  def __contains__(cls, item):
    try:
      cls(item)
    except ValueError:
      return False
    return True    

class BaseEnum(Enum, metaclass=MetaEnum):
    pass

class DeviceType(BaseEnum):
  NMOS = "nch"
  PMOS = "pch"

@dataclass
class BaseParametricObject:
  
  def to_df(self) -> DataFrame:
    # generate list of lists
    def to_list(val):
      if isinstance(val, list):
        return val
      return [val]  
    lists= [to_list(getattr(self, attr)) for attr in self.__annotations__ if getattr(self, attr) is not None] 
    prod = cartesian_product(lists)
    columns = [attr for attr in self.__annotations__ if getattr(self, attr) is not None]
    return DataFrame({
      col: prod[columns.index(col)]
      for col in columns
    })
    
  def __array__(self) -> ndarray:
    return self.to_df().to_numpy()

  def to_array(self) -> ndarray:
    return self.__array__()

@dataclass
class Sizing(BaseParametricObject):
  wch: Optional[Union[float, List[float]]] = None
  lch: Optional[Union[float, List[float]]] = None
  
@dataclass
class DcOp(BaseParametricObject):
  vgs: Optional[Union[float, List[float]]] = None
  vds: Optional[Union[float, List[float]]] = None
  vsb: Optional[Union[float, List[float]]] = None
  ids: Optional[Union[float, List[float]]] = None

@dataclass
class SizingSpecification(BaseParametricObject):
  vgs: Optional[Union[float, List[float]]] = None
  vds: Optional[Union[float, List[float]]] = None
  vsb: Optional[Union[float, List[float]]] = None
  ids: Optional[Union[float, List[float]]] = None
  # for simplicity of implementation, include specifications in DcOp as control knobs
  lch: Optional[Union[float, List[float]]] = None
  gmoverid: Optional[Union[float, List[float]]] = None
  gm: Optional[Union[float, List[float]]] = None

@dataclass
class ElectricModel(BaseParametricObject):
  jd: Optional[Union[float, List[float]]] = None
  ids: Optional[Union[float, List[float]]] = None
  gm: Optional[Union[float, List[float]]] = None
  gds: Optional[Union[float, List[float]]] = None
  cgg: Optional[Union[float, List[float]]] = None
  cgs: Optional[Union[float, List[float]]] = None
  cgd: Optional[Union[float, List[float]]] = None
  cdb: Optional[Union[float, List[float]]] = None
  csb: Optional[Union[float, List[float]]] = None
  ft: Optional[Union[float, List[float]]] = None
  av: Optional[Union[float, List[float]]] = None
  
@dataclass
class Device:
  lut: DataFrame
  bsim4params: DataFrame
  type: DeviceType
  ref_width: float

  def __init__(
    self, 
    lut_path: DeviceLutPath, 
    bsim4params_path: Optional[DeviceLutPath] = None, 
    device_type: str = "nch", 
    ref_width: float = 10e-6,
    lut_varmap: Optional[dict] = None,
    bsim4params_varmap: Optional[dict] = None,
  ):
    assert device_type in DeviceType, f"Invalid device type: {device_type}"
    self.type = DeviceType(device_type)
    self.path = lut_path
    self.lut = read_csv(lut_path)
    self.bsim4params = None
    self.ref_width = ref_width
    # remove columns with all NaN values
    self.lut = self.lut.dropna(axis=1, how="all")
    if bsim4params_path:
      self.bsim4params = read_csv(bsim4params_path)
      self.bsim4params = self.bsim4params.dropna(axis=1, how="all")
    if lut_varmap:
      self.lut = self.lut.rename(columns=lut_varmap)
      # remove columns that are not in the lut_varmap
      self.lut = self.lut[[col for col in self.lut.columns if col in lut_varmap.values()]]
    if bsim4params_varmap and bsim4params_path:
      self.bsim4params = self.bsim4params.rename(columns=bsim4params_varmap)
      self.bsim4params = self.bsim4params[[
        col for col in self.bsim4params.columns if col in bsim4params_varmap.values()]]
      
  def find_nearest_unique(self, value: float, query) -> Tuple[float, int]:
    array = self.lut[query].unique()
    index = abs(array - value).argmin()
    return float(array[index]), int(index)
  
  # FIXME: add possibility of simply 
  # returning LUT query results without a query in mind
  def look_up(
    self, 
    xcols: List[str], 
    ycols: List[str], 
    target: Dict[str, List], 
    return_xy: bool = False, 
    **kwargs) -> DataFrame:
    """Look up values in the LUT using an unstructured grid data interpolation

    Args:
        xcols (List[str]): Input data columns
        ycols (List[str]): Output data columns
        target (Dict[str, List]): Points where to sample the output data

    Returns:
        DataFrame: Interpolated output data DataFrame(ydata, columns=ycols)
    """
    assert all([col in self.lut.columns for col in xcols]), "Invalid xcols. Possible values: {self.lut.columns}"
    assert all([col in target for col in xcols]), f"Invalid target columns. Possible values: {xcols}"
    
    ycols = [col for col in ycols if col in self.lut.columns]
    
    interp_method = kwargs.get("interp_method", "pchip")
    order = kwargs.get("order", 2)
    interp_mode = kwargs.get("interp_mode", "default") # default, griddata
    distance_metric = kwargs.get("distance_metric", "euclidean")
    dist_metric_kwargs = kwargs.get("dist_metric_kwargs", {})
    yxcols = list(set(ycols+xcols))
    newdf = self.lut[yxcols]
    target_points = squeeze(array(cartesian_product(list(target.values()))))
    if len(target_points.shape) == 1:
      target_points = target_points.reshape(-1, 1)
    target_df = DataFrame(target_points.T, columns=xcols)
    target_df = target_df[xcols]
    # assert every target column is between the min and max of the LUT
    assert all(
      [target_df[col].between(
        self.lut[col].min(), self.lut[col].max()).all() 
        for col in xcols]
    ), "Target points out of LUT bounds"

    if interp_mode == "default":
      distances = cdist(target_df.to_numpy(), newdf[xcols].to_numpy(), 
        metric=distance_metric, **dist_metric_kwargs)
      idx = distances.argpartition(2, axis=1)[:, :2]
      bot_row = newdf.iloc[idx[:, 0]]
      top_row = newdf.iloc[idx[:, 1]]
      if bot_row.equals(top_row):
        return bot_row[ycols] if not return_xy else bot_row[yxcols]
      newdf = DataFrame(columns=newdf.columns)
      interpolated_idxs = []
      for i in range(len(target_df)):
        newdf = concat([newdf, bot_row.iloc[[i]], target_df.iloc[[i]], top_row.iloc[[i]]])
        interpolated_idxs.append(i*3+1)
      newdf = newdf.reset_index()
      newdf = newdf.drop(columns=["index"])
      newdf = newdf.interpolate(method=interp_method, order=order)
      # return only the interpolated values
      newdf = newdf.iloc[interpolated_idxs]
      return newdf[yxcols] if return_xy else newdf[ycols]
      
    elif interp_mode == "griddata":
      # Matrix operations to perform unstructured grid data interpolation... it's possibly much faster
      raise NotImplementedError
    
    assert False, "Invalid interp_mode. Possible values: default, pandas, griddata"
  
  def sizing(self, sizing_spec: Optional[SizingSpecification], return_dcop=False, **kwargs) -> Union[Tuple[DcOp,Sizing], Sizing]:
    """Device sizing from DC operating point and target electric parameters
    Forward propagaton of the flow of Gm/Id method.
    Args:
        dcop (DcOp): DC operating point

    Returns:
        DataFrame: Resulting device sizing and electrical parameters
    """
    default_sizing = SizingSpecification(
      vgs=self.lut["vgs"].mean(),
      vds=self.lut["vds"].mean(),
      vsb=0.0,
      lch=self.lut["lch"].min(),
      gmoverid=10.0,
      gm=1e-3,
    )
    
    if sizing_spec is None:
      sizing_spec = default_sizing
    
    assert all([col in self.lut.columns for col in sizing_spec.__annotations__]), "Invalid sizing_spec"
    assert (sizing_spec.vgs is not None), "vgs is required"
    assert (sizing_spec.vds is not None), "vds is required"
    assert (sizing_spec.vsb is not None), "vsb is required"
    assert (sizing_spec.lch is not None), "lch is required"
    assert (sizing_spec.gmoverid is not None), "gmoverid is required" 
    assert (sizing_spec.gm is not None)\
      or (sizing_spec.ids is not None), "gm or id is required"
    
    kwargs = {
      "interp_method": kwargs.get("interp_method", "pchip"),
      "interp_mode": kwargs.get("interp_mode", "default"),
    }
    ycols = ["jd", "lch"]
    xcols = ["vgs", "vds", "vsb", "lch", "gmoverid"]
    target = {
      "vgs": sizing_spec.vgs if isinstance(sizing_spec.vgs, list) else [sizing_spec.vgs],
      "vds": sizing_spec.vds if isinstance(sizing_spec.vds, list) else [sizing_spec.vds],
      "vsb": sizing_spec.vsb if isinstance(sizing_spec.vsb, list) else [sizing_spec.vsb],
      "lch": sizing_spec.lch if isinstance(sizing_spec.lch, list) else [sizing_spec.lch], # I included channel length to replace instrinsic gain spec. Using Av would require early voltage Va to also be extracted.
      "gmoverid": sizing_spec.gmoverid if isinstance(sizing_spec.gmoverid, list) else [sizing_spec.gmoverid]
    }
    closest_target = self.look_up(xcols, ycols, target, return_xy=True, **kwargs)
    sizing = Sizing(lch=closest_target["lch"].values)
    dcop = DcOp(
      vgs=sizing_spec.vgs, 
      vds=sizing_spec.vds, 
      vsb=sizing_spec.vsb
    )
    if sizing_spec.gm is None:
      dcop.ids = sizing_spec.ids
      sizing.wch = sizing_spec.ids / closest_target["jd"].values
    else:
      dcop.ids = sizing_spec.gm / closest_target["gmoverid"].values
      sizing.wch = dcop.ids / closest_target["jd"].values
    return (dcop, sizing) if return_dcop else sizing
  
  def electric_model(self, dcop: DcOp, sizing: Sizing, **kwargs) -> ElectricModel:
    """Electric model from from device sizing and DC Operating point
    Backwards propagaton of the flow of Gm/Id method.
    Args:
        sizing (Dict): Device sizing
        dcop (DcOp): DC operating point containing voltages and currents

    Returns:
        Dict: Electrical model parameters
    """
    wch_ratio = sizing.wch / self.ref_width
    target_jd = array([dcop.ids]).flatten() / array([sizing.wch]).flatten() 
    kwargs = {
      "interp_method": kwargs.get("interp_method", "pchip"),
      "interp_mode": kwargs.get("interp_mode", "default"),
    }
    ycols = ["jd", "gm", "gds", "cgg", "cgs", "cgd", "cdb", "csb", "ft", "av"]
    xcols = ["vgs", "vds", "vsb", "lch", "jd"]
    target = {
      "vgs": dcop.vgs if isinstance(dcop.vgs, list) else [dcop.vgs],
      "vds": dcop.vds if isinstance(dcop.vds, list) else [dcop.vds],
      "vsb": dcop.vsb if isinstance(dcop.vsb, list) else [dcop.vsb],
      "lch": sizing.lch if isinstance(sizing.lch, list) else [sizing.lch],
      "jd": target_jd.tolist()
    }
    reference_electric_model = self.look_up(xcols, ycols, target, **kwargs)
    pdb.set_trace()
    #NOTE: simple model assuming linear scaling with channel width
    #FIXME: Use bsim4 model to effectively scale the device
    electric_model = ElectricModel()
    for col in [col for col in electric_model.__annotations__ if col in reference_electric_model.columns]:
      electric_model.__setattr__(col, reference_electric_model[col].values)
      if (col.startswith('c') or col.startswith('g') ):# capacitances or conductances
        electric_model.__setattr__(col, electric_model.__getattribute__(col)*wch_ratio)
    return electric_model
  
  def thermal_noise_psd(self, dcop: DcOp, sizing: Sizing, **kwargs):
    raise NotImplementedError
  
  def flicker_noise_psd(self, dcop: DcOp, sizing: Sizing, f: ndarray, **kwargs):
    raise NotImplementedError
  
  def noise_summary(self, dcop: DcOp, sizing: Sizing, **kwargs):
    raise NotImplementedError
  
  def wave_vs_wave(self, dc_op: DcOp):
    raise NotImplementedError
  

# Examples

def test_device(lut_varmap):
  device = Device(
    "/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv", 
    device_type="nch", lut_varmap=lut_varmap
  )
  print(device.lut.head())

def test_device_dc_op(lut_varmap):
  device = Device(
    "/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv", 
    device_type="nch", lut_varmap=lut_varmap
  )
  
  dcop = DcOp(vgs=0.5, vds=0.6, vsb=0.0, ids=1e-3)
  print(dcop.to_array())
  print(dcop.to_df())
  
  print(device.find_nearest_unique(1.8, "vgs"))
  new_dcop = DcOp(
    vgs=device.lut["vgs"].unique().tolist(), 
    vds=device.lut["vds"].mean(), 
    vsb=0.0,
  )
  print(new_dcop.to_array())
  print(new_dcop.to_df())

def test_device_look_up(lut_varmap):
  device = Device(
    "/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv",
    device_type="nch", lut_varmap=lut_varmap
  )
  input_cols = [ "vgs", "vds", "vsb", "lch", "gmoverid"]
  output_cols = ["av", "jd", "ft"]
  target = {
    "vgs": [device.lut["vgs"].mean()],
    "vds": [0.9],
    "vsb": [0.0],
    "lch": [device.lut["lch"].min()*1],
    "gmoverid": [1.0, 27.0]
  }
  pprint(target)
  print("Interpolating...")
  
  kwargs = {
    "interp_method": "nearest",
    "interp_mode": "default",
  }
  print("Default:")
  print("Nearest:")
  row = device.look_up(input_cols, output_cols, target,return_xy=True, **kwargs)
  print(row)
  
  kwargs = {
    "interp_method": "linear",
    "interp_mode": "default",
  }
  print("Linear:")
  row = device.look_up(input_cols, output_cols, target,return_xy=True, **kwargs)
  print(row)
  
  kwargs = {
    "interp_method": "pchip",
    "interp_mode": "default",
  }
  print("PCHIP:")
  row = device.look_up(input_cols, output_cols, target,return_xy=True, **kwargs)
  print(row)

def test_device_sizing(lut_varmap):
  device = Device(
    "/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv",  
    device_type="nch", lut_varmap=lut_varmap
  )
  
  sizing_spec = SizingSpecification(
    vgs=0.5, vds=0.6, vsb=0.0,
    lch=device.lut["lch"].min(), 
    gmoverid=10.0,
    gm=1e-3
  )

  print(sizing_spec.to_df())
  
  sizing = device.sizing(sizing_spec)
  
  print(sizing.to_df())  
  
  sizing_spec = SizingSpecification(
    vgs=0.5, vds=0.6, vsb=0.0,
    lch=device.lut["lch"].min(), 
    gmoverid=24.0,
    gm=1e-3
  )
  
  print(sizing_spec.to_df())
  
  sizing = device.sizing(sizing_spec)
  
  print(sizing.to_df())
  
def test_device_electric_model(lut_varmap):
  device = Device(
    "/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv",
    device_type="nch", lut_varmap=lut_varmap
  )
  
  sizing_spec = SizingSpecification(
    vgs=0.5, vds=0.6, vsb=0.0,
    lch=device.lut["lch"].min(), 
    gmoverid=16.0,
    ids=1e-3
  )

  print(sizing_spec.to_df())
  
  dcop, sizing = device.sizing(sizing_spec, return_dcop=True)
  
  print(dcop.to_df())
  print(sizing.to_df())
  
  electric_model = device.electric_model(dcop, sizing)
  
  print(electric_model.to_df())

if __name__ == "__main__":
  lut_varmap = {
    "vgs": "vgs_n",
    "lch": "length_wave",
    "weff": "weff_n",
    "leff": "leff_n",
    "vgseff": "vgseff_n",
    "vds": "vds_n",
    "vdsat": "vdsat_n",
    "vsb": "vsb_n",
    "vbseff": "vbseff_n",
    "vth": "vth_n",
    "qg": "qg_n",
    "qd": "qd_n",
    "qs": "qs_n",
    "gm": "gm_n",
    "gds": "gds_n",
    "ids": "id_n",
    "jd": "jd_n",
    "cgs": "cgs_n",
    "cgd": "cgd_n",
    "cdb": "cdb_n",
    "csb": "csb_n",
    "cgg": "cgg_n",
    "gmoverid": "gmoverid_n",
    "ft": "ft_n",
    "av": "av_n",
    "fom_bw": "FoM_BW_n",
    "fom_nbw": "FoM_Nbw_n"
  }
  lut_varmap = {v: k for k, v in lut_varmap.items()}

  bsim4_varmap = {
    'u0': 'U0_n',
    'lp': 'LP_n',
    'uq': 'UA_n',
    'vfb': 'VFB_n',
    'phis': 'PHIs_n',
    'eu': 'EU_n',
    'c0': 'C0_n',
    'uc': 'UC_n',
    'ud': 'UD_n',
    'tox': 'TOXE_n',
    'epsrox': 'EPSROX_n',
    'af': 'AF_n',
    'ef': 'EF_n',
    'ntnoi': 'NTNOI_n'
}
  bsim4_varmap = {v: k for k, v in bsim4_varmap.items()}
  
  #test_device(lut_varmap)
  
  #test_device_dc_op(lut_varmap)
  
  #test_device_look_up(lut_varmap)
  
  #test_device_sizing(lut_varmap)
  
  test_device_electric_model(lut_varmap)
  
  