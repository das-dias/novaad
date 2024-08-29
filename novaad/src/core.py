#from pydantic import BaseModel
from dataclasses import dataclass
from pandas import DataFrame, read_csv
from pandas.core.reshape.util import cartesian_product
from pathlib import Path
from typing import List, Tuple, Union, Optional
from enum import Enum, EnumMeta
from numpy import ndarray, array, abs, squeeze
from scipy.interpolate import interpn

import pdb

__cfg = "../novaad/cfg.yml"

def input_referred_flicker_noise_psd(): pass

def input_referred_thermal_noise_psd(): pass

def interpolate(bot_row, top_row):pass

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
class DcOp:
  vgs: Optional[Union[float, List]] = None
  vds: Optional[Union[float, List]] = None
  vsb: Optional[Union[float, List]] = None
  id: Optional[Union[float, List]] = None
  gm: Optional[Union[float, List]] = None
  gds: Optional[Union[float, List]] = None
  gmbs: Optional[Union[float, List]] = None
  wch: Optional[Union[float, List]] = None
  lch: Optional[Union[float, List]] = None
  
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
    lut_varmap: Optional[dict] = None,
    bsim4params_varmap: Optional[dict] = None,
    ref_width: float = 3e-6
  ):
    assert device_type in DeviceType, f"Invalid device type: {device_type}"
    self.type = DeviceType(device_type)
    self.path = lut_path
    self.lut = read_csv(lut_path)
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
      self.bsim4params = self.bsim4params[[col for col in self.bsim4params.columns if col in bsim4params_varmap.values()]]

  def find_nearest_unique(self, value: float, query) -> Tuple[float, int]:
    array = self.lut[query].unique()
    index = abs(array - value).argmin()
    return float(array[index]), int(index)
  
  def look_up(self, dc_op: Optional[DcOp]=None, **kwargs):
    interp_method = kwargs.get("interp_method", "pchip")
    usage_mode = kwargs.get("usage_mode", "forward")
    default = DcOp(
      vgs=self.lut["vgs"].unique().tolist(),
      vds=self.lut["vds"].mean(), 
      vsb=0.0, 
      lch=self.lut["lch"].min(),
    )
    if usage_mode == "graph": # from (vgs, vds, vsb, lch) to cross evaluation for graph plotting
      if dc_op is None: dc_op = default
      pass
    if usage_mode == "forward": # from (vgs, vds, vsb, lch, gm) to (id, gm, gds, gmbs, wch)
      pass
    if usage_mode == "backward": # from (wch, lch, vgs, vds, vsb) to (id, gm, gds, gmbs)
      pass
    
  def geometry(self, dc_op: DcOp):
    pass
  
  def thermal_noise_psd(self, dc_op: DcOp):
    pass
  
  def flicker_noise_psd(self, dc_op: DcOp, f:ndarray):
    pass
  
  def wave_vs_wave(self, dc_op: DcOp):
    pass
  
  

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
  
  device = Device(
    "/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv", 
    "/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_bsim4_params.csv", 
    "nch", lut_varmap=lut_varmap, bsim4params_varmap=bsim4_varmap
  )
  print(device.lut.head())
  print(device.bsim4params)
  
  dcop = DcOp(vgs=0.8, vds=0.8, vsb=0.0, lch=30e-9)
  print(dcop.to_array())
  print(dcop.to_df())
  
  print(device.find_nearest_unique(1.8, "vgs"))
  new_dcop = DcOp(
    vgs=device.lut["vgs"].unique().tolist(), 
    vds=device.lut["vds"].mean(), 
    vsb=0.0, 
    lch=device.lut["lch"].min()
  )
  print(new_dcop.to_array())
  print(new_dcop.to_df())
  
  