from dataclasses import dataclass
from pandas import DataFrame, read_csv, write_csv
from pathlib import Path
from typing import List, Tuple, Union, Enum, Optional
from numpy import ndarray

def input_referred_flicker_noise_psd(): pass

def input_referred_thermal_noise_psd(): pass

def interpolate(bot_row, top_row):pass

DeviceLutPath = Union[str, Path]

class DeviceType(Enum):
  NMOS = "nch"
  PMOS = "pch"
  

@dataclass
class DcOp(object):
  vgs: Optional[Union[float, ndarray]] = None
  vds: Optional[Union[float, ndarray]] = None
  vbs: Optional[Union[float, ndarray]] = None
  id: Optional[Union[float, ndarray]] = None
  gm: Optional[Union[float, ndarray]] = None
  gds: Optional[Union[float, ndarray]] = None
  gmbs: Optional[Union[float, ndarray]] = None
  width: Optional[Union[float, ndarray]] = None
  length: Optional[Union[float, ndarray]] = None
  ref_width: Optional[float] = None

@dataclass
class Device(object):
  lut: DataFrame
  bsim4params: DataFrame
  type: DeviceType
  ref_width: float

  def __init__(self, lut_path: DeviceLutPath, bsim4params_path: DeviceLutPath, device_type: str):
    assert device_type in DeviceType.__members__.values(), f"Invalid device type: {device_type}"
    self.type = DeviceType(device_type)
    self.path = lut_path
    self.lut = read_csv(lut_path)
    self.bsim4params = read_csv(bsim4params_path)
    
  def look_up(self, dc_op: DcOp):
    pass
  
  def predict_noise(self, dc_op: DcOp):
    pass
  
  def geometry(self, dc_op: DcOp):
    pass
  
  def gm_id_vs_jd(self, dc_op: DcOp):
    pass
  
  def ft_vs_jd(self, dc_op: DcOp):
    pass
  
  def ft_vs_gm_id(self, dc_op: DcOp):
    pass
  
  def av_vs_gm_id(self, dc_op: DcOp):
    pass
  
  def fom_bw_vs_jd(self, dc_op: DcOp):
    pass
  
  def fom_nfbw_vs_jd(self, dc_op: DcOp):
    pass
  
  def fo(cl: float, dc_op: DcOp):
    pass
  
  def cgg_vs_vgs(self, dc_op: DcOp):
    pass
  
  def c_vs_gm_id(self, dc_op: DcOp):
    pass
  
  def c_vs_jd(self, dc_op: DcOp):
    pass
  
  def ron_vs_jd(self, dc_op: DcOp):
    pass
  
    

if __name__ == "__main__":
  device = Device(
    "/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv", 
    "/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_bsim4_params.csv", 
    "nch"
  )
  print(device.lut.head())
  print(device.bsim4params)
  