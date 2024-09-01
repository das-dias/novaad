# plotting and user control
from dash import Dash, html, dcc, callback, Output, Input
from pandas import DataFrame
import plotly.express as px

from .core import Device, SizingSpecification, DcOp, ElectricModel, Sizing

class GuiApp:
  def __init__(self):
    raise NotImplementedError("This is a placeholder for the GUI app")
  
  def run(self):
    raise NotImplementedError("This is a placeholder for the GUI app")