import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ρ = 1000 # kg/m3
g = 9.81 # m/s2
sol_datafile = './field_measurements/DF27_period1_four_days.csv'


data = pd.read_csv(sol_datafile, parse_dates=['Unnamed: 0'])
ds_sapflow = data['Total Flow(cm3/h)'].values.copy()*1000
ds_Ψx = (data['Water Potential (MPa)'].values*1e6)/(ρ*g) #mm

