#import numpy as np

#from Visualizing import plotSeries, ACF
#from Models import GridSearch, FitEvaluate, TransferLearning, generalTuning

from Visualizing import *
from Models import *

N = 2000 
T = N / 10
t = np.linspace(0,T,N)
stdv = 0.3
sin_series = np.sin(t) + np.random.normal(0,stdv,N)
cos_series = np.cos(t) + np.random.normal(0,stdv,N)
abs_series = abs(sin_series)
incr_sin = t * sin_series
flunc_sin = (t-100) + (t-100) ** 2 * (np.sin((t-100)) + np.random.normal(0,stdv,N))
sin_trend = 20 * (np.sin( np.pi * (np.sqrt(4 * t + 1) - 1) + 0.5 * np.random.normal(0,stdv,N))) + t 

series_dict = {'t':t,
              'sin(t)':sin_series,
              'cos(t)':cos_series,
              '|sin(t)|':abs_series,
              'Increasing Sine':incr_sin,
              'Fluctuating Sine':flunc_sin,
              'Sine with a Trend':sin_trend}

#plotSeries(series_dict)

#ACF(sin_series,lags=100)

# to perform a grid search over the parameter
#params_grid = {'input_size': [3,30,70],
#               'hidden_units':[100,[100,50],[100,50,50]],
#               'dropout': [True, False],
#               'learning_rate':[4e-5],
#               'n_ahead':[10],
#               'val_split': [0.2],
#               'epochs':[10],
#               'verbose':[False],
#               'plot':[False]}

#model, logs = GridSearch(sin_series,params_grid)

params = {'input_size': 70,
          'hidden_units':[100,50],
          'dropout': False,
          'learning_rate':4e-5,
          'n_ahead':50,
          'val_split': 0.2,
          'epochs':10,
          'verbose': True,
          'plot': True}
model, mse, hist = FitEvaluate(sin_series,params)

#TransferLearning(cos_series[-100:],params,model=model)

#generalTuning(sin_series[-100:],incr_sin[-100:],params)