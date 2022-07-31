# IPython log file

# Import data processing packages
import pandas as pd
import numpy as np

#Time Series and Metrics Packages
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error
from prophet.diagnostics import performance_metrics, cross_validation

#Time Series Modeling Packages
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from statsmodels.tsa.statespace import sarimax 
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from darts.models.forecasting import tbats
from darts.models.forecasting.gradient_boosted_model import LightGBMModel
from darts.models.forecasting.random_forest import RandomForest
from darts import TimeSeries
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from prophet import Prophet
import lightgbm as lgb

#Visualization 
import matplotlib. pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", 'No frequency') 
accomodations = pd.read_csv('../NZ Data/Accomodation Data Cleaned.csv', parse_dates = True, \
                       index_col = 0)
accomodations.head()
#TotalWellington Prophet
y = accomodations['TotalWellingtonOccupancy']

res = seasonal_decompose(y, model='mul')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')
# Preparing our data for prophet
y_prophet = accomodations[['TotalWellingtonOccupancy']]
y_prophet = y_prophet.reset_index()
y_prophet.rename(columns = {'Date':'ds', 'TotalWellingtonOccupancy':'y'}, inplace = True)
y_prophet.head()
get_ipython().run_cell_magic('capture', '--no-stderr', "#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr --no-stdout', "#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr --no-stdout --no-display', "#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr,--no-stdout --no-display', "#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr, --no-stdout --no-display', "#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr --no-stdout --no-display', "#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr --no-stdout --no-display output', "#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "\n#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--std-out', "\n#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--stdout', "\n#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stdout', "\n#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-display', "\n#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-display', "\n#Initial model we will tweak this model a bit\nm = Prophet();\n\nm.fit(y_prophet);\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS');\n")
get_ipython().run_cell_magic('capture', '', "\n#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr', "\n#Initial model we will tweak this model a bit\nm = Prophet()\n\nm.fit(y_prophet)\n    \ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr', "\n#Initial model we will tweak this model a bit\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr', "#Initial model we will tweak this model a bit\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr --no-stdout --no-display', "#Initial model we will tweak this model a bit\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr --no-stdout --no-display --no-output', "#Initial model we will tweak this model a bit\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr --no-stdout --no-display --output', "#Initial model we will tweak this model a bit\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '--no-stderr --no-stdout --no-display output', "#Initial model we will tweak this model a bit\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
from __future__ import print_function
import sys
get_ipython().run_cell_magic('capture', '--no-stderr --no-stdout --no-display output', "#Initial model we will tweak this model a bit\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\nm = Prophet()\n%logstop m.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\nm = Prophet()\n%logstop[]\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\nm = Prophet()\n%logstop\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\nm = Prophet()\n%logstop[]\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\nm = Prophet()\n%logstop\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\nm = Prophet()\n%logstop[]\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
y_pred.tail()
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\n%logstop[]\nm = Prophet()\n\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
from __future__ import print_function
import sys
get_ipython().run_line_magic('logstop[]', '')
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\n\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', 'output', "#Initial model we will tweak this model a bit\n%logstop[]\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
#Initial model we will tweak this model a bit
get_ipython().run_line_magic('logstop[]', '')
m = Prophet()
m.fit(y_prophet)
y_pred = m.make_future_dataframe(periods = 24, freq = 'MS')
get_ipython().run_cell_magic('capture', '', "#Initial model we will tweak this model a bit\n%logstop[]\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '', "#Initial model we will tweak this model a bit\n%logstop[] #to stop error messages\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '', "#Initial model we will tweak this model a bit\n%logstop #to stop error messages\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '', "#Initial model we will tweak this model a bit\n%logstop- #to stop error messages, mostly unnecessary INFO\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
get_ipython().run_cell_magic('capture', '', "#Initial model we will tweak this model a bit\n%logstop; #to stop error messages, mostly unnecessary INFO\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
#Have to place this here to avoid further extraneous messages from prophet
#Get very long during certain processes
#Cannot avoid the first one (due to cmdstanpy)
get_ipython().run_line_magic('logstar-', '')
import logging
logger = logging.getLogger('cmdstanpy')
logger.setLevel(logging.ERROR)
logger2 = logging.getLogger('prophet')
logger2.setLevel(logging.ERROR)
#Have to place this here to avoid further extraneous messages from prophet
#Get very long during certain processes
#Cannot avoid the first one (due to cmdstanpy)
get_ipython().run_line_magic('logstart-', '')
import logging
logger = logging.getLogger('cmdstanpy')
logger.setLevel(logging.ERROR)
logger2 = logging.getLogger('prophet')
logger2.setLevel(logging.ERROR)
#Have to place this here to avoid further extraneous messages from prophet
#Get very long during certain processes
#Cannot avoid the first one (due to cmdstanpy)
get_ipython().run_line_magic('logstart', '')
import logging
logger = logging.getLogger('cmdstanpy')
logger.setLevel(logging.ERROR)
logger2 = logging.getLogger('prophet')
logger2.setLevel(logging.ERROR)
#Have to place this here to avoid further extraneous messages from prophet
#Get very long during certain processes
#Cannot avoid the first one (due to cmdstanpy)

import logging
logger = logging.getLogger('cmdstanpy')
logger.setLevel(logging.ERROR)
logger2 = logging.getLogger('prophet')
logger2.setLevel(logging.ERROR)
y_fore = m.predict(y_pred)
y_fore.tail()
#looks okay
fig1 = m.plot(y_fore)
#First model plot, maybe we can improve on this
plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(y_fore['ds'], y_fore['yhat'], alpha=0.7, label='Predicted')

plt.legend()

plt.show()
#Some cross validation to check our metrics
#Manually create cutoffs as cross_validation does not do monthly
cutoffs = pd.date_range(start='2014-01-01', end = '2017-09-01', freq='2MS')
y_cv = cross_validation(m, horizon = '730 days', cutoffs = cutoffs)
y_metrics = performance_metrics(y_cv)
y_metrics
get_ipython().run_cell_magic('capture', '', "#Initial model we will tweak this model a bit\n; #to stop error messages, mostly unnecessary INFO\nm = Prophet()\nm.fit(y_prophet)\ny_pred = m.make_future_dataframe(periods = 24, freq = 'MS')\n")
#Initial model we will tweak this model a bit
("#to stop error messages, mostly unnecessary INFO")
m = Prophet()
m.fit(y_prophet)
y_pred = m.make_future_dataframe(periods = 24, freq = 'MS')
