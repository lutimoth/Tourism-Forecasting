#!/usr/bin/env python
# coding: utf-8

# # New Zealand Time-series Forecasting Model

# # Pre-processing and Modeling 
# 
# In this notebook we are going to do pre-processing and modeling on our time-series data. We are going to decide what we want to focus on with this modeling. Much of the data will need de-trending and seasonality removed. We will be basing our model selection using the [pycaret modeling](https://github.com/naturesbless/Tourism-Forecasting/blob/main/Notebooks/Modeling%20with%20Pycaret.ipynb) that was described in our previous notebook. Many of these models will perform detrending and/or seasonality removal for us but we will still inspect each model for these things to properly set our parameters. There will be some hyperparameter tuning but many models are self-tuning and thus always attemp to converge on the lowest cost function possible.
# 
# We'll start with the tourism arrivals data and then look at the accomodation data. With oour arrivals data, we will be dealing with the issue of the COVID-19 Pandemic. This will be a test of how forecasting models can handle the pandemic. The accomodation data was collected up to just before the COVID-19 Pandemic and thus has a more traditional modeling method.
# 
# Please refer to the Table of Contents to jump to any of the models of interest.
# 
# <font size = "4"> **[1.0 Modeling Tourism Data](#1.0-Modeling-Tourism-Data)** </font>
# - **[1.1 Modeling Business Time Series](#1.1-Modeling-Business-Time-Series)**
#     - [1.1.1 Modeling Business Visits to Auckland (Exp. Smoothing)](#1.1.1-Modeling-Business-Visits-to-Auckland-(Exp.-Smoothing))
#     - [1.1.2 Modeling Business Visits to Christchurch (SARIMAX)](#1.1.2-Modeling-Business-Visits-to-Christchurch-(SARIMAX))
#     - [1.1.3 Modeling Business Visits to Wellington (Exp. Smoothing)](#1.1.3-Modeling-Business-Visits-to-Wellington-(Exp.-Smoothing))
#     - [1.1.4 Modeling Business Visits to Queenstown (ETS)](#1.1.4-Modeling-Business-Visits-to-Queenstown-(ETS))
#     - [1.1.5 Modeling Business Visits to New Zealand (Exp. Smoothing)](#1.1.5-Modeling-Business-Visits-to-New-Zealand-(Exp.-Smoothing))
# - **[1.2 Modeling Holiday and Vacation Visits Time Series](#1.2-Modeling-Holiday-and-Vacation-Visits)**
#     - [1.2.1 Modeling Holiday and Vacation Visits to Auckland (Exp. Smoothing)](#1.2.1-Modeling-Holiday-and-Vacation-Visits-to-Auckland-(Exp.-Smoothing))
#     - [1.2.2 Modeling Holiday and Vacation Visits to Christchurch (ARIMA)](#1.2.2-Modeling-Holiday-and-Vacation-Visits-to-Christchurch-(ARIMA))
#     - [1.2.3 Modeling Holiday and Vacation Visits to Wellington (ETS)](#1.2.3-Modeling-Holiday-and-Vacation-Visits-to-Wellington-(ETS))
#     - [1.2.4 Modeling Holiday and Vacation Visits to Queenstown (ETS)](#1.2.4-Modeling-Holiday-and-Vacation-Visits-to-Queenstown-(ETS))
#     - [1.2.5 Modeling Holiday and Vacation Visits to New Zealand (BATS)](#1.2.5-Modeling-Holiday-and-Vacation-Visits-to-New-Zealand-(BATS))
# - **[1.3 Modeling All Travel to Ports](#1.3-Modeling-All-Travel-to-Ports)**
#   - [1.3.1 Modeling All Travel to Auckland (ETS)](#1.3.1-Modeling-All-Travel-to-Auckland-(ETS))
#   - [1.3.2 Modeling All Travel to Christchurch (ARIMA)](#1.3.2-Modeling-All-Travel-to-Christchurch-(ARIMA))
#   - [1.3.3 Modeling All Travel to Wellington (ETS)](#1.3.3-Modeling-All-Travel-to-Wellington-(ETS))
#   - [1.3.4 Modeling All Travel to Queenstown (Auto_ARIMA)](#1.3.4-Modeling-All-Travel-to-Queenstown-(Auto_ARIMA))
#   - [1.3.5 Modeling All Travel to New Zealand (Exp. Smoothing)](#1.3.5-Modeling-All-Travel-to-New-Zealand-(Exp.-Smoothing))
#   
# - **[1.4 Attempting Alternate Methods on Select Data](#1.4-Attempting-Alternate-Methods-on-Select-Data)**
#     - [1.4.1 Using BATS on Queestown](#1.4.1-Using-BATS-on-Queestown)
#     - [1.4.2 Adding Exogenous Factors](#1.4.2-Adding-Exogenous-Factors)
# - **[1.5 Thoughts on Arrivals Data](#1.5-Thoughts-on-Arrivals-Data)**
# 
# <font size = "4"> **[2.0 Modeling Accomodation Data](#2.0-Modeling-Accomodation-Data)** </font>
# 
# 
#   - **[2.1 All Accomodation Occupancy Models](#2.1-All-Accomodation-Occupancy-Models)**
#     - [2.1.1 Modeling Total Occupancy in Auckland (Random Forest)](#2.1.1-Modeling-Total-Occupancy-in-Auckland-(Random-Forest))
#     - [2.1.2 Modeling Total Occupancy in Canterbury (ETS)](#2.1.2-Modeling-Total-Occupancy-in-Canterbury-(ETS))
#     - [2.1.3 Modeling Total Occupancy in Wellington (FB Prophet)](#2.1.3-Modeling-Total-Occupancy-in-Wellington-(FB-Prophet))
#     - [2.1.4 Modeling Total Occupancy in Queenstown (AdaBoost)](#2.1.4-Modeling-Total-Occupancy-in-Queenstown-(AdaBoost))
#     - [2.1.5 Modeling Total Occupancy in New Zealand (K-Nearest Neighbors)](#2.1.5-Modeling-Total-Occupancy-in-New-Zealand-(K-Nearest-Neighbors))
#   - **[2.2 Modeling Motel Occupancy](#2.2-Modeling-Motel-Occupancy)**
#     - [2.2.1 Modeling Motel Occupancy in Auckland (ETS)](#2.2.1-Modeling-Motel-Occupancy-in-Auckland-(ETS))
#     - [2.2.2 Modeling Motel Occupancy in Canterbury (AdaBoost)](#2.2.2-Modeling-Motel-Occupancy-in-Canterbury-(AdaBoost))
#     - [2.2.3 Modeling Motel Occupancy in Wellington (Random Forest)](#2.2.3-Modeling-Motel-Occupancy-in-Wellington-(Random-Forest))
#     - [2.2.4 Modeling Motel Occupancy in Queenstown (BATS)](#2.2.4-Modeling-Motel-Occupancy-in-Queenstown-(BATS))
#     - [2.2.5 Modeling Motel Occupancy in New Zealand (LightGBM)](#2.2.5-Modeling-Motel-Occupancy-in-New-Zealand-(LightGBM))
#   - **[2.3 Modeling Hotel Occupancy](#2.3-Modeling-Hotel-Occupancy)**
#     - [2.3.1 Modeling Hotel Occupancy in Auckland (Random Forest)](#2.3.1-Modeling-Hotel-Occupancy-in-Auckland-(Random-Forest))
#     - [2.3.2 Modeling Hotel Occupancy in Canterbury (Exp. Smoothing)](#2.3.2-Modeling-Hotel-Occupancy-in-Canterbury-(Exp.-Smoothing))
#     - [2.3.3 Modeling Hotel Occupancy in Wellington (Random Forest)](#2.3.3-Modeling-Hotel-Occupancy-in-Wellington-(Random-Forest))
#     - [2.3.4 Modeling Hotel Occupancy in Queenstown (FB Prophet)](#2.3.4-Modeling-Hotel-Occupancy-in-Queenstown-(FB-Prophet))
#     - [2.3.5 Modeling Hotel Occupancy in New Zealand (ETS)](#2.3.5-Modeling-Hotel-Occupancy-in-New-Zealand-(ETS))
#   -**[2.4 Thoughts on Accomodations Data](#2.4-Thoughts-on-Accomodations-Data)**
#   
# <font size = "4"> **[3.0 Final Thoughts](#3.0-Final-Thoughts)** </font>

# In[1]:


# Lets import as many packages as we can here at once

# Import data processing packages
import pandas as pd
import numpy as np

# Time Series and Metrics Packages
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error
from prophet.diagnostics import performance_metrics, cross_validation
from darts.metrics import metrics

# Time Series Modeling Packages
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

# Visualization 
import matplotlib. pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", 'No frequency') 


# # 1.0 Modeling Tourism Data
# 
# Not all of the `arrivals` data is complete so we want to make sure we pick datasets that have less issues. This will minimize the amount of issues that we have when modeling and allow us to make decent forecasts. We are going to look at regions that are parallel with our accomodations test. For our region of `Christchurch` the equivalent is `Canterbury` which is the region.
# 
# For our `arrivals` data we are going to look at the following regions as these have the most complete time-series data:
# > - Auckland
# > - Christchurch
# > - Wellington
# > - Queenstown
# > - All New Zealand Ports
#     
# For each of these regions, we are going to look at the following reasons for travel:
# > - Business
# > - Holiday
# > - Visiting Friends
# > - Total Travel
# 

# In[3]:


# Load our Tourism Data
# Remove extra white space at end of column

arrivals = pd.read_csv('../NZ Data/Arrivals Data Cleaned.csv', skipinitialspace = True, parse_dates = True,                        index_col = 0)
arrivals.columns = arrivals.columns.str.replace(r' $','', regex = True)


# In[4]:


# Check to make sure index is datetime
arrivals.index


# In[5]:


arrivals.head()


# In[6]:


# Load our model results from the pycaret notebook to look at which models might work best for us
potential_models = pd.read_csv('../NZ Data/arrivals_model_results.csv',index_col = 0)
potential_models.head()


# We have here the list of all of our models that we tested using the pycaret model. As discussed in our previous notebook, we are not interested in actually modeling out all of these trends due to issues with the data or lack of historic data. We'll start with looking at our Arrivals dataset. Please refer to the pycaret notebook in the GitHub Repository for more information on what the final models that were generated by pycaret were.
# 
# We will be forecasting the following datasets from the Arrivals dataset:
# - TOTAL ALL TRAVEL PURPOSESAuckland
# - TOTAL ALL TRAVEL PURPOSESChristchurch
# - TOTAL ALL TRAVEL PURPOSESWellington
# - TOTAL ALL TRAVEL PURPOSESQueenstown
# - TOTAL ALL TRAVEL PURPOSESTOTAL NEW ZEALAND PORTS
# - BusinessAuckland
# - BusinessChristchurch
# - BusinessWellington
# - BusinessQueenstown
# - BusinessTOTAL NEW ZEALAND PORTS
# - Holiday/VacationAuckland
# - Holiday/VacationChristchurch
# - Holiday/VacationWellington
# - Holiday/VacationQueenstown
# - Holiday/VacationTOTAL NEW ZEALAND PORTS
# 
# This is capture a wide breadth of models without overwhelming ourselves. Thus to make our above model results table more readable, we will subset and look at only the subset we are interested in.

# In[7]:


# Creating list of subset

arrivals_model_list = ['TOTAL ALL TRAVEL PURPOSESAuckland', 'TOTAL ALL TRAVEL PURPOSESChristchurch',                        'TOTAL ALL TRAVEL PURPOSESWellington', 'TOTAL ALL TRAVEL PURPOSESQueenstown',                      'TOTAL ALL TRAVEL PURPOSESTOTAL NEW ZEALAND PORTS',                      'BusinessAuckland','BusinessChristchurch','BusinessWellington', 'BusinessQueenstown',                      'BusinessTOTAL NEW ZEALAND PORTS',                      'Holiday/VacationAuckland','Holiday/VacationChristchurch', 'Holiday/VacationWellington',                      'Holiday/VacationQueenstown','Holiday/VacationTOTAL NEW ZEALAND PORTS']


# In[8]:


# List of the models we are working on

arrivals_model_sub = potential_models[potential_models['time_series'].isin(arrivals_model_list)]
arrivals_model_sub


# ## 1.1 Modeling Business Time Series
# 
# In this section we will model out our `accomodation` model time series for business visits and analyze those results. We will run through a multitude of models and elaborate on each of these as we run through them with a summary at the end.

# ### 1.1.1 Modeling Business Visits to Auckland (Exp. Smoothing)

# In[9]:


# Start with BusinessAuckland, we will use ExponentialSmoothing to model 
# First lets look at the data and understand its seasonality and ACF/PACF

y = arrivals['BusinessAuckland']

res = seasonal_decompose(y, model='mult')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# <br>
# Since we have opted to use exponential smoothing, we are not required to take the first difference <br>
# Exponential smoothing can handle trend data <br>
# The statsmodel implementation of Exponential Smoothing also optimizes its hyperparameters. We mainly need to make sure we input the appropriate trend and seasonal which we can see is "multiplicative" from the data that we can see.
# <br>

# In[10]:


exp = ExponentialSmoothing(y, trend = 'mul', seasonal = 'mul', seasonal_periods = 12, damped_trend = True)


# In[11]:


exp_fit = exp.fit(method = 'bh')
exp_fit.summary()


# In[12]:


plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show()


# In[13]:


y_fore = exp_fit.forecast(36)


# In[14]:


plt.figure(figsize=(12,8))

plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# 
# We can see that the model forecast is somewhat disjointed from the data below. It is predicting a somewhat dampened and stagnant trend. This is due to the COVID-19 pandemic decline. While the data is coming back up, the model still struggles to make appropriate decisions because the primary source of forecasting is from the previous data.

# ### 1.1.2 Modeling Business Visits to Christchurch (SARIMAX)

# In[15]:


# BUSINESS CHRISTCHURCH
# Pycaret helped us decide that a linear model with deseasonilization and detrending would be the best model to utilize
# Due to the linearity of this data, we shall use SARIMAX as it is the combination of Linear Regression and SARIMA 

y = arrivals['BusinessChristchurch']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[16]:


dftest = adfuller(y, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)


# In[17]:


# Lets first look at the ACF and PACF 

fig, ax = plt.subplots(2, figsize=(8,12))

# plot the ACF
plot_acf(y, ax=ax[0], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

# place marker lines on lags 12 and 24 to highlight the seasonality
ax[0].axvline(12, color='red', ls='--', alpha=0.8, lw=0.7, label='lag = 12')
ax[0].axvline(24, color='orange', ls='--', alpha=1, lw=0.8, label='lag = 24')
ax[0].legend()

# plot the PACF
plot_pacf(y, ax=ax[1], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

plt.show()


# In[18]:


#I will use auto_arima in order to determine ARIMA orders
auto_arima(y, m=12).summary()


# In[19]:


model = sarimax.SARIMAX(y, order=(0,1,1), seasonal_order=(1,0,1,12))
model_fit = model.fit()

forecast = model_fit.forecast(24)
pred = model_fit.predict()[12:]

model_fit.summary()


# In[20]:


# Looking at how our predicted models did
plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(pred, alpha=0.7, label='Predicted')
plt.plot(forecast, label='Forecasted Data', alpha=0.5, ls='--')

plt.legend()
plt.show()


# We see here that while the forecasted SARIMAX data did okay at continuin with the data forecast but predicts a downward trend due to the heavy COVID-19 influence. This will be a trend we continue to see for the rest of our data.

# ### 1.1.3 Modeling Business Visits to Wellington (Exp. Smoothing)

# In[21]:


# business wellington
y = arrivals['BusinessWellington']

res = seasonal_decompose(y, model='add')

#'Plot the original data, the trend, the seasonality, and the residuals'
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[22]:


exp = ExponentialSmoothing(y, trend = 'add', seasonal = 'add', seasonal_periods = 12, damped_trend = True)
exp_fit = exp.fit(method = 'bh')

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show();


# In[23]:


exp_fit.summary()


# In[24]:


y_fore = exp_fit.forecast(36)
plt.figure(figsize=(12,8))

plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ### 1.1.4 Modeling Business Visits to Queenstown (ETS)

# In[25]:


# Business Queenstown using ETS
y = arrivals['BusinessQueenstown']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[26]:


ETS = ETSModel(y, error = 'add', trend = 'add', seasonal = 'add', damped_trend = True, seasonal_periods = 12)
ETS_fit = ETS.fit()

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show()


# In[27]:


ETS_fit.summary()


# In[28]:


y_fore = ETS_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# We see that ETS is able to make forecast that are somewhat accurate and follow the return of the upward trend. While the forecasts based on initial data had negative predicitions, our future predictions are able to adapt to the growing trend and not predict negative values.

# ### 1.1.5 Modeling Business Visits to New Zealand (Exp. Smoothing)

# In[29]:


#All BUsiness NZ
y = arrivals['BusinessTOTAL NEW ZEALAND PORTS']

res = seasonal_decompose(y, model='mul')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[30]:


exp = ExponentialSmoothing(y, trend = 'mul', seasonal = 'mul', seasonal_periods = 12, damped_trend = True)
exp_fit = exp.fit(method = 'ls')

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show()


# In[31]:


exp_fit.summary()


# In[32]:


y_fore = exp_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ## 1.2 Modeling Holiday and Vacation Visits
# 
# This will be modeling holiday and vacation visitors to various New Zealand ports.

# ### 1.2.1 Modeling Holiday and Vacation Visits to Auckland (Exp. Smoothing)

# In[33]:


#Holiday/VacationAuckland Exp Smoothing
y = arrivals['Holiday/VacationAuckland']

res = seasonal_decompose(y, model='mul')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[34]:


exp = ExponentialSmoothing(y, trend = 'mul', seasonal = 'mul', damped_trend = True)
exp_fit = exp.fit(method = 'ls')

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show()


# In[35]:


exp_fit.summary()


# In[36]:


y_fore = exp_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ### 1.2.2 Modeling Holiday and Vacation Visits to Christchurch (ARIMA)

# In[37]:


#Holiday/VacationChristchurch ARIMA
y = arrivals['Holiday/VacationChristchurch']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[38]:


fig, ax = plt.subplots(2, figsize=(8,12))

# plot the ACF
plot_acf(y, ax=ax[0], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

# place marker lines on lags 12 and 24 to highlight the seasonality
ax[0].axvline(12, color='red', ls='--', alpha=0.8, lw=0.7, label='lag = 12')
ax[0].axvline(24, color='orange', ls='--', alpha=1, lw=0.8, label='lag = 24')
ax[0].legend()

# plot the PACF
plot_pacf(y, ax=ax[1], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

plt.show()


# In[39]:


dftest = adfuller(y, autolag = 'AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ", dftest[1])
print("3. Num Of Lags : ", dftest[2])
print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
print("5. Critical Values :")
for key, val in dftest[4].items():
    print("\t",key, ": ", val)


# In[40]:


#I will use auto_arima in order to determine ARIMA orders
auto_arima(y, m=12).summary()


# In[41]:


model = sarimax.SARIMAX(y, order=(0,1,0), seasonal_order=(2,0,1,12))
model_fit = model.fit()

forecast = model_fit.forecast(24)
pred = model_fit.predict()[12:]

model_fit.summary()


# In[42]:


plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(pred, alpha=0.7, label='Predicted')
plt.plot(forecast, label='Forecasted Data', alpha=0.5, ls='--')

plt.legend()
plt.show()
# plt.plot(forecast)


# ### 1.2.3 Modeling Holiday and Vacation Visits to Wellington (ETS)

# In[43]:


#Holiday/VacationWellington ETS
y = arrivals['Holiday/VacationWellington']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[44]:


ETS = ETSModel(y, error = 'add', trend = 'add', seasonal = 'add', damped_trend = True, seasonal_periods = 12)
ETS_fit = ETS.fit()

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show()


# In[45]:


ETS_fit.summary()


# In[46]:


y_fore = ETS_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ### 1.2.4 Modeling Holiday and Vacation Visits to Queenstown (ETS)

# In[47]:


#Holiday/VacationQueenstown ETS
y = arrivals['Holiday/VacationQueenstown']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[48]:


ETS = ETSModel(y, error = 'add', trend = 'add', seasonal = 'add', damped_trend = True, seasonal_periods = 12)
ETS_fit = ETS.fit()

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show()


# In[49]:


ETS_fit.summary()


# In[50]:


y_fore = ETS_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ### 1.2.5 Modeling Holiday and Vacation Visits to New Zealand (BATS)

# In[51]:


#Holiday/VacationTOTAL NEW ZEALAND PORT
y = arrivals[['Holiday/VacationTOTAL NEW ZEALAND PORTS']]

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[52]:


#Create TimeSeries datatype
ts = TimeSeries.from_series(y)

#Create the BATS model and fit
model = tbats.BATS()
model_fit = model.fit(ts)

#Predict with the model
y_pred = model_fit.predict(36)

#Convert back to dataframe so we can use it
y_pred_df = y_pred.pd_dataframe()
y_pred_df.head()


# In[53]:


#Plot onto the time series
plt.figure(figsize=(15,8))

plt.plot(y, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_pred_df)


# We can see that the BATS model has the most gentle curves of the data and actually seems to be following a trend that includes growth. While it is predicting 0s, we see no negative values and it predicted quite well to the data. BATS seems to have handled the COVID-19 data quite well and is attempting to forecast trneds similar to old cyclical historic trends. 

# ## 1.3 Modeling All Travel to Ports
# 
# This will be modeling all visits to all ports.

# ### 1.3.1 Modeling All Travel to Auckland (ETS)

# In[54]:


#TOTAL ALL TRAVEL PURPOSESAuckland ETS
y = arrivals['Holiday/VacationQueenstown']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[55]:


ETS = ETSModel(y, error = 'add', trend = 'add', seasonal = 'add', damped_trend = True, seasonal_periods = 12)
ETS_fit = ETS.fit()

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show()


# In[56]:


ETS_fit.summary()


# In[57]:


y_fore = ETS_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ### 1.3.2 Modeling All Travel to Christchurch (ARIMA)

# In[58]:


#TOTAL ALL TRAVEL PURPOSESChristchurch arima
y = arrivals['TOTAL ALL TRAVEL PURPOSESChristchurch']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[59]:


fig, ax = plt.subplots(2, figsize=(8,12))

# plot the ACF
plot_acf(y, ax=ax[0], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

# place marker lines on lags 12 and 24 to highlight the seasonality
ax[0].axvline(12, color='red', ls='--', alpha=0.8, lw=0.7, label='lag = 12')
ax[0].axvline(24, color='orange', ls='--', alpha=1, lw=0.8, label='lag = 24')
ax[0].legend()

# plot the PACF
plot_pacf(y, ax=ax[1], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

plt.show()


# In[60]:


#I will use auto_arima in order to determine ARIMA orders
auto_arima(y, m=12).summary()


# In[61]:


model = sarimax.SARIMAX(y, order=(0,1,2), seasonal_order=(2,0,1,12))
model_fit = model.fit()

forecast = model_fit.forecast(24)
pred = model_fit.predict()[12:]

model_fit.summary()


# In[62]:


plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(pred, alpha=0.7, label='Predicted')
plt.plot(forecast, label='Forecasted Data', alpha=0.5, ls='--')

plt.legend()
plt.show()
# plt.plot(forecast)


# ### 1.3.3 Modeling All Travel to Wellington (ETS)

# In[63]:


#TOTAL ALL TRAVEL PURPOSESWellington ETS
y = arrivals['TOTAL ALL TRAVEL PURPOSESWellington']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[64]:


ETS = ETSModel(y, error = 'add', trend = 'add', seasonal = 'add', damped_trend = True, seasonal_periods = 12)
ETS_fit = ETS.fit()

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()

plt.show()


# In[65]:


ETS_fit.summary()


# In[66]:


y_fore = ETS_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ### 1.3.4 Modeling All Travel to Queenstown (Auto_ARIMA)

# In[67]:


#TOTAL ALL TRAVEL PURPOSESQueenstown auto_arima
y = arrivals['TOTAL ALL TRAVEL PURPOSESQueenstown']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[68]:


fig, ax = plt.subplots(2, figsize=(8,12))

# plot the ACF
plot_acf(y, ax=ax[0], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

# place marker lines on lags 12 and 24 to highlight the seasonality
ax[0].axvline(12, color='red', ls='--', alpha=0.8, lw=0.7, label='lag = 12')
ax[0].axvline(24, color='orange', ls='--', alpha=1, lw=0.8, label='lag = 24')
ax[0].legend()

# plot the PACF
plot_pacf(y, ax=ax[1], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

plt.show()


# In[69]:


#I will use auto_arima in order to determine ARIMA orders
auto_arima(y, m=12, method = 'cg', information_criterion = 'bic', error_action = 'ignore').summary()


# In[70]:


model = sarimax.SARIMAX(y, order=(2,1,0), seasonal_order=(2,0,1,12))
model_fit = model.fit()

forecast = model_fit.forecast(36)
pred = model_fit.predict()[12:]

model_fit.summary()


# In[71]:


plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(pred, alpha=0.7, label='Predicted')
plt.plot(forecast, label='Forecasted Data', alpha=0.5, ls='--')

plt.legend()
plt.show()
# plt.plot(forecast)


# ### 1.3.5 Modeling All Travel to New Zealand (Exp. Smoothing)

# In[72]:


#TOTAL ALL TRAVEL PURPOSESTOTAL NEW ZEALAND PORTS
y = arrivals['TOTAL ALL TRAVEL PURPOSESTOTAL NEW ZEALAND PORTS']

res = seasonal_decompose(y, model='mul')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[73]:


exp = ExponentialSmoothing(y, trend = 'mul', seasonal = 'mul', damped_trend = True)
exp_fit = exp.fit(method = 'ls')

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show()


# In[74]:


exp_fit.summary()


# In[75]:


y_fore = exp_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ## 1.4 Attempting Alternate Methods on Select Data
# 
# Due to the poor modeling of a few models, I am going to try a few things. One will be attempting BATs on one of the methods and the other will be attempting a SARIMAX method and use exogenous factor such as GDP. This will be something that may be a proof of concept for helping us deal with the COVID-19 pandemic data.

# ### 1.4.1 Using BATS on Queestown
# 
# Due to the nature of BATS being able to adapt to zero by using Box-Cox transformations and understanding trend changes, we are able to better predict when we have 0 values without having negative trends. The model learns that 0 is the lowest potential value without continuing the trend into negativity. Compared to our traditional forecasting, the extreme negative slope caused by the COVID-19 Pandemic does not have as heavy of an impact on the later data.

# In[76]:


# This model really struggled so we will be attempting a BATS model on it! 

#TOTAL ALL TRAVEL PURPOSESQueenstown auto_arima
y = arrivals['TOTAL ALL TRAVEL PURPOSESQueenstown']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[77]:


#Create TimeSeries datatype
ts = TimeSeries.from_series(y)

#Create the BATS model and fit
model = tbats.BATS()
model_fit = model.fit(ts)

#Predict with the model
y_pred = model_fit.predict(36)

#Convert back to dataframe so we can use it
y_pred_df = y_pred.pd_dataframe()
y_pred_df.head()


# In[78]:


#Plot onto the time series
plt.figure(figsize=(15,8))

plt.plot(y, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_pred_df)


# The BATS model seems to be handling datasets with 0 data a lot better than some of our other models. It at least is able to avoid projecting negative values. We see that BATS has similar forecasting to our historical data and perhaps could be a decent predictor for our COVID data until we get more data as we return to normal.

# ### 1.4.2 Adding Exogenous Factors
# 
# We can also look at using other features to model our data. By looking at Exogenous factors such as GDP which are non-zero, we may be able to apply these trends to our data and see how that can impact our model and hopefully prevent it from having consistently negative predictions. Exogenous factors can help us provide information to the model by using external trends that impact our data. In this instance, we will be using GDP.
# 
# GDP may drop but will never go to 0 and thus should help us deal with the large swath of 0 values in our data. Hopefully, it can build on the existing SARIMAX model and help us create a slightly more consistent and better model.

# In[79]:


#Lets add in our GDP data

gdp = pd.read_csv('../NZ Data/NZ_GDP_QTRLY1960-2021.csv', parse_dates = True)
gdp.head()


# In[80]:


# The data that we have is quarterly. So lets first make the data properly date-time and then upsample to monthly
gdp = gdp.rename(columns = {'Quarter':'Date'})
gdp['Date'] = pd.to_datetime(gdp['Date'])


# In[81]:


gdp.index = gdp['Date']
gdp.index


# In[82]:


gdp.drop('Date', axis = 1, inplace = True)
gdp.head()


# In[83]:


#Upsample to monthly
gdp = gdp.resample('MS').ffill()
gdp.dropna(inplace = True)
gdp.index


# In[84]:


#Lets look at what we're dealing with

gdp.plot()


# In[85]:


#TOTAL ALL TRAVEL PURPOSESQueenstown auto_arima
#We wil fit this SARIMAX model we used earlier but now add Exogenous features
y = arrivals[['TOTAL ALL TRAVEL PURPOSESQueenstown']]
y


# In[86]:


start_date = pd.to_datetime('1987-03-01')
end_date = pd.to_datetime('2022-02-01')
y = y.query('index > @start_date and index < @end_date')
y


# In[87]:


#I will use auto_arima in order to determine ARIMA orders
auto_arima(y, X = gdp, m=12, method = 'cg', information_criterion = 'bic', error_action = 'ignore').summary()


# In[88]:


model = sarimax.SARIMAX(y, exog = gdp, order=(3,0,0), seasonal_order=(1,0,0,12))
model_fit = model.fit()

pred = model_fit.predict(exog = gdp)[24:]
forecast = model_fit.forecast(36, exog = gdp.iloc[:36])

model_fit.summary()


# In[89]:


plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(pred, alpha=0.7, label='Predicted')
plt.plot(forecast, label='Forecasted Data', alpha=0.5, ls='--')

plt.legend()
plt.show()
# plt.plot(forecast)


# We can see that while the exogenous features might have helped with the post-forecast trend, it continues to be negative due to the huge dip in GDP around the same time as our COVID data because SARIMAX is not a constrained model. We do see that the trend returns upward with increasing GDP and perhaps is a sign of some improvement.

# ## 1.5 Thoughts on Arrivals Data
# 
# Overall, our models did alright when it came to existing data. It fitted well without overfitting. However, once the pandemic hit there was a struggle to forecast out into the future due to the extremes causes by the pandemic dataset. One of the main issues we saw was the penchant for predicting negative values. This is due to the fact that many time series data is not constrained and therefore will follow the trends that it sees occuring without considering that there must be some minimum. 
# 
# Alternatively, some models handled that quite well. ETS and BATS seemed to best handle the extremes of the COVID data. Focusing on these models may have been best for the time series analysis and I think that it comes down to the way they are implemented. While these models focus on how the data trends and is seasonal, ETS smooths the data set and softens the blow of such extremes as well as not being too concerned with stationarity. BATS allows us to test numerous transformations such as Box-Cox, Arima-error, and other parameters. The implementation of BATS in DARTS is self-tuning. By its nature BATS is tuned into seasonal trends and also applies some exponential smoothing. We are able to thus use these models to be a little constrained and make sure we are staying within the historical data.
# 
# While I tried some initial exploration using GDP data, I think there can be further explorationg using more exogenous variables to create a better prediction. The GDP data helped in the sense of emphasizing that there is an upward trend but failed to adapt for the negative values. The GDP value actually exacerbated the start of the negative values as itself also dropped. While the prediction forecast is overall better if we perhaps add the absolute difference of the max vaulue from 0 and forecast it upwards then it could demonstrate an acceptable forecast but as it stands the Exog modelin is not enough.

# # 2.0 Modeling Accomodation Data

# We will now be modeling our accomodations data. As noted in our data wrangling notebook, we will be focusing on the occupancy of these accomodations. While we have both the capacity and occupancy data available, understanding occupancy trends is more important as it notes how many visitors are ACTUALLY staying at these hotels. Unlike the arrivals dataset, however, this data was not collected up through the COVID-19 Pandemic yet and currently is only available up to the third quarter of 2019. Thus, we will be modeling this data as if we were forecasting into 2020-2021 and hopefully when that data becomes available we can compare it to our models.
# 
# For our `accomodations` data we are going to look at the following regions as these have the most complete time-series data:
# > - Auckland
# > - Canterbury
# > - Wellington
# > - Queenstown
# > - Total New Zealand
#     
# For each of these regions, we are going to look at the following accomodations:
# > - Total Occupancy
# > - Motels
# > - Hotels

# In[90]:


accomodations = pd.read_csv('../NZ Data/Accomodation Data Cleaned.csv', parse_dates = True,                        index_col = 0)
accomodations.head()


# In[91]:


accomodations_model_list = pd.read_csv('../NZ Data/accomodations_model_results.csv')
accomodations_model_list


# ## 2.1 All Accomodation Occupancy Models
# 
# This is modeling occupancy for all of our accomodations.

# ### 2.1.1 Modeling Total Occupancy in Auckland (Random Forest)

# In[92]:


#TotalAucklandOccupancy random forest w/ cond deseasonalize and detrend
y = accomodations[['TotalAucklandOccupancy']]

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[93]:


# Code from machinelearningmastery, Jason Brownlee at 
# https://machinelearningmastery.com/random-forest-for-time-series-forecasting/
# transform a time series dataset into a supervised learning dataset

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]
 
# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = random_forest_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		#print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions


#Our function for multistep forecast
def multistep_forecast(steps, y_list):
    y_pred = y_list.copy()
    
    for s in range(steps):
        train = series_to_supervised(y_pred, n_in=steps)
            # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
            # fit model
        model = RandomForestRegressor(n_estimators=1000)
        model.fit(trainX, trainy)
        row = y_pred[-steps:]
        yhat = model.predict(np.asarray([row]))
        y_pred.append(yhat[0])
    return y_pred


# In[94]:


# Convert data to a list
y_list = list(y['TotalAucklandOccupancy'])
len(y_list)


# In[95]:


# transform the time series data into supervised learning
data = series_to_supervised(y_list, n_in=50)
# evaluate
mae, y, yhat = walk_forward_validation(data, n_test=50)
#print('MAE: %.3f' % mae)
# plot expected vs predicted
plt.plot(y, label='Expected')
plt.plot(yhat, label='Predicted')
plt.legend()
plt.show()


# In[96]:


# transform the time series data into supervised learning
n = 50
y_pred = multistep_forecast(n, y_list)


# In[97]:


len(y_pred)


# In[98]:


pred_index = range(len(y_pred) - n, len(y_pred))


# In[99]:


plt.figure(figsize = (12,10))
plt.plot(y_list, label='Expected')
(line1,) = plt.plot(pred_index, y_pred[-n:], label='Predicted')
plt.legend()
plt.show()


# In[100]:


#Use DARTS as our comparison

from darts.models.forecasting.random_forest import RandomForest


# In[101]:


y = accomodations[['TotalAucklandOccupancy']]
rf = RandomForest(lags = 24)
#Convert to timeseries
y_ts = TimeSeries.from_dataframe(y)
rf.fit(y_ts)

y_pred = rf.predict(24)
y_pred_df = TimeSeries.pd_dataframe(y_pred)


# In[102]:


plt.figure(figsize=(15,8))

plt.plot(y, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_pred_df)


# In[103]:


y = accomodations['TotalAucklandOccupancy']
fig, ax = plt.subplots(2, figsize=(8,12))

# plot the ACF
plot_acf(y, ax=ax[0], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

# place marker lines on lags 12 and 24 to highlight the seasonality
ax[0].axvline(12, color='red', ls='--', alpha=0.8, lw=0.7, label='lag = 12')
ax[0].axvline(24, color='orange', ls='--', alpha=1, lw=0.8, label='lag = 24')
ax[0].legend()

# plot the PACF
plot_pacf(y, ax=ax[1], vlines_kwargs={'ls':'--', 'linewidth': 0.7}, lags=26)

plt.show()


# In[104]:


auto_arima(y, m=12, method = 'cg', information_criterion = 'bic', error_action = 'ignore').summary()


# In[105]:


model = sarimax.SARIMAX(y, order=(1,1,2), seasonal_order=(1,0,0,12))
model_fit = model.fit()

forecast = model_fit.forecast(24)
pred = model_fit.predict()[12:]


# In[106]:


plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(pred, alpha=0.7, label='Predicted')
plt.plot(forecast, label='Forecasted Data', alpha=0.5, ls='--')

plt.legend()
plt.show()
# plt.plot(forecast)


# In[107]:


y = accomodations['TotalAucklandOccupancy']
y = y.astype('float64')

ETS = ETSModel(y, error = 'add', trend = 'add', seasonal = 'add', seasonal_periods = 12)
ETS_fit = ETS.fit()

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()

plt.show()


# In[108]:


ETS_fit.summary()


# In[109]:


y_fore = ETS_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# For our first model, I wanted to test a variety of models to each other. Pycaret recommended using a Random Forest model. I found a model that was written out by Jason Brownlee with some adapatations for our multi-step forecasting. I then compared this to the prepackaged DARTS model and ETS. Of these, the Random Forest model that was self-written seemed to perform quite well so I decided to keep using it for our future forecasts. 

# ### 2.1.2 Modeling Total Occupancy in Canterbury (ETS)

# In[110]:


#TotalCanterbury ETS
y = accomodations['TotalCanterburyOccupancy']
y = y.astype('float64')


res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[111]:


ETS = ETSModel(y, error = 'add', trend = 'add', seasonal = 'add', seasonal_periods = 12)
ETS_fit = ETS.fit()

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
y = y.astype('float64')

plt.legend()

plt.show()


# In[112]:


ETS_fit.summary()


# In[113]:


y_fore = ETS_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ### 2.1.3 Modeling Total Occupancy in Wellington (FB Prophet)

# In[114]:


#TotalWellington Prophet
y = accomodations['TotalWellingtonOccupancy']

res = seasonal_decompose(y, model='mul')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed');
res.seasonal.plot(ax=ax2, title='Seasonality');
res.trend.plot(ax=ax3, title='Trend');
res.resid.plot(ax=ax4, title='Residual IDs');


# In[115]:


# Preparing our data for prophet
y_prophet = accomodations[['TotalWellingtonOccupancy']]
y_prophet = y_prophet.reset_index()
y_prophet.rename(columns = {'Date':'ds', 'TotalWellingtonOccupancy':'y'}, inplace = True)
y_prophet.head()


# In[116]:


#Initial model we will tweak this model a bit
m = Prophet()
m.fit(y_prophet)
y_pred = m.make_future_dataframe(periods = 24, freq = 'MS')


# In[117]:


y_pred.tail()


# In[118]:


#Have to place this here to avoid further extraneous messages from prophet
#Get very long during certain processes
#Cannot avoid the first one (due to cmdstanpy)

import logging
logger = logging.getLogger('cmdstanpy')
logger.setLevel(logging.ERROR)
logger2 = logging.getLogger('prophet')
logger2.setLevel(logging.ERROR)


# In[119]:


y_fore = m.predict(y_pred)
y_fore.tail()


# In[120]:


#looks okay
fig1 = m.plot(y_fore)


# In[121]:


#First model plot, maybe we can improve on this
plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(y_fore['ds'], y_fore['yhat'], alpha=0.7, label='Predicted')

plt.legend()

plt.show()


# In[122]:


#Some cross validation to check our metrics
#Manually create cutoffs as cross_validation does not do monthly
cutoffs = pd.date_range(start='2014-01-01', end = '2017-09-01', freq='2MS')
y_cv = cross_validation(m, horizon = '730 days', cutoffs = cutoffs)
y_metrics = performance_metrics(y_cv)
y_metrics


# In[123]:


#Lets tune a little
m = Prophet(daily_seasonality = False, seasonality_mode = 'multiplicative')
m.add_country_holidays(country_name = 'US')
m.fit(y_prophet)
y_pred = m.make_future_dataframe(periods = 24, freq = 'MS')
y_pred.tail()


# In[124]:


y_fore = m.predict(y_pred)
y_fore.tail()


# In[125]:


#First model plot, maybe we can improve on this
plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(y_fore['ds'], y_fore['yhat'], alpha=0.7, label='Predicted')

plt.legend()

plt.show()


# In[126]:


cutoffs = pd.date_range(start='2014-01-01', end = '2017-09-01', freq='2MS')
y_cv = cross_validation(m, horizon = '730 days', cutoffs = cutoffs)
y_metrics = performance_metrics(y_cv)
y_metrics.head()


# Seems like model got worse, we'll stick to the defaults for forecasting for now!

# ### 2.1.4 Modeling Total Occupancy in Queenstown (AdaBoost)
# 
# Similar to our Random Forest model, I used the foundation of that code to create this AdaBoost multi-step model as it was similar in function.

# In[127]:


#TotalQueenstown AdaBoost

y = accomodations['TotalQueenstownOccupancy']

res = seasonal_decompose(y, model='mul')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[128]:


# fit AdaBoost model and make a one step prediction
def random_ada_forecast(train, testX):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = AdaBoostClassifier(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_ada(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = random_ada_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		#print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

def multistep_ada(steps, y_list):
    y_pred = y_list.copy()
    
    for s in range(steps):
        train = series_to_supervised(y_pred, n_in=steps)
            # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
            # fit model
        model = AdaBoostClassifier(n_estimators=1000)
        model.fit(trainX, trainy)
        row = y_pred[-steps:]
        yhat = model.predict(np.asarray([row]))
        y_pred.append(yhat[0])
    return y_pred


# In[129]:


# Convert data to a list
y_list = list(y)
# transform the time series data into supervised learning
data = series_to_supervised(y_list, n_in=100)
# evaluate
mae, y, yhat = walk_forward_ada(data, n_test=100)
#print('MAE: %.3f' % mae)
# plot expected vs predicted
fig = plt.figure(figsize = (12,10))
plt.plot(y, label='Expected')
plt.plot(yhat, label='Predicted')
plt.legend()
plt.show()


# In[130]:


# set amount to forecast
n = 50
#Forecast forward
y_pred = multistep_ada(n, y_list)
pred_index = range(len(y_pred) - n, len(y_pred))

#plot our forecasting
plt.figure(figsize = (12,10))
plt.plot(y_list, label='Expected')
(line1,) = plt.plot(pred_index, y_pred[-n:], label='Predicted')
plt.legend()
plt.show()


# An interesting issue with the AdaBoost model is it seemed to regress to a mean and flatten out over time. There is one other model that uses this method and something similar happens but later in the forecast.

# ### 2.1.5 Modeling Total Occupancy in New Zealand (K-Nearest Neighbors)
# 
# Using the foundation of the random-forest I am able to create a time forecasting model using KNN and multi-step.

# In[131]:


#TotalTotalnew Zealand KNN 
y = accomodations[['TotalTotal New ZealandOccupancy']]

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[132]:


def random_knn_forecast(train, testX):
	# transform list into array
	train = np.asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = KNeighborsRegressor(n_neighbors = 10)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	return yhat[0]
 
# walk-forward validation for univariate data
def walk_forward_knn(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = random_knn_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		#print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

def multistep_knn(steps, y_list):
    y_pred = y_list.copy()
    
    for s in range(steps):
        train = series_to_supervised(y_pred, n_in=steps)
            # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
            # fit model
        model = KNeighborsRegressor(n_neighbors = 10)
        model.fit(trainX, trainy)
        row = y_pred[-steps:]
        yhat = model.predict(np.asarray([row]))
        y_pred.append(yhat[0])
    return y_pred


# In[133]:


# Convert data to a list
y_list = list(y['TotalTotal New ZealandOccupancy'])
# transform the time series data into supervised learning
data = series_to_supervised(y_list, n_in=100)
# evaluate
mae, y, yhat = walk_forward_knn(data, n_test=100)
#print('MAE: %.3f' % mae)
# plot expected vs predicted
fig = plt.figure(figsize = (12,10))
plt.plot(y, label='Expected')
plt.plot(yhat, label='Predicted')
plt.legend()
plt.show()


# In[134]:


# set amount to forecast
n = 50
#Forecast forward
y_pred = multistep_knn(n, y_list)
pred_index = range(len(y_pred) - n, len(y_pred))

#plot our forecasting
plt.figure(figsize = (12,10))
plt.plot(y_list, label='Expected')
(line1,) = plt.plot(pred_index, y_pred[-n:], label='Predicted')
plt.legend()
plt.show()


# ## 2.2 Modeling Motel Occupancy

# ### 2.2.1 Modeling Motel Occupancy in Auckland (ETS)

# In[135]:


#Motels Auckland ETS
y = accomodations['MotelsAucklandOccupancy']
y = y.astype('float64')

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[136]:


ETS = ETSModel(y, error = 'add', trend = 'add', seasonal = 'add', seasonal_periods = 12)
ETS_fit = ETS.fit()

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()

plt.show()


# In[137]:


ETS_fit.summary()


# In[138]:


y_fore = ETS_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ### 2.2.2 Modeling Motel Occupancy in Canterbury (AdaBoost)

# In[139]:


#Motels Canterbury Adaboost
y = accomodations['MotelsCanterburyOccupancy']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[140]:


# Convert data to a list
y_list = list(y)
# transform the time series data into supervised learning
data = series_to_supervised(y_list, n_in=100)
# evaluate
mae, y, yhat = walk_forward_ada(data, n_test=100)
#print('MAE: %.3f' % mae)
# plot expected vs predicted
plt.figure(figsize = (12,10))
plt.plot(y, label='Expected')
plt.plot(yhat, label='Predicted')
plt.legend()
plt.show()


# In[141]:


# set amount to forecast
n = 50
#Forecast forward
y_pred = multistep_ada(n, y_list)
pred_index = range(len(y_pred) - n, len(y_pred))

#plot our forecasting
plt.figure(figsize = (12,10))
plt.plot(y_list, label='Expected')
(line1,) = plt.plot(pred_index, y_pred[-n:], label='Predicted')
plt.legend()
plt.show()


# ### 2.2.3 Modeling Motel Occupancy in Wellington (Random Forest)

# In[142]:


#Motels Wellington Decision Tree
y = accomodations['MotelsWellingtonOccupancy']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[143]:


# Convert data to a list
y_list = list(y)
# transform the time series data into supervised learning
data = series_to_supervised(y_list, n_in=100)
# evaluate
mae, y, yhat = walk_forward_validation(data, n_test=100)
#print('MAE: %.3f' % mae)
# plot expected vs predicted
plt.plot(y, label='Expected')
plt.plot(yhat, label='Predicted')
plt.legend()
plt.show()


# In[144]:


# choose how many steps forward
n = 50
#run func
y_pred = multistep_forecast(n, y_list)
#find length 
pred_index = range(len(y_pred) - n, len(y_pred))
#plot
plt.figure(figsize = (12,10))
plt.plot(y_list, label='Expected')
(line1,) = plt.plot(pred_index, y_pred[-n:], label='Predicted')
plt.legend()
plt.show()


# ### 2.2.4 Modeling Motel Occupancy in Queenstown (BATS)

# In[145]:


#Motels Queenstown BATS
y = accomodations[['MotelsQueenstownOccupancy']]

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[146]:


#Create TimeSeries datatype
ts = TimeSeries.from_series(y)

#Create the BATS model and fit
model = tbats.BATS()
model_fit = model.fit(ts)

#Predict with the model
y_pred = model_fit.predict(36)

#Convert back to dataframe so we can use it
y_pred_df = y_pred.pd_dataframe()
y_pred_df.head()


# In[147]:


#Plot onto the time series
plt.figure(figsize=(15,8))

plt.plot(y, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_pred_df)


# ### 2.2.5 Modeling Motel Occupancy in New Zealand (LightGBM)

# In[148]:


#Motels Total New Zealand
#lightGBM using DARTS
y = accomodations[['MotelsTotal New ZealandOccupancy']]

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[149]:


from darts.models.forecasting.gradient_boosted_model import LightGBMModel


# In[150]:


lgbm = LightGBMModel(lags = 24)


# In[151]:


#Convert to timeseries
y_ts = TimeSeries.from_dataframe(y)

lgbm.fit(y_ts)


# In[152]:


y_pred = lgbm.predict(24)


# In[153]:


y_pred_df = TimeSeries.pd_dataframe(y_pred)


# In[154]:


plt.figure(figsize=(15,8))

plt.plot(y, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_pred_df)


# ## 2.3 Modeling Hotel Occupancy

# ### 2.3.1 Modeling Hotel Occupancy in Auckland (Random Forest)

# In[155]:


#Hotels Auckland Random Forest
y = accomodations['HotelsAucklandOccupancy']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[156]:


# Convert data to a list
y_list = list(y)
# transform the time series data into supervised learning
data = series_to_supervised(y_list, n_in=100)
# evaluate
mae, y, yhat = walk_forward_validation(data, n_test=100)
#print('MAE: %.3f' % mae)
# plot expected vs predicted
plt.plot(y, label='Expected')
plt.plot(yhat, label='Predicted')
plt.legend()
plt.show()


# In[157]:


# transform the time series data into supervised learning
n = 50
y_pred = multistep_forecast(n, y_list)
pred_index = range(len(y_pred) - n, len(y_pred))
plt.figure(figsize = (12,10))
plt.plot(y_list, label='Expected')
(line1,) = plt.plot(pred_index, y_pred[-n:], label='Predicted')
plt.legend()
plt.show()


# ### 2.3.2 Modeling Hotel Occupancy in Canterbury (Exp. Smoothing)

# In[158]:


#Hotels Canterbury Exp Smooth
y = accomodations['HotelsCanterburyOccupancy']

res = seasonal_decompose(y, model='mult')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[159]:


exp = ExponentialSmoothing(y, trend = 'mul', seasonal = 'mul', damped_trend = True)
exp_fit = exp.fit(method = 'ls')

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()
plt.show()


# In[160]:


exp_fit.summary()


# In[180]:


y_fore = exp_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(exp_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)
plt.axvline(pd.to_datetime('2011-02-22'), color='red', ls='--', alpha=1, lw=2, label='Christchurch Earthquake')
plt.legend(loc='lower right')


# What we can see from the Canterbury hotel data is that here was an extreme event. This event was the Christchurch earthquake on February 22nd, 2011. After an initial off prediction by the model, it adjusted quickly to the huge drop in data. Then after some decreasing, the model eventually adapated to the new trend levels and was slowly able to come back and create the model trends based on historical data. I believe this is a similar effect to what will happen with the COVID-19 pandemic.

# ### 2.3.3 Modeling Hotel Occupancy in Wellington (Random Forest)

# In[162]:


#hotels Wellington Random Forest
y = accomodations['HotelsWellingtonOccupancy']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[163]:


# Convert data to a list
y_list = list(y)
# transform the time series data into supervised learning
data = series_to_supervised(y_list, n_in=100)
# evaluate
mae, y, yhat = walk_forward_validation(data, n_test=100)
#print('MAE: %.3f' % mae)
# plot expected vs predicted
plt.figure(figsize = (12,10))
plt.plot(y, label='Expected')
plt.plot(yhat, label='Predicted')
plt.legend()
plt.show()


# In[164]:


# transform the time series data into supervised learning
n = 50
y_pred = multistep_forecast(n, y_list)
pred_index = range(len(y_pred) - n, len(y_pred))
plt.figure(figsize = (12,10))
plt.plot(y_list, label='Expected')
(line1,) = plt.plot(pred_index, y_pred[-n:], label='Predicted')
plt.legend()
plt.show()


# ### 2.3.4 Modeling Hotel Occupancy in Queenstown (FB Prophet)

# In[165]:


#Hotels Queenstown Prophet
y = accomodations['HotelsQueenstownOccupancy']

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[166]:


# Preparing our data for prophet
y_prophet = accomodations[['HotelsQueenstownOccupancy']]
y_prophet = y_prophet.reset_index()
y_prophet.rename(columns = {'Date':'ds', 'HotelsQueenstownOccupancy':'y'}, inplace = True)
y_prophet.head()


# In[167]:


#Sticking to default model
m = Prophet()
m.fit(y_prophet)
y_pred = m.make_future_dataframe(periods = 24, freq = 'MS')
y_pred.tail()


# In[168]:


y_fore = m.predict(y_pred)
y_fore.tail()


# In[169]:


#First model plot, maybe we can improve on this
plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(y_fore['ds'], y_fore['yhat'], alpha=0.7, label='Predicted')

plt.legend()

plt.show()


# In[170]:


cutoffs = pd.date_range(start='2014-01-01', end = '2017-09-01', freq='2MS')
y_cv = cross_validation(m, horizon = '730 days', cutoffs = cutoffs)
y_metrics = performance_metrics(y_cv)
y_metrics.head()


# ### 2.3.5 Modeling Hotel Occupancy in New Zealand (ETS)

# In[171]:


#Hotels TNZ ETS
y = accomodations['TotalCanterburyOccupancy']
y = y.astype('float')

res = seasonal_decompose(y, model='add')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[172]:


ETS = ETSModel(y, error = 'add', trend = 'add', seasonal = 'mul', seasonal_periods = 12)
ETS_fit = ETS.fit()

plt.figure(figsize=(12,8))

plt.title('Original and Predicted')
plt.plot(y, label='Original')
plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')

plt.legend()

plt.show()


# In[173]:


ETS_fit.summary()


# In[174]:


y_fore = ETS_fit.forecast(36)
plt.figure(figsize=(15,8))

plt.plot(ETS_fit.fittedvalues, alpha=0.7, label='Predicted')
(line1,) = plt.plot(y_fore)


# ## 2.4 Thoughts on Accomodations Data
# 
# Overall, the models performed much better on the accomodations dataset due to the lack of extreme values. I think that some of the models could utilize additional parameter tuning but overall I am happy with the forecasting and the metrics that we see. There is not much here to say except that the AdaBoost and KNN models performed surprisingly well in terms of forecasting future values. However, for each of the Random Forest, AdaBoost, and KNN models, I would have liked to explore the deseasonalization and detrending practices for making them have more stationarity. This way, perhaps the models would have been more accurate after adding them back in. It was a level of complexity I did not get to explore.
# 
# I think the Prophet did quite well although I had to find ways to work around its lack of built-in monthly forecasting and our traditional ARIMA/SARIMAX models performed better than they did on the `accomodation` dataset. Many of the packages are self-tuning or have great optimization to make hyperparameter tuning much easier.
# 
# Additionally, I think the Canterbury data set is a good example of how forecasts will initially lag with extreme situations and then over time adapt. There is less gap in the Canterbury data set as it was a shorter recovery period than the pandemic and also it did not completely reach 0 which is the issue with the pandemic data. I am hopeful that with a short period of time, we can see models performing with some normalcy.

# # 3.0 Final Thoughts
# 
# It was a great exercise working through so many different time series models. I really enjoyed having to think about unique solutions to each of the problems that were presented by each model. It really gave me a great in-depth look at the flexibility at how I can approach time series problems in the future. There are definitely a lot of very powerful packages out there that make it much more accessible to perform time series analysis with less complexity. Understanding the need for deseasonalization and detrending is still important and making sure to do the right kind is key to model success. There is still much to learn.
# 
# For the `arrivals` data, I would encourage the New Zealand goverenment to focus on using BATS or ETS models for the time being to create forecasts. They handle the pandemic data the best and will make somewhat more accurate forecast by nature of minimizing negative values. This is important as New Zealand decides to open its borders. As new information comes in, updating these models with new information will be key.
# 
# As the data becomes more like the `arrivals` data with less extreme occurances, then we can transition to other models such as SARIMAX or even continue using ETS and BATS. The time series models had a much better time predicting with this data. I would like to see the pandemic data for these accomodations and see how the models perform. 
# 
# For future exploration, I think it would be key to utilize exogenous variables to create a more complete picture of these forecasts. Additionally, making sure we explore seasonality and trends in some of the machine learning models will give us more improvement to the models. We can always create better models by exploring more complexity and udnerstanding the data as a whole. Comparing our models to new collected data will give us more of an indea of how confident we can be in these models and whether there needs to be some adjustments. 
