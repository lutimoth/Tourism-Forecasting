#!/usr/bin/env python
# coding: utf-8

# # Modeling with Pycaret
# 
# To begin my exploration into modeling we will use PyCaret to create some rough forecasting models to understand how the data may be forecast. AFter this exploration if the results are promising, then we will move on to pick specific models to use in order to create a more tailored and specific model targeting our forecasting needs.

# In[1]:


import pandas as pd
import prophet
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


# As always, load the data

arrivals = pd.read_csv('../NZ Data/Arrivals Data Cleaned.csv', skipinitialspace = True, parse_dates = True,                        index_col = 0)
accomodations = pd.read_csv('../NZ Data/Accomodation Data Cleaned.csv', skipinitialspace = True, parse_dates = True,                        index_col = 0)


# In[3]:


arrivals.columns = arrivals.columns.str.replace(r' $','', regex = True)


# In[4]:


arrivals.head()


# In[5]:


arrivals.isna().sum()


# In[6]:


arrivals.dtypes


# In[7]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[8]:


y = arrivals[['BusinessAuckland']]


# In[9]:


res = seasonal_decompose(y, model='additive')

# Plot the original data, the trend, the seasonality, and the residuals 
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(10,12), sharex=True);

res.observed.plot(ax=ax1, title='Observed')
res.seasonal.plot(ax=ax2, title='Seasonality')
res.trend.plot(ax=ax3, title='Trend')
res.resid.plot(ax=ax4, title='Residual IDs')


# In[10]:


import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 


# In[11]:


from pycaret.time_series import TSForecastingExperiment
from pycaret.time_series import pull


# In[12]:


exp = TSForecastingExperiment()
exp.setup(arrivals['BusinessAuckland'], fh=24, seasonal_period = 12)


# In[13]:


exp.check_stats()


# In[14]:


models = exp.models()
models = models.reset_index()
models


# In[15]:


model_list = list(models['ID'])
model_list


# In[16]:


# This is a sample to demonstrate what the modeling does
# The compare models will run through multiple models and test on them all
exp.compare_models(include = model_list)


# In[17]:


from tqdm import tqdm


# In[18]:


all_results = []
final_model = {}


# In[19]:


# We will run through all of these. For each, the outputs are similar to above. 
# We will set verbose = False to minimize the amount of clutter in our notebook
for col in tqdm(arrivals.columns):
    
    # initialize setup
    exp=TSForecastingExperiment()
    exp.setup(data = arrivals[col], fh=24, seasonal_period = 12, verbose = False)
    
    # compare all models
    best_model = exp.compare_models(include = model_list, verbose = False)
    
    # capture the compare result grid and store best model in list
    p = exp.pull().iloc[0:1]
    p['time_series'] = str(col)
    all_results.append(p)
    
    # finalize model i.e. fit on entire data including test set
    #f = finalize_model(best_model)
    
    # attach final model to a dictionary
    #final_model[col] = f
    
    # save transformation pipeline and model as pickle file 
    #save_model(f, model_name='trained_models/' + str(i), verbose=False)
    
# concat the results and display the end result
arrivals_results = pd.concat(all_results,axis=0)
arrivals_results


# In[20]:


arrivals_results.to_csv('../NZ Data/arrivals_model_results.csv')


# In[21]:


# Now we repeat this process for our accomodations dataset
accomodations.head()


# In[22]:


# To cut down on runtime we are going to only focus on the subset of data that we are interested in
# We want to match up with the data we are looking at in the Arrivals Dataset which will focus on the following regions:
# Auckland, Canterbury (Christchurch), Wellington, Queenstown, Total
# We are interested in TOTAL OCCUPANCY for ALL regions
total_accomodation = ['TotalAucklandOccupancy', 'TotalCanterburyOccupancy', 'TotalWellingtonOccupancy',                 'TotalQueenstownOccupancy', 'TotalTotal New ZealandOccupancy']
total_accomodation = list(total_accomodation)

# We are interested in MOTEL OCCUPANCY for ALL regions
motel_accomodation = ['MotelsAucklandOccupancy', 'MotelsCanterburyOccupancy', 'MotelsWellingtonOccupancy',                 'MotelsQueenstownOccupancy', 'MotelsTotal New ZealandOccupancy']
motel_accomodation = list(motel_accomodation)

# We are intersted in HOTEL OCCUPANCY for the following regions:AUCKLAND, ROTORUA, WELLINGTON, QUEENSTOWN, TOTAL
hotels_wanted = ['HotelsAucklandOccupancy', 'HotelsCanterburyOccupancy', 'HotelsWellingtonOccupancy',                 'HotelsQueenstownOccupancy', 'HotelsTotal New ZealandOccupancy']
hotel_accomodation = accomodations[hotels_wanted].columns
hotel_accomodation


# In[23]:


accom_list = list(total_accomodation + motel_accomodation + hotels_wanted)
accom_list


# In[24]:


accomodations_subset = accomodations[accom_list]
accomodations_subset.head()


# In[25]:


all_results = []
final_model = {}


# In[27]:


# We will go straight into modeling as the outputs are similar for our previous dataset

for col in tqdm(accomodations_subset.columns):
    
    # initialize setup
    exp=TSForecastingExperiment()
    exp.setup(data = accomodations[col], fh=24, seasonal_period = 12, verbose = False)
    
    # compare all models
    best_model = exp.compare_models(include = model_list, verbose = False)
    
    # capture the compare result grid and store best model in list
    p = exp.pull().iloc[0:1]
    p['time_series'] = str(col)
    all_results.append(p)
    
    # finalize model i.e. fit on entire data including test set
    #f = finalize_model(best_model)
    
    # attach final model to a dictionary
    #final_model[col] = f
    
    # save transformation pipeline and model as pickle file 
    #save_model(f, model_name='trained_models/' + str(i), verbose=False)
    
# concat the results and display the end result
accomodations_results = pd.concat(all_results,axis=0)
accomodations_results


# In[28]:


accomodations_results.to_csv('../NZ Data/accomodations_model_results.csv')


# Now that we have an idea of which models are the best options for our time series, we can go back to our other notebook and perform more in-depth modeling, training, and tuning on these specific models.
# 
