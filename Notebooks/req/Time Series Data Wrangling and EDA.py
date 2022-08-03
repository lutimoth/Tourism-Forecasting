#!/usr/bin/env python
# coding: utf-8

# # New Zealand Accomodation and Tourism Data

# # Data Wrangling and Exploratory Data Analysis
# 
# In this notebook, we aim to examine time series data from the New Zealand government regarding its tourism and visitors to see what interesting information we can glean from these datasets. Our end goal is to perform forecasting predictions on these datasets as borders open following the COVID-19 pandemic and how the country can prepare itself for an influx of visitors.
# 
# This dataset was downloaded from the New Zealand government's data store website: https://infoshare.stats.govt.nz/
# 
# I was able to query the website for specific datasets of interest. The website contains all sorts of data from population, visa, travel, tourism, and business data. It is a wealth of information for any data scientist to explore. I chose to look at the data specifically referring to the Accomodation Occupancy and Capacity data that was organized by region. I also looked at visitor entries for each New Zealand port and their reason for visiting.

# In[1]:


# Load the usual things
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1.0 Data Cleaning
# 
# Our first step is to make sure the data is clean. Lets start with our "Accomodation Occupation and Capacity" data to see what's going on.

# In[2]:


accomodation = pd.read_csv("../NZ Data/Accomodation Occupancy and Capacity by Region - Original.csv", header = None)


# In[3]:


#Lets explore our data and see what's going on with it! 
accomodation


# ## 1.1.0 Accomodation Dataset Cleaning
# 
# It seems like there will be a good amount of cleaning that needs to be done. This was downloaded from a government website and as such there is some extra information provided that we don't necessarily need. These next few steps will be cleaning up the data and making it ready for exploration. We want to make the data machine readable and to really focus on the information at hand.
# 
# Some information on the dataset as a whole:
# - Monthly capacity is how much space is available on a monthly basis
# - Monthly occupancy is how much space is occupied on a monthly basis
# 
# From first glance, it seems that the data at the end may be extra strings that contain no numerical data and the first few rows are strings that explain the data in each column but is not populating each column. We will fill in the missing data.

# In[4]:


# Investigate the end of the dataframe
# We want to see where the strings end
accomodation.tail(28)


# In[5]:


# We know the last 27 rows are not useful, lets drop these and clean up the tail

accomodation.drop(accomodation.tail(27).index, inplace = True)
accomodation.tail()


# ## 1.1.1 Cleaning up the header columns
# 
# Now that we've cleaned up the tail of the data, lets clean up those first few rows we saw.
# We'll first shorten the length of the strings, fill in missing values, and then combine them into one string. 

# In[6]:


# Replace the lengthy Capacity and Occupancy strings with just occupancy and capacity

accomodation.iloc[2] = accomodation.iloc[2].str.replace(r'^Capacity.*$', 'Capacity', regex = True)
accomodation.iloc[2] = accomodation.iloc[2].str.replace(r'^Occupancy.*$', 'Occupancy', regex = True)
accomodation.head()


# In[7]:


# Now that the strings are shortened, lets fill in the missing rows.
# We know that each region will have a "capacity" and "occupancy"
# I will fill in the blanks of the locations by using fillna and forwardfill

accomodation.iloc[1] = accomodation.iloc[1].fillna(method = 'ffill')
accomodation.head()


# In[8]:


# We will do the same for the accomodation names

accomodation.iloc[0] = accomodation.iloc[0].fillna(method = 'ffill')
accomodation.head()


# In[9]:


# Lets turn them into one singular string so we can use them as one header column 
# We join these columns together

accomodation.iloc[0,1:] = accomodation.iloc[0:3,1:].agg(''.join)
accomodation.head()


# In[10]:


# Drop the 2 extra rows we no longer need

accomodation.drop(accomodation.index[[1,2]], inplace = True)

accomodation.head()


# In[11]:


# Lets set our first row to be our column headers making sure we drop it
accomodation = accomodation.rename(columns = accomodation.iloc[0]).loc[1:]
accomodation.head()


# ## 1.1.2 Creating our Datetime column
# 
# Now that we have our header columns, we need to make sure we have our datetime column for our timeseries data. We can see that the time is in the format of Year-Month but has the letter M in the middle signifying month. Lets prepare our data now!

# In[12]:


# Rename our NaN first column to Date and convert it to DateTime then set it as index 
accomodation = accomodation.rename(columns = {np.NaN:'Date'})
accomodation.head(2)


# In[13]:


#DateTime Conversion
accomodation['Date'] = pd.to_datetime(accomodation['Date'], yearfirst = True, format = '%YM%m')
accomodation.head(2)


# In[14]:


#Set date-time as Index
accomodation = accomodation.set_index("Date")
accomodation.head(2)


# In[15]:


#Save our data as cleaned for future usage - done once
#accomodation.to_csv('../NZ Data/Accomodation Data Cleaned.csv')


# ## 1.2.0 Cleaning and checking on our Arrivals data
# 
# Now that we've cleaned our data for accomdations, we should make sure to clean our data for oru visitor data! This data set looks visitor entry data from all ports of New Zealand. It also lists their purpose entry. We will run through this in less cells as the proces is similar to our process for the Accomodations data.

# In[16]:


# Load Data for Arrivals
arrivals = pd.read_csv('../NZ Data/Arrivals for All Visitors by Port and Purpose - Original.csv', header = None)


# In[17]:


arrivals


# In[18]:


arrivals.tail(40)


# In[19]:


# Dropping the bottom strings

arrivals.drop(arrivals.tail(39).index, inplace = True)
arrivals.tail(5)


# ## 1.2.1 String simplification and Date-Time
# 
# Similar to last dataset we'll clean up the strings by grouping the regions and then the purpose for their visit. Unlike the last dataset, only the reason for entry and port of entry will be kept. The data is known to be for all countries and for visitors of all countries.
# 
# We will drop the 2 rows we don't need and then forward fill and combine the two rows we do want to keep.
# 
# We will also be doing our date-time conversion in one cell to simplify the process.

# In[20]:


arrivals.head(5)


# In[21]:


arrivals.drop(arrivals.head(2).index, inplace = True)
arrivals.head(3)


# In[22]:


arrivals.iloc[0] = arrivals.iloc[0].fillna(method = 'ffill')
arrivals.head(3)


# In[23]:


# String joining, dropping extra rows, setting as header all in one cell

arrivals.iloc[0,1:] = arrivals.iloc[0:2,1:].agg(''.join)
arrivals.drop(arrivals.index[[1]], inplace = True)
arrivals = arrivals.rename(columns = arrivals.iloc[0]).iloc[1:]

arrivals.head(3)


# In[24]:


#Lets drop the "airport" from each of our columns

arrivals.columns = arrivals.columns.str.strip(r'airport$')
arrivals.head()


# In[25]:


# Datetime creation in one cell
arrivals = arrivals.rename(columns = {' ':'Date'})
arrivals['Date'] = pd.to_datetime(arrivals['Date'], yearfirst = True, format = '%YM%m')
arrivals = arrivals.set_index("Date")
arrivals.head(2)


# In[26]:


# Sanity Check for our data

arrivals


# ## 2.0 Data Exploration
# 
# Here we finally get to look at and explore our data for some initial thoughts. We will plot out the data and make sure it is numeric. Here we just want to look at all of our time series data as a whole and see pieces of data might be useful in our pursuit of forecasting. Not all the data may be complete neough for us to predict with and some of it will need further pre-processing in order to be ready. 
# 
# Before we can do that, we see that there are some data points where instead of NaN they were inputted as "..".
# 
# First, we will make convert these to NaN so that we are able to treat these as numeric data instead of having strings. Then we will be able to convert these dataframes to floats and plot them out on graphs.

# In[27]:


# Replacing the ".." with NaN so that we may convert to numeric
accomodation = accomodation.replace('..', np.nan)
arrivals = arrivals.replace('..', np.nan)


# In[28]:


# Checking our dtypes
accomodation.dtypes


# In[29]:


# converting accomodation to float
accomodation = accomodation.astype('float')
accomodation.dtypes


# In[30]:


# checking dtype for arrivals
arrivals.dtypes


# In[31]:


# converting from object to float
arrivals = arrivals.astype('float')
arrivals.dtypes


# In[32]:


#Save our data as cleaned for future usage - done once
#arrivals.to_csv('../NZ Data/Arrivals Data Cleaned.csv')


# ## 2.1 Exploring the Accomodation Dataset.
# 
# Now that we have converted to Float we can begin the real exploration! We will plot out the graphs in different chunks. Instead of graphing all 330 columns of data which would be unwieldy and time-consuming, we will plot out segments of our data in order to allow us to understand how each group of data really looks like visually.

# In[33]:


#Lets see what different kinds of occupany we have by looking at Auckland

accomodation.filter(regex=('AucklandOccupancy$')).columns


# For the accomodation dataset it looks like we have the following accomodations: "Total", "Hotels", "Motels", "Backpackers", and "Holiday Parks".
# 
# We'll also focus on occupancy. Capacity may be useful to predict in the future but for our initial visualization we are going to focus on occupancy and preparing for occupancy which will relate to the amount of capacity we need as well as our staffing needs.

# In[34]:


#We will start with our Total occupancy for ALL accomodation types

total_accomodation = accomodation.filter(regex=('^Total')).filter(regex=('Occupancy$')).columns
accomodation[total_accomodation].plot(subplots = True, grid = True, sharex=False,                                       layout = (int(len(total_accomodation)/3)+1, 3), figsize = (20,50));


# In[35]:


#lets look at hotels
accomodation[hotel_accomodation].plot(subplots = True, grid = True, sharex=False,                                       layout = (int(len(hotel_accomodation)/3)+1, 3), figsize = (20,50));


# In[36]:


#Lets look at motels
motel_accomodation = accomodation.filter(regex=('^Motel')).filter(regex=('Occupancy$')).columns

accomodation[motel_accomodation].plot(subplots = True, grid = True, sharex=False,                                       layout = (int(len(motel_accomodation)/3)+1, 3), figsize = (20,50));


# In[37]:


#Lets look at backpackers
backpacker_accomodation = accomodation.filter(regex=('^Backpackers')).filter(regex=('Occupancy$')).columns

accomodation[backpacker_accomodation].plot(subplots = True, grid = True, sharex=False,                                       layout = (int(len(backpacker_accomodation)/3)+1, 3), figsize = (20,50));


# In[38]:


#Lets look at Holiday Parks
holidaypark_accomodation = accomodation.filter(regex=('^Holiday')).filter(regex=('Occupancy$')).columns

accomodation[holidaypark_accomodation].plot(subplots = True, grid = True, sharex=False,                                       layout = (int(len(holidaypark_accomodation)/3)+1, 3), figsize = (20,50));


# ## 2.2 Thoughts on the Accomodation Dataset
# 
# I want to explore the initial insights I see from the graphing of each of these data sets. Some of these data sets are definitely too inconsistent to be used as for forecasting but with some of these data sets we may be able to impute missing data.
# 
# ### Total
# 
# The total data is very consistent. There seems to be no issues with missing data in these sets. The trends are definitely seasonal with a few perhaps even having an upward trend. We will definitely need to preprocess this information and remove the seasonal efects before using it for forecasting. 
# 
# ### Hotels
# 
# The hotel data is surprisingly sparse. There are many regions which are missing data. I would not be very confident using many region's data for forecasting. I think that the existing hotel data that is existing would be interesting to look at but not for the usage of forecasting. 
# 
# ### Motels
# 
# Motel data was very well documented. Over the course of the data, there seems to be no missing data for motels. I think that we begin to see some trends of increasing occupancy with obvious seasonal effects. I think that exploring motel forecasting could be useful for helping this specific accomodation prepare for an increasing influx of visitors. Motels may be the most impacted by a growing visitor population and gives us the best opportunity for making targeted forecasting.
# 
# 
# ### Backpackers
# 
# For some regions, backpacking is very consistent. With some regions missing backpacking data entirely. I think that there is little reason to really look into backpacking forecasting as these are going to be in the wilderness. The usual accomodations will be in the nature. The spaces are naturally limited and controlled by a wilderness authority. There is no way to really easily expand capacity but the occupancy is limited by nature. While seasonal, there seems to have been limited growth in this region.
# 
# ### Holiday Parks
# 
# Holiday parks that have data have very clear season effects. True to their name, these parks are generally most occupied during the holiday season. While many regions have sparse data there are still many regions which are able to give us good data across the timespan of the data. Still, there is little growh in this particular accomodation and historical data can probably be used to make some confident forecasting for the future as these parks open back up. 
# 
# ### Overall Thoughts
# 
# I think that I want to stick to using either total forecasting for overall knowledge across the industry to understand trends. I definitely want to dig deeper into the motels accomodations so that we can get a better understanding for this particular accomodation and help them out with their forecasting a bit more. 
# 
# I definitely think that focusing on certain regions like Auckland, Rotorua, and Queenstown could be useful as these are huge tourist areas. I will continue to dig into other regions and see which regions have the most complete data that maybe we can use for further exploration and analysis. 
# 
# The total dataset is very useful for that last part and I think that we can use that for forecasting regions that may begin to see the highest levels of tourism. WE can combine this with the next dataset which is the Visitor dataset and use that to help corroborate with our accomodation dataset.

# ## 3.1 Exploring the Arrivals Dataset.
# 
# Now that we've looked at our accomodations, lets explore our arrival numbers in a similar manner. If we can forecast this dataset alongside the accomodations data we can use both to prepare multiple industries at once. Knowing how many people may be incoming to New Zealand we can prepare multiple industries such as food service, accomodation, and even airport security.

# In[39]:


arrivals.head()


# In[40]:


#lets examine the reasons for entry for each location
arrivals.columns

#looks like we have business, holiday/vacation, visit friends, and total travel!


# In[41]:


#Lets look at TOTAL ENTRY!
total_entry = arrivals.filter(regex=('^TOTAL')).columns

arrivals[total_entry].plot(subplots = True, grid = True, sharex=False,                                       layout = (int(len(total_entry)/2)+1, 2), figsize = (20,50));


# In[42]:


#Lets look at business
business_entry = arrivals.filter(regex=('^Business')).columns

arrivals[business_entry].plot(subplots = True, grid = True, sharex=False,                                       layout = (int(len(business_entry)/2)+1, 2), figsize = (20,50));


# In[43]:


#Lets look at holiday
holiday_entry = arrivals.filter(regex=('^Holiday')).columns

arrivals[holiday_entry].plot(subplots = True, grid = True, sharex=False,                                       layout = (int(len(holiday_entry)/2)+1, 2), figsize = (20,50));


# In[44]:


#Lets look at visiting friends
visit_entry = arrivals.filter(regex=('^Visit')).columns

arrivals[visit_entry].plot(subplots = True, grid = True, sharex=False,                                           layout = (int(len(visit_entry)/2)+1, 2), figsize = (20,50));


# ## 3.2 Thoughts on the Tourism Data
# 
# Here are the initial thoughts for the tourism data to each of these ports. 
# 
# ### Total
# 
# We see a pretty consistent trend of growth overall. There is also definitely a seasonal trend to the data. We will have to preprocess for the seasonal and trends in order to analyse this data. We do see the sudden plummet due to the COVID-19 pandemic and that will be our challenge. To predict our growth forecast back to our normal levels.
# 
# ### Business
# 
# Business travel seems to be pretty consistent throughout the year with a definite growing trend. This makes sense as business is constant with seasons of highs and lows but with less overall impact on business travel. We definitely will need to adapt for this growth trend up until the COVID-19 pandemic.
# 
# ### Holiday
# 
# As expected of holiday travel we see a definite seasonal effect. However, we see slightly less of a growth trend impacting the data than with other types of travel. This is surprising since we expect holiday travel to New Zealand to continue to grow but the growth trends seem minimal compared to the effect of seasons.
# 
# ### Visiting Friends
# 
# Visiting friends seems to have both a minimal growth trend but a highly seasonal effect. This mirrors the holiday data and maybe we can use similar models for forecasting both because of it.
# 
# ### Overall Thoughts
# 
# Overall, the travel data seems to follow similar trends. This means that maybe focusing our efforts on forecasting "total" and "business" travel would be in our best interest. The ports of `Dunedin`, `Hamilton`, `Palmerton North`, `Rotorua`, and `Queenstown` all start later as these ports open up later. It may be a bit of a challenge to accurately forecast these ports with the limited data but our greatest challenging is forecasting post-pandemic. The goal of our project will be to see if we can project what travel will be like as ports in New Zealand open and what the rate of tourism will be like as we return to international travel.

# ## 4.0 Final Thoughts and Plans 
# 
# As we look at this data, I definitely have an idea of how I want to approach the forecasting.
# 
# I want to focus on regional growth trends for the accomodation and ports of entry. I think understanding how these two trends corrolate will be interesting. 
# 
# Moving forward, I definitely will need to pre-process out all seasonal data and trend data for multiple data sets. Instead of forecasting for every timeseries, I think creating a model for a few select models will be key. For the accomodation and the tourism data I want to focus on regions that are matched as well as datasets with more complete data. These datasets include the regions of `Auckland`, `Queenstown`, `Rotorua`, `Dunedin`, and `Wellington`. Amongst others. I will go into more detail for these regions in the pre-processing and modeling notebooks. 
# 
# I am excited to look at these datasets and begin forecasting.
