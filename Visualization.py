# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:52:25 2023

@author: Adeolu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 05:39:52 2023

@author: Adeolu
"""

import pandas as pd
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import err_ranges as err



def data_frame(file, indicator):
    """
    

    Parameters
    ----------
    file : A csv format
        The use of a csv document gotten from World bank.
    indicator : The Series name
        DESCRIPTION: The name of the series for use, the 
        imported data has various indicators.

    Returns
    -------
    country_data : The country name
        as a column.
    year_data : The year 
        as a column and returns the country as a row.

    """
    data = pd.read_csv(file)
    
    country_data = data
    
    data = data.drop(['Indicator Code', 'Country Code'], axis=1)
    
    year_data = data.T
    year_data = year_data.rename(columns=year_data.iloc[0])
    year_data = year_data.drop(index=year_data.index[0], axis=0)
    year_data = year_data.reset_index()
    year_data = year_data.rename(columns={"index":"Year"})
    
    return country_data, year_data, data





#With the use of the defined function to read into a panda dataframe
file = 'data.csv'
indicator = 'CO2 emissions (kt)'
df_countries, df_year, df_data = data_frame(file, indicator)





#Pre processing of data 
df_year = df_year.drop(0, axis=0)
print(df_year)



#Mergeing all the indicators in a single dataframe
#drop NaN values
df1 = df_year[df_year['Year'] == '1990']
df2 = df_year[df_year['Year'] == '2005']
df3 = df_year[df_year['Year'] == '2019']
frames = [df1, df2, df3]
df = pd.concat(frames)
df = df.dropna(axis=1)
print(df)




# Define the data for the chart
# Select the 'Angola', 'Brazil', and 'China' columns
categories = ['1990', '2005', '2019']
series1 = df['United States']
series2 = df['China']
series3 =  df['India']
series4 = df['Zambia']

# Setting the width of the bars
bar_width = 0.2

# Plot the series
plt.bar(categories, series1, width=bar_width, label='USA')
plt.bar(np.arange(len(categories))+bar_width, series2, width=bar_width, label='CHINA')
plt.bar(np.arange(len(categories))+(2*bar_width), series3, width=bar_width, label='INDIA')


plt.xlabel('Year')
plt.ylabel('Co2 emission')
plt.title('Co2 emission of three countries')
plt.legend()
plt.savefig('Trend.png', bbox_inches='tight', dpi=300)
plt.show()





