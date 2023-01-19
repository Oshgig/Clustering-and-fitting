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



# Assume we have a DataFrame called 'df'

# Get the column names
column_names = df_year.columns

# Print the column names
print(column_names)


df_year = df_year.drop(0, axis=0)
print(df_year)


x_axis = df_year['Year']
y_axis = df_year['China']

# Plot the bar chart
plt.bar(x_axis, y_axis)

# Add labels and title
#plt.figure(figsize=(6, 6))
plt.xlabel('Year')
plt.ylabel('CO2 emission')
plt.xticks(rotation=45, fontsize=7)
plt.title('China Co2 emission')
plt.savefig('CO2 emission.png', bbox_inches='tight', dpi=300)
plt.show()

