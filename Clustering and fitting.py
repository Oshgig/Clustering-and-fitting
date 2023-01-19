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
from scipy import interpolate
import err_ranges as err
from sklearn.linear_model import LinearRegression



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
    #year_data = year_data.rename(columns={"index":"Year"})
    
    return country_data, year_data, data





#With the use of the defined function to read into a panda dataframe
file = 'data.csv'
indicator = 'CO2 emissions (kt)'
df_countries, df_year, df_data = data_frame(file, indicator)



#The years intended for used for the clustering
#cleaning of data and dropping NaN values
x = '1990'
y = '2019'
data = df_data.loc[df_data.index, ["Country Name", x, y]]
data1= data[[x, y]].dropna()






#conversion to an array
# scatter plot for all countries of years 1990 and 2020
a = data1.to_numpy()
print(a)

plt.figure()
plt.scatter(a[:,0], a[:,1])
plt.savefig('1990 and 2019 raw data.png')
plt.title('Scatter plot of all countries')
plt.show()



#normalizing the data can make a big difference in the results. 
#Some clustering algorithms are sensitive to the scale of the data, 
#and this sensitivity can be reduced by normalizing the data.
a = a / np.sqrt(np.sum(a ** 2))
print(a)


#To get the number of clusters 
#random_state creates random number anytime it is ran
#The range represents the x-axis which determines the number of clusters
WCSS = []
for k in range(1,10):
    k_means = cluster.KMeans(n_clusters=k, init='k-means++',\
                             max_iter=300, random_state=0)
    k_means.fit(a)
    WCSS.append(k_means.inertia_)
    
    

#To get cluster counts
#Centroid to get the mean of clusters
labels = k_means.labels_
print('label is',labels)
centroid = k_means.cluster_centers_
print('centroid is',centroid)


plt.plot(range(1, 10), WCSS)
plt.title('Elbow')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.savefig('Number of cluster.png', bbox_inches='tight', dpi=300)
plt.show()



#Plotting of clusters
# using the number of clusters gotten
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300,\
                n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(a)
print(y_kmeans)

plt.scatter(a[y_kmeans == 0, 0], a[y_kmeans == 0, 1], s = 50, c = 'purple',\
            label = 'Cluster 0')
plt.scatter(a[y_kmeans == 1, 0], a[y_kmeans == 1, 1], s = 50, c = 'orange',\
            label = 'Cluster 1')
plt.scatter(a[y_kmeans == 2, 0], a[y_kmeans == 2, 1], s = 50, c = 'green',\
            label = 'Cluster 2')
#plt.scatter(a[y_kmeans == 3, 0], a[y_kmeans == 3, 1], s = 50, c = 'blue')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],\
            s = 50, c = 'black', marker = '+', label = 'Centroids')
plt.title('Clustering of CO2 emission')
plt.legend()
plt.savefig('Cluster.png', bbox_inches='tight', dpi=300)
plt.show()





# define the true objective function
def objective(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    
    return scale * np.exp(growth * (t-1950)) 
    
    
    

#Pre processing of  data
df2 = df_year
df2 = df2.drop(df2.index[[0]],axis=0)
df2['Year'] = df2['index']




#converting from object to float
# setting the x-axis to the year and y axis to co2 emission of china
df2['Year'] = df2['Year'].astype('int64')
df2['China'] = df2['China'].astype('int64')
x_axis = df2['Year'] 
y_axis = df2['China']
x1 = x_axis.to_numpy()
y2 = y_axis.to_numpy()



# fit exponential growth
popt, covar = opt.curve_fit(objective, x_axis, 
                            y_axis)
print("Fit parameter", popt)




# fit exponential growth
# usetting p0 as the popt values
popt, covar = opt.curve_fit(objective, x_axis, 
                            y_axis, p0=[2.4e6, 0.056])
print("Fit parameter", popt)

df2["China_exp"] = objective(df2["Year"], *popt)

plt.figure()
plt.plot(x_axis, y_axis, label="data")
plt.plot(x_axis, df2['China_exp'], label="fit")

plt.legend()
plt.xlabel("Year")
plt.ylabel("Co2 emission")
plt.title("China Final fit exponential growth")
plt.savefig('China fit growth.png', bbox_inches='tight', dpi=300)
plt.show()



# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
# print(sigma)

low, up = err.err_ranges(x_axis, objective, popt, sigma)
# print('lowandup',covar)

# low, up = err.err_ranges(df_pop["date"], logistics, popt, sigma)



##create a line plot of the original data, the fit and the
#error ranges
plt.figure()
plt.title("Fill between Exponential growth")
plt.plot(x_axis, y_axis, label="data")
plt.plot(x_axis, df2["China_exp"], label="fit")

plt.fill_between(x_axis, low, up, alpha=0.5)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Co2 emission")
plt.savefig('Fill between.png', bbox_inches='tight', dpi=300)
plt.show()




# Create a linear regression object
model = LinearRegression()
print('gggg',np.shape(x1))
print(np.shape(y2))

x1 = x1.reshape(-1, 1)
#y2 = y2.reshape(1, -1)

# Fit the model to the data
vvv = model.fit(x1,y2)
print(vvv)


# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(np.min (x1), np.max(x1) + 1, 1)
x_line = x_line.reshape(-1, 1)


# Calculate the output for the range
df2['China_exp']= df2['China_exp'].to_numpy()
y_line = df2['China_exp']


# Use interp1d to create a function for interpolating y values
interp_func = interpolate.interp1d(x_line.flatten(), y_line)
predict_year = 2000

#initialize a list to add the population growth predictions
prediction = [] 
predict = interp_func(predict_year).item()
prediction.append(predict)

#print out the value of the prediction
print(prediction)


# create a line plot for the mapping function
plt.plot(x_axis, y_axis, label="data")
plt.plot(x_line, y_line, '--', color='black', label='fit')
plt.scatter(predict_year, prediction, marker=('+'), s=100, color='red', 
            label=f"co2 emission of {predict_year}prediction is {predict}.")
plt.title('Prediction of future errors')
plt.legend(loc = 'upper left')
plt.savefig('curve fit.png', bbox_inches='tight', dpi=300)
plt.show()



