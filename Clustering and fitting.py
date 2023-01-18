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
    #year_data = year_data.rename(columns={"index":"Year"})
    
    return country_data, year_data, data





#With the use of the defined function to read into a panda dataframe
file = 'data.csv'
indicator = 'CO2 emissions (kt)'
df_countries, df_year, df_data = data_frame(file, indicator)

# print(df_countries)

#The years intended for used for the clustering
#cleaning of data and dropping NaN values
x = '1990'
y = '2019'
data = df_data.loc[df_data.index, ["Country Name", x, y]]
data1= data[[x, y]].dropna()
print(data1)





#conversion to an array
a = data1.to_numpy()
print(a.shape)

plt.figure()
plt.scatter(a[:,0], a[:,1])
plt.savefig('1990 and 2020 raw data.png')
plt.title('Raw Data')
plt.show()




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
print(labels)
centroid = k_means.cluster_centers_
print(centroid)


plt.plot(range(1, 10), WCSS)
plt.title('getting the cluster')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()



#Plotting of clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300,\
                n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(a)
print(y_kmeans)

plt.scatter(a[y_kmeans == 0, 0], a[y_kmeans == 0, 1], s = 50, c = 'purple')
plt.scatter(a[y_kmeans == 1, 0], a[y_kmeans == 1, 1], s = 50, c = 'orange')
plt.scatter(a[y_kmeans == 2, 0], a[y_kmeans == 2, 1], s = 50, c = 'green')
#plt.scatter(a[y_kmeans == 3, 0], a[y_kmeans == 3, 1], s = 50, c = 'blue')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1],\
            s = 50, c = 'black', marker = '+', label = 'Centroids')
plt.title('Cluster of CO2 emission')
plt.legend()
plt.show()



# define the true objective function
def objective(x, a, b, c, d):
    return a*x + b*x**4 + c*x**3 + d




df2 = df_year
df2 = df2.drop(df2.index[[0]],axis=0)
df2['Year'] = df2['index']

print(df2)
dfg = df2.to_numpy()

#converting from object to float
df2['Year'] = df2['Year'].astype('int64')
df2['China'] = df2['China'].astype('int64')

# fit exponential growth
popt, covar = opt.curve_fit(objective, df2["Year"], 
                            df2["China"])
print("Fit parameter", popt)



#To get an array of optimal values for the parameters which minimizes 
#the sum of squares of residuals.
#plotting of original data against the fit exponential data
df2["China_exp"] = objective(df2["Year"], *popt)


plt.figure()
plt.plot(df2["Year"], df2["China"], label="data")
plt.plot(df2["Year"], df2["China_exp"], label="fit")

plt.legend()
plt.xlabel("Year")
plt.ylabel("China")
plt.title("Final fit exponential growth")
plt.show()





# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)

low, up = err.err_ranges(df2["Year"], objective, popt, sigma)

plt.figure()
plt.title("Error Range")
plt.plot(df2["Year"], df2["China"], label="data")
plt.plot(df2["Year"], df2["China_exp"], label="fit")

plt.fill_between(df2["Year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.show()


