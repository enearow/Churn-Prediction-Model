# Churn-Prediction-Model
This is to determine the number of churners the organization has and how to make sure that customers dont churn considering how expensive it is to get new customers. So retention is very important for the organization.

#Import all the necessary Libraries for this project 

import pyodbc
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
import seaborn as sns
import plotly.offline as pyoff
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.offline import iplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
from lifetimes.utils import *
from lifetimes import BetaGeoFitter,GammaGammaFitter
from lifetimes.plotting import plot_probability_alive_matrix, plot_frequency_recency_matrix, plot_period_transactions, plot_cumulative_transactions,plot_incremental_transactions
from lifetimes.generate_data import beta_geometric_nbd_model
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases, plot_period_transactions,plot_history_alive
warnings.filterwarnings("ignore")


cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=DESKTOP-2NRFU9A;"
                      "Database=AdventureWorksDW2017;"
                      "Trusted_Connection=yes;")
                      
                      #Read dataframe from SQL server Native client 11.0.

data = pd.read_sql_query("""
Select DC.CustomerKey as CustomerID,
DG.EnglishCountryRegionName as Region,
Year(getdate())-Year(DC.BirthDate) AS Age,
CASE WHEN Year(getdate())-Year(DC.BirthDate) <= 40 THEN 'Youth'  WHEN Year(getdate())-Year(DC.BirthDate) <= 60 THEN 'Adult'
ELSE 'Senior Citizen' END Age_Band,
CASE WHEN  DC.MaritalStatus = 'M' THEN 'Married' ELSE 'Single' END MaritalStatus,
CASE WHEN DC.Gender = 'M' THEN 'Male' ELSE 'Female' END Gender,
DC.YearlyIncome,
DC.TotalChildren,
DC.EnglishEducation as Education,
DC.EnglishOccupation as Occuption,
CASE WHEN DC.HouseOwnerFlag = 1 THEN 'Yes' ELSE 'No' END HouseOwner,
DC.NumberCarsOwned,
DF.SalesAmount,
DC.CommuteDistance,
DF.SalesOrderLineNumber as Quantity,
DF.OrderDate,
DF.UnitPrice as Price,
CASE WHEN (Datediff(Month,DF.OrderDate,'2014-01-31')) >= 8 THEN 'Yes' ELSE 'No' END Churn,
DS.SalesReasonName
From [dbo].[DimCustomer] AS DC
LEFT JOIN [dbo].[FactInternetSales] AS DF
ON DF.CustomerKey = DC.CustomerKey
LEFT JOIN [dbo].[DimSalesReason] AS DS
ON DS.SalesReasonKey = DF.PromotionKey
LEFT JOIN [dbo].[DimGeography] AS DG
ON DG.GeographyKey = DC.GeographyKey
""", cnxn)

print('DataFrame:')
data.head()

print( data.shape)

#Check for missing values

data.isnull().sum()

# Get unique count for each variable
data.nunique()

# Check variable data types
data.dtypes

# the number of customers
data.CustomerID.nunique()

# Exploratory Data Analysis
Here our main interest is to get an understanding as to how the given attributes relate to the 'Churn' status.

labels = 'Churned', 'Retained'
sizes = [data.Churn[data['Churn']=='Yes'].count(), data.Churn[data['Churn']=='No'].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()


So about 38.4% of the customers have churned. So the baseline model could be to predict that 38.4% of the customers will churn. Given 38.4% is a small number, we need to ensure that the chosen model does predict with great accuracy this 38.4% as it is of interest to the AHG to identify and keep this bunch as opposed to accurately predicting the customers that are retained.

# We first review the 'Status' relation with categorical variables
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.countplot(x='Region', hue = 'Churn',data = data, ax=axarr[0][0])
sns.countplot(x='Age_Band', hue = 'Churn',data = data, ax=axarr[0][1])
sns.countplot(x='MaritalStatus', hue = 'Churn',data = data, ax=axarr[1][0])
sns.countplot(x='HouseOwner', hue = 'Churn',data = data, ax=axarr[1][1])
sns.countplot(x='Education', hue = 'Churn',data = data, ax=axarr[2][0])
sns.countplot(x='SalesReasonName', hue = 'Churn',data = data, ax=axarr[2][1])

<matplotlib.axes._subplots.AxesSubplot at 0x1a2797206a0>

We note the following:
Majority of the data is from persons from United States. However, the proportion of churned customers is with inversely related to the population of customers alluding to the AHG possibly having a problem (maybe not enough customer service resources allocated) in the areas where it has fewer clients.
The proportion of Married customers churning is also greater than that of Single customers Interestingly, majority of the customers that churned are those who own houses. Given that majority of the customers are HouseOwners could prove this to be just a coincidence.
The lager proportion of customers who purcahse items based on the price will churn. however, the AHG to look at pricing policy and restrategizes either by making provisions for items with low price but of good qualities too.
Customers with Bachelors degree are potential churnners, this group makes up the major share of the market and needed to be managed very well.
There are no clear difference in the gender of customers and Female and Male customers have same churning proportions, so customer customer gender has little or nothing to do with the churning attitude of the customer but the product quality and price of the product could create room for it.

# Relations based on the continuous data attributes
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='Age',x = 'Churn', hue = 'Churn',data = data , ax=axarr[0][0])
sns.boxplot(y='TotalChildren',x = 'Churn', hue = 'Churn',data = data, ax=axarr[0][1])
sns.boxplot(y='Quantity',x = 'Churn', hue = 'Churn',data = data, ax=axarr[1][0])
sns.boxplot(y='Price',x = 'Churn', hue = 'Churn',data = data, ax=axarr[1][1])
sns.boxplot(y='YearlyIncome',x = 'Churn', hue  = 'Churn',data = data, ax=axarr[2][0])
sns.boxplot(y='SalesAmount',x = 'Churn', hue  = 'Churn',data = data, ax=axarr[2][1])

<matplotlib.axes._subplots.AxesSubplot at 0x1a27b6f83c8>

We note the following:
The older customers are churning at more than the younger ones alluding to a difference in service preference in the age categories. The company may need to review their target market or review the strategy for retention
Neither the product nor the salary has a significant effect on the likelihood to churn.
Worryingly, the AHG is losing customers with significant Revenue which is likely to hit their Profit.
Neither the Age nor the salary has a significant effect on the likelihood to churn.
Pricing is the major reason for customer churning. AHG needs to come up with new pricing strategy that will match up with what competitors are offering.


# Calculating RFM values

#extract year, month and day
data['OrderDay'] = data.OrderDate.apply(lambda x: dt.datetime(x.year, x.month, x.day))
data.head()

# calculate RFM values
rfm = data.groupby('CustomerID').agg({
    'OrderDate' : lambda x: (pin_date - x.max()).days,
    'Quantity' : 'count', 
    'TotalSum' : 'sum'})
# rename the columns
rfm.rename(columns = {'OrderDate' : 'Recency', 
                      'Quantity' : 'Frequency', 
                      'TotalSum' : 'Monetary'}, inplace = True)
rfm.head()

As the three columns are grouped by customers and count the days from the max date value, Recency is the days since the last purchase of a customer. Frequency is the number of purchases of a customer and Monetary is the total amount of spend of a customer.

# RFM quartiles
Let's group the customers based on Recency and Frequency. We will use quantile values to get three equal percentile groups and then make three separate groups. As the lower Recency value is the better, we will label them in decreasing order.

# create labels and assign them to tree percentile groups 
r_labels = range(4, 0, -1)
r_groups = pd.qcut(rfm.Recency, q = 4, labels = r_labels)
f_labels = range(1, 5)
f_groups = pd.qcut(rfm.Frequency, q = 4, labels = f_labels)
m_labels = range(1, 5)
m_groups = pd.qcut(rfm.Monetary, q = 4, labels = m_labels)

m_groups.head()

# make a new column for group labels
rfm['R'] = r_groups.values
rfm['F'] = f_groups.values
rfm['M'] = m_groups.values

# sum up the three columns
rfm['RFM_Segment'] = rfm.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']), axis = 1)
rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis = 1)
rfm.head()

With this value, we can go further analysis such as what is the average values for each RFM values or leveling customers in total RFM score.

# calculate average values for each RFM
rfm_agg = rfm.groupby('RFM_Score').agg({
    'Recency' : 'mean',
    'Frequency' : 'mean',
    'Monetary' : ['mean', 'count']
})
rfm_agg.round(1).head()

The final score will be the aggregated value of RFM and we can make groups based on the RFM_Score

# assign labels from total score
score_labels = ['Green', 'Bronze', 'Silver', 'Gold']
score_groups = pd.qcut(rfm.RFM_Score, q = 4, labels = score_labels)
rfm['RFM_Level'] = score_groups.values
rfm.head()

# Customer Segmentation with Kmeans¶

# plot the distribution of RFM values
fig, axarr = plt.subplots(3, 1, figsize=(20, 30))
plt.subplot(3, 1, 1); sns.distplot(rfm.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()

# define function for the values below 0
def neg_to_zero(x):
    if x <= 0:
        return 1
    else:
        return x

# apply the function to Recency and MonetaryValue column 
rfm['Recency'] = [neg_to_zero(x) for x in rfm.Recency]
rfm['Monetary'] = [neg_to_zero(x) for x in rfm.Monetary]
rfm.head()

# unskew the data
rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)
rfm_log.head()

rfm_log.describe()

# scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

# transform into a dataframe
rfm_scaled = pd.DataFrame(rfm_scaled, index = rfm.index, columns = rfm_log.columns)
rfm_scaled.head()

# plot the distribution of RFM values
fig, axarr = plt.subplots(3, 1, figsize=(20, 30))
plt.subplot(3, 1, 1); sns.distplot(rfm_scaled.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm_scaled.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm_scaled.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()

# K-means clustering
With the Elbow method, we can get the optimal number of clusters.

# the Elbow method
wcss = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters= k, init= 'k-means++', max_iter= 300)
    kmeans.fit(rfm_scaled)
    wcss[k] = kmeans.inertia_
    
    # plot the WCSS values
sns.pointplot(x = list(wcss.keys()), y = list(wcss.values()))
plt.xlabel('K Numbers')
plt.ylabel('WCSS')
plt.show()

# clustering
clus = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 300,)
clus.fit(rfm_scaled)

# Assign the clusters to datamart
rfm['K_Cluster'] = clus.labels_
rfm.head()

# Visualization¶

# assign cluster column 
rfm_scaled['K_Cluster'] = clus.labels_
rfm_scaled['RFM_Level'] = rfm.RFM_Level
rfm_scaled.reset_index(inplace = True)

rfm_scaled.head()

# melt the dataframe
rfm_melted = pd.melt(frame= rfm_scaled, id_vars= ['CustomerID', 'RFM_Level', 'K_Cluster'], var_name = 'Metrics', value_name = 'Value')
rfm_melted.head()

# a snake plot with RFM
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'RFM_Level', data = rfm_melted)
plt.title('Snake Plot of RFM')
plt.legend(loc = 'upper right')

# a snake plot with K-Means
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'K_Cluster', data = rfm_melted)
plt.title('Snake Plot of RFM')
plt.legend(loc = 'upper right')

# a snake plot with K-Means
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'K_Cluster', data = rfm_melted)
plt.title('Snake Plot of K_cluster')
plt.legend(loc = 'upper right')

# Heatmap

# the mean value for each cluster
cluster_avg = rfm.groupby('RFM_Level').mean().iloc[:, 0:3]
cluster_avg.head()

# the mean value in total 
total_avg = rfm.iloc[:, 0:3].mean()
total_avg

# the proportional mean value
prop_rfm = cluster_avg/total_avg - 1
prop_rfm

# heatmap
sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True)
plt.title('Heatmap of RFM quantile')
plt.plot()

# the mean value for each cluster
cluster_avg_K = rfm.groupby('K_Cluster').mean().iloc[:, 0:3]
cluster_avg_K.head()

# the proportional mean value
prop_rfm_K = cluster_avg_K/total_avg - 1
prop_rfm_K

# heatmap
sns.heatmap(prop_rfm_K, cmap= 'Blues', fmt= '.2f', annot = True)
plt.title('Heatmap of K-Means')
plt.plot()

3D Scatter plot with Plotly
We can also check how the clusters are distributed across each RFM value.


sub = []

myColors = ['#db437b', '#d3d64d', '#568ce2', '#b467bc']

for i in range(3):

    df = rfm_scaled[rfm_scaled.K_Cluster == i]

    x = df.Recency

    y = df.Frequency

    z = df.Monetary

    color = myColors[i]

    

    trace = go.Scatter3d(x = x, y = y, z = z, name = str(i),

                         mode = 'markers', marker = dict(size = 5, color = color, opacity = .7))

    sub.append(trace)

data = [sub[0], sub[1], sub[2]]

layout = go.Layout(margin = dict(l = 0, r = 0, b = 0, t = 0),

                  scene = dict(xaxis = dict(title = 'Recency'), yaxis = dict(title = 'Frequency'), zaxis = dict(title = 'Monetary')))

fig_1 = go.Figure(data = data, layout = layout)

iplot(fig_1)

sub_2 = []

level = rfm_scaled.RFM_Level.tolist()

for i in range(4):

    df = rfm_scaled[rfm_scaled.RFM_Level == level[i]]

    x = df.Recency

    y = df.Frequency

    z = df.Monetary

    color = myColors[i]

    

    trace = go.Scatter3d(x = x, y = y, z = z, name = level[i], 
                         mode = 'markers', marker = dict(size = 5, color = color, opacity = .7))

    sub_2.append(trace)

data = [sub_2[0], sub_2[1], sub_2[2], sub_2[3]]

layout = go.Layout(margin = dict(l = 10, r = 0, b = 0, t = 0), 

                  scene = dict(xaxis = dict(title = 'Recency'), yaxis = dict(title = 'Frequency'), zaxis = dict(title = 'Monetary')))

fig_2 = go.Figure(data = data, layout = layout)

iplot(fig_2)

Conclusion.
We talked about how to get RFM values from customer purchase data, and we made two kinds of segmentation with RFM quantiles and K-Means clustering methods. With this result, we can now figure out who are our ‘golden’ customers, the most profitable groups. This also tells us on which customer to focus on and to whom give special offers or promotions for fostering loyalty among customers. We can select the best communication channel for each segment and improve new marketing strategies.

1
​
