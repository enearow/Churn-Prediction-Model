#!/usr/bin/env python
# coding: utf-8

# # Project
# 
# To Develop a predictive model using Python Modeller to automatically notify AHG leaders 90 days before a customer is due to churn, i e when a customer’s churn propensity exceeds a specific threshold, say 0 7

# ### Import all the necessary Libraries for this project

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt
import pylab
import pyodbc 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# data colelction and preprocessing
from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
# for data visualisation and statistical analysis
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression


# In[2]:


#Create a connection from Python to SQL
cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=DESKTOP-2NRFU9A;"
                      "Database=AdventureWorksDW2017;"
                      "Trusted_Connection=yes;")


# # Import all the required tables from SQL Server DW

# In[3]:


#Import Factinternetsales from AHG SQL Datawarehouse
fctintsales = pd.read_sql_query("""
SELECT
[CustomerKey]
,[OrderQuantity]
,[SalesAmount]
,[OrderDate]
,[SalesTerritoryKey]
,[SalesOrderNumber]
,[UnitPrice]
FROM [dbo].[FactInternetSales]""", cnxn)


# In[4]:


#Its a good practive to visualise the newly created datasets
fctintsales.head()


# In[5]:


#Import Customer Table SQL datawarehouse using the existing connections
custtbl = pd.read_sql_query("""
SELECT
[CustomerKey]
,[BirthDate]
,[MaritalStatus]
,[Gender]
,[YearlyIncome]
,[TotalChildren]
,[NumberChildrenAtHome]
,[EnglishEducation]
,[EnglishOccupation]
,[HouseOwnerFlag]
,[NumberCarsOwned]
,[DateFirstPurchase]
,[CommuteDistance]
FROM [dbo].[DimCustomer]""", cnxn)


# In[6]:


#Its a good practive to visualise the newly created datasets using .head() function
# .dytpes function will return the datatypes

custtbl.head(5)
#custtbl.dtypes


# In[7]:


custtbl.dtypes


# In[8]:


#Import Factinternetsalesreason from SQL Datawarehouse
fctinsalesreason = pd.read_sql_query("""
SELECT *
FROM [dbo].[FactInternetSalesReason]""", cnxn)


# In[9]:


#Its a good practive to visualise the newly created datasets using the .head(x) function
fctinsalesreason.head(5)


# In[10]:


#Import salesreason from SQL Datawarehouse
salesreason = pd.read_sql_query("""
SELECT *
FROM [dbo].[DimSalesReason]""", cnxn)


# In[11]:


#Its a good practive to visualise the newly created datasets using the .head(x) function
salesreason.head(20)


# ### Summarise customer transactions and tag customers based on the churn criteria given (8 months above)

# In[12]:


fctintsales[fctintsales['CustomerKey'] == 11000]


# For us to get the churnners and returned it is very important we prepare our fact internet sales data by getting the time between various orders, group it by sales order number, customer key and order date and aggregate it by the sum of sales amount and order quantity thereafter we drive the date difference.

# In[13]:


#Let get time between various orders
Timebtworders = fctintsales #Asign fctintsales to enable us pick out the needed columns
Timebtworders = Timebtworders[['SalesOrderNumber','CustomerKey','OrderDate','SalesAmount','OrderQuantity','UnitPrice']]


# In[14]:


#Group the time between various orders by sales order number, customerkey and orderdate aggregate by sum
#order quantity and sales amount
Timebtworders = Timebtworders.groupby(['SalesOrderNumber','CustomerKey','OrderDate','UnitPrice']).agg({'SalesAmount':'sum', 
                         'OrderQuantity':'sum'})


# In[15]:


#Display the new table 
Timebtworders.head(5)


# In[16]:


#Join the field names to the newly aggregate fields names
Timebtworders.columns = Timebtworders.columns.ravel()


# In[17]:


#Reset the index
Timebtworders = Timebtworders.reset_index()


# In[18]:


Timebtworders[Timebtworders['CustomerKey'] == 11000]


# From the result of the above code you could see differece in the order quantity for various order numbers and date  

# In[19]:


#Lets sort by customerkey and order date for easy arrangement
Timebtworders.sort_values(['CustomerKey','OrderDate']).head()


# In[20]:


#Lets get the difference in the order date for every customer
Timebtworders['diff'] = Timebtworders.sort_values(['CustomerKey','OrderDate']).groupby('CustomerKey')['OrderDate'].diff()


# In[21]:


#Display the table to show the difference in date
Timebtworders.sort_values(['CustomerKey','OrderDate']).head()


# From the out put above you could see that the 'Diff" is not in the right structure lets convert it properly to float for easy calculation.

# In[22]:


#lets specify the dtypes to timedelta64[D] for days
Timebtworders['diff'] = Timebtworders['diff'].astype('timedelta64[D]')


# In[23]:


#Display the result
Timebtworders.sort_values(['CustomerKey','OrderDate']).head()


# In[24]:


#Let see the data types we have before we can proceed 
Timebtworders.dtypes


# In[25]:


Timebtworders['diff'] = Timebtworders['diff'].fillna(0).astype(int) #Replace Na with 0 to enable uniformity in the column


# In[26]:


Timebtworders.sort_values(['CustomerKey','OrderDate']).head() #Display the outcome


# The Diff column has changed to all integer making it ready for use
# 
# Let group the sum, max, min, mean and median for SalesAmount, OrderQuantity, OrderDate and diff for every customerKey

# In[27]:


new_fctintsales = Timebtworders.groupby('CustomerKey').agg({'SalesAmount':['sum', 'max', 'min','mean','median'], 
                         'OrderQuantity':['sum', 'max', 'min','mean','median'], 
                         'OrderDate':'max','diff':['sum', 'max', 'min','mean','median']})


# In[28]:


#Its a good practive to visualise the newly created datasets using the .head(x) function
new_fctintsales.head(5)


# From the result above, the minimume quantity purcahsed by each customer is 1 and the last purcahse date clearly showing under maximume orderdate. 

# In[29]:


#Join the field names to the newly aggregate fields names
new_fctintsales.columns = ["_".join(new_fctintsales) for new_fctintsales in new_fctintsales.columns.ravel()]


# In[30]:


#Reset the index
new_fctintsales = new_fctintsales.reset_index()


# In[31]:


#Its a good practive to visualise the newly created datasets using the .head(x) function
new_fctintsales.head(10)


# In[32]:


#Merging the aggregated field with customer table
result = pd.merge(new_fctintsales,
                 custtbl,
                 on='CustomerKey')
result.head()


# In[33]:


#Prepare Salesreason data and merge required tables
reason = pd.merge(fctinsalesreason,
                 salesreason,
                 on='SalesReasonKey')
reason.head()


# In[34]:


reason.sort_values(['SalesOrderNumber']).head()


# In[35]:


#Dummify the Salesreasonname so that the values becomes variables
reason_ = pd.get_dummies(reason.SalesReasonName,columns=['Price', 'On Promotion', 'Magazine Advertisement','TV', 'Manufacturer', 'Review','Demo Event', 'Sponsorship', 'Quality', 'Other'])


# In[36]:


#Concatenate reason and reason_
reason = pd.concat([reason, reason_], axis=1)


# In[37]:


#Its a good practive to visualise the newly created datasets using the .tail(x) function will return the x bottom rows
reason.tail(10)


# In[38]:


reason.sort_values(['SalesOrderNumber']).head()


# In[39]:


#show unique salesreason name
salesreason.SalesReasonName.unique()


# In[40]:


#Its a good practive to visualise the newly created datasets using the .head(x) function
reason.head(5)


# In[41]:


#Re Import Factinternetsales from SQL Datawarehouse
fctintsales = pd.read_sql_query("""
SELECT
[CustomerKey]
,[SalesOrderNumber]
FROM [dbo].[FactInternetSales]""", cnxn)


# In[42]:


#Join factinternetsales back to prepared Salesreason data
reason = pd.merge(fctintsales,
                 reason,
                 on='SalesOrderNumber')


# In[43]:


reason.head(10)


# In[44]:


#Aggregate Sales Reason to prevent duplication of customer information
reason = reason.groupby('CustomerKey').agg({'Manufacturer':'max', 
                         'On Promotion':'max', 
                         'Other':'max',
                          'Price':'max',
                            'Quality':'max',
                              'Review':'max',
                                'Television  Advertisement':'max'}).reset_index()


# In[45]:


#Its a good practive to visualise the newly created datasets using the .head(x) function
reason.head(5)


# In[46]:


#left join main data back to summarise sales reason data
final_data = pd.merge(result,
                 reason,
                 on='CustomerKey', how='left')


# In[47]:


#Its a good practive to visualise the newly created datasets using the .head(x) function
final_data.head(5)


# ## Get Customer Age, Tenure and Churn Columns
# 
# We Need to get Tenure of the transaction in months, what we need to do is to minus the last date of transaction which is the maximum OrderDate  from the first purcase date divide in months. The last purchase date is the Maximum Order date, and we are using 2013-06-01 as our cuttoff date for transaction which we will use to get the Churn field. 

# In[48]:


#derive age, tenure and churn field
final_data['Tenure_Months'] = ((final_data['OrderDate_max'].dt.date -  final_data['DateFirstPurchase'])/np.timedelta64(1, 'M')).astype(int)
final_data['Age_years'] = ((final_data['OrderDate_max'].dt.date -  final_data['BirthDate'])/np.timedelta64(1, 'Y')).astype(int)
final_data['Last_purchase_date'] = final_data['OrderDate_max'].dt.date
final_data['Cuttoff_date'] = '2013-06-01'


# In[49]:


#Its a good practive to visualise the newly created datasets using the .head(x) function
final_data.head(5)
final_data.dtypes


# Cuttoff Date and Last Purchase date are not in the right data types, we need to format these columns from object to date format. We will use the code below to achieve it.

# In[50]:


#Convert dates from objects to date formats
final_data['Last_purchase_date'] = pd.to_datetime(final_data['Last_purchase_date'],format="%Y-%m-%d")
final_data['Cuttoff_date'] = pd.to_datetime(final_data['Cuttoff_date'],format="%Y-%m-%d")


# In[51]:


#Create Churn field (Target variable)
#We will categorize it to 1 and 0, which is yes or no
final_data['Churn'] = np.where(final_data['Last_purchase_date'] <= final_data['Cuttoff_date'], 1, 0) 


# In[52]:


#Its a good practive to visualise the newly created datasets using the .head(x) function
final_data.head(5)


# In[53]:


#dummfy all categorical variables
#final_data.dtypes
final_data = pd.get_dummies(final_data, columns=['MaritalStatus','Gender','EnglishEducation','EnglishOccupation','HouseOwnerFlag','CommuteDistance'], drop_first = False)


# In[54]:


#Its a good practice to visualise the newly created datasets using the .head(x) function
final_data.head(5)


# In[55]:


#Print the  variables name
print(list(final_data.columns))


# In[56]:


#Count the number of columns in our dataframe
len(final_data.columns)


# In[57]:


# Check for missing values
#final_data.isnull().values.any()
final_data.columns[final_data.isnull().any()].tolist()


# # Exploratory Data Analysis

# In[58]:


#visualize Churn
labels = 'Churned', 'Retained'
sizes = [final_data.Churn[final_data['Churn']==1].count(), final_data.Churn[final_data['Churn']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()


# ## Churn 
# 
# So about 30.3% of the customers have churned. So the baseline model could be to predict that 30.3% of the customers will churn. Given 30.3% is a small number, we need to ensure that the chosen model does predict with great accuracy this 30.3% as it is of interest to the AHG to identify and keep this bunch as opposed to accurately predicting the customers that are retained.

# In[59]:


#Churn by customer yearly income
plt.figure(figsize=(10,6))
final_data[final_data['Churn']==1]['YearlyIncome'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Churn=1')
final_data[final_data['Churn']==0]['YearlyIncome'].hist(alpha=0.5,color='red',
                                              bins=30,label='Churn=0')
plt.legend()
plt.xlabel('Yearly Income')


# Customer yearly income is one of the determinant of churnners the propensity to churn is more with customers with 60,000 yearly income.

# In[60]:


#Churn by customer yearly income
plt.figure(figsize=(10,6))
final_data[final_data['Churn']==1]['OrderQuantity_sum'].hist(alpha=0.5,color='red',
                                              bins=30,label='Churn=1')
final_data[final_data['Churn']==0]['OrderQuantity_sum'].hist(alpha=0.5,color='black',
                                              bins=30,label='Churn=0')
plt.legend()
plt.xlabel('OrderQuantity_sum')


# In[61]:


final_data.boxplot('SalesAmount_sum','Churn',rot = 30,figsize=(5,6))


# In[62]:


#reviewing the no of records in the target field - Churn
final_data['Churn'].value_counts()


# ## Unsupervised Technique
# 
# As part of feature engineering, we will use RFM analysis to create recency, frequency and monetary(lifetime value) attributes for each customer. Using the derived RFM variables or RFM Score and create a clustering model  using K means algorithms to generate customer segments. Finally use elbow method to select the best number of clusters

# In[63]:


#extract year, month and day
Timebtworders = Timebtworders[['SalesOrderNumber','CustomerKey','OrderDate','SalesAmount','OrderQuantity','UnitPrice']]


# In[64]:


#extract year, month and day
Timebtworders['OrderDay'] = Timebtworders.OrderDate.apply(lambda x: dt.datetime(x.year, x.month, x.day))
Timebtworders.head()


# In[65]:


# print the time period
print('Min : {}, Max : {}'.format(min(Timebtworders.OrderDay), max(Timebtworders.OrderDay)))
print('The last transaction date is January 28th 2014')


# In[66]:


# pin the last date
pin_date = max(Timebtworders.OrderDay) + dt.timedelta(1)


# In[67]:


# Create total spend dataframe
Timebtworders['TotalSum'] = Timebtworders.OrderQuantity * Timebtworders.UnitPrice
Timebtworders.head()


# ## Calculating RFM values

# In[68]:


# calculate RFM values
rfm = Timebtworders.groupby('CustomerKey').agg({
    'OrderDate' : lambda x: (pin_date - x.max()).days,
    'OrderQuantity' : 'count', 
    'TotalSum' : 'sum'})
# rename the columns
rfm.rename(columns = {'OrderDate' : 'Recency', 
                      'OrderQuantity' : 'Frequency', 
                      'TotalSum' : 'Monetary'}, inplace = True)
rfm.head()


# As the three columns are grouped by customers and count the days from the max date value, Recency is the days since the last purchase of a customer. Frequency is the number of purchases of a customer and Monetary is the total amount of spend of a customer.

# ## RFM quartiles
# 
# Let's group the customers based on Recency and Frequency. We will use quantile values to get three equal percentile groups and then make three separate groups. As the lower Recency value is the better, we will label them in decreasing order.

# In[69]:


# create labels and assign them to tree percentile groups 
r_labels = range(4, 0, -1)
r_groups = pd.qcut(rfm.Recency, q = 4, labels = r_labels)
f_labels = range(1, 5)
f_groups = pd.qcut(rfm.Frequency, q = 4, labels = f_labels)
m_labels = range(1, 5)
m_groups = pd.qcut(rfm.Monetary, q = 4, labels = m_labels)

m_groups.head()


# In[70]:


# make a new column for group labels
rfm['R'] = r_groups.values
rfm['F'] = f_groups.values
rfm['M'] = m_groups.values


# In[71]:


# sum up the three columns
rfm['RFM_Segment'] = rfm.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']), axis = 1)
rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis = 1)
rfm.head()


# With this value, we can go further analysis such as what is the average values for each RFM values or leveling customers in total RFM score.

# In[72]:


# calculate average values for each RFM
rfm_agg = rfm.groupby('RFM_Score').agg({
    'Recency' : 'mean',
    'Frequency' : 'mean',
    'Monetary' : ['mean', 'count']
})
rfm_agg.round(1).head()


# In[73]:


# assign labels from total score
# The final score will be the aggregated value of RFM and we can make groups based on the RFM_Score
score_labels = ['Green', 'Bronze', 'Silver', 'Gold']
score_groups = pd.qcut(rfm.RFM_Score, q = 4, labels = score_labels)
rfm['RFM_Level'] = score_groups.values
rfm.head()


# ## Customer Segmentation with Kmeans

# In[74]:


# plot the distribution of RFM values
fig, axarr = plt.subplots(3, 1, figsize=(20, 30))
plt.subplot(3, 1, 1); sns.distplot(rfm.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()


# In[75]:


# define function for the values below 0
def neg_to_zero(x):
    if x <= 0:
        return 1
    else:
        return x


# In[76]:


# apply the function to Recency and MonetaryValue column 
rfm['Recency'] = [neg_to_zero(x) for x in rfm.Recency]
rfm['Monetary'] = [neg_to_zero(x) for x in rfm.Monetary]
rfm.head()


# In[77]:


# unskew the data
rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)
rfm_log.head()


# In[78]:


rfm_log.describe()


# In[79]:


# scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)


# In[80]:


# transform into a dataframe
rfm_scaled = pd.DataFrame(rfm_scaled, index = rfm.index, columns = rfm_log.columns)
rfm_scaled.head()


# In[81]:


# plot the distribution of RFM values
fig, axarr = plt.subplots(3, 1, figsize=(20, 30))
plt.subplot(3, 1, 1); sns.distplot(rfm_scaled.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm_scaled.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm_scaled.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()


# ## K-means clustering
# 
# With the Elbow method, we can get the optimal number of clusters.

# In[82]:


# the Elbow method
from sklearn.cluster import KMeans
wcss = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters= k, init= 'k-means++', max_iter= 300)
    kmeans.fit(rfm_scaled)
    wcss[k] = kmeans.inertia_


# In[83]:


# plot the WCSS values
sns.pointplot(x = list(wcss.keys()), y = list(wcss.values()))
plt.xlabel('K Numbers')
plt.ylabel('WCSS')
plt.show()


# In[84]:


# clustering
clus = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 300,)# Lets select 3 clusters 
clus.fit(rfm_scaled)


# In[85]:


# Assign the clusters to datamart
rfm['K_Cluster'] = clus.labels_
rfm.head()


# In[86]:


# assign cluster column 
rfm_scaled['K_Cluster'] = clus.labels_
rfm_scaled['RFM_Level'] = rfm.RFM_Level
rfm_scaled.reset_index(inplace = True)


# In[87]:


rfm_scaled.head()


# In[88]:


# melt the dataframe
rfm_melted = pd.melt(frame= rfm_scaled, id_vars= ['CustomerKey', 'RFM_Level', 'K_Cluster'], var_name = 'Metrics', value_name = 'Value')
rfm_melted.head(10)


# In[89]:


# a snake plot with RFM
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'RFM_Level', data = rfm_melted)
plt.title('Snake Plot of RFM')
plt.legend(loc = 'upper right')

# a snake plot with K-Means
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'K_Cluster', data = rfm_melted)
plt.title('Snake Plot of RFM')
plt.legend(loc = 'upper right')


# In[90]:


# a snake plot with K-Means
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'K_Cluster', data = rfm_melted)
plt.title('Snake Plot of K_cluster')
plt.legend(loc = 'upper right')


# In[91]:


# the mean value for each cluster
cluster_avg = rfm.groupby('RFM_Level').mean().iloc[:, 0:3]
cluster_avg.head()


# In[92]:


# the mean value in total 
total_avg = rfm.iloc[:, 0:3].mean()
total_avg


# In[93]:


# the proportional mean value
prop_rfm = cluster_avg/total_avg - 1
prop_rfm


# In[94]:


# heatmap
sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True)
plt.title('Heatmap of RFM quantile')
plt.plot()


# In[95]:


# the mean value for each cluster
cluster_avg_K = rfm.groupby('K_Cluster').mean().iloc[:, 0:3]
cluster_avg_K.head()


# In[96]:


# the proportional mean value
prop_rfm_K = cluster_avg_K/total_avg - 1
prop_rfm_K


# In[97]:


# heatmap
sns.heatmap(prop_rfm_K, cmap= 'Blues', fmt= '.2f', annot = True)
plt.title('Heatmap of K-Means')
plt.plot()


# ## Supervised Technique
# 
# We will split the Non Churn dataset into two, train and Hold out sample, thereafter concatenate the Non Churnners and  Churners dataset to model. The holdout sample data will be used for scoring and deployment

# In[98]:


#Keep an hold out sample for scoring and deployment. 
#extract only churners
Churners = final_data.loc[final_data['Churn'] == 1] 
#Extract only first 6973 rows from Non churners
non_churners = final_data.loc[final_data['Churn'] == 0]
Score_data = final_data.loc[final_data['Churn'] == 0].tail(6000)


# In[99]:


#Counting Churners
Churners['Churn'].value_counts()


# In[100]:


#Counting non churners
non_churners['Churn'].value_counts()


# ### Lets randomly seperate the 12876 Non-Churnners into two groups:
#  >50% - Non-Churners are for training, validating, and testing the model.
#  
#  >50% - Score_data - for scoring the model which is going to be our hold-out sample 
# 
# We will use the train_test_split funtions to achieve random selection of the data, this is not for model purpose or data partitioning, its only to split the data and select a hold out sample.

# In[101]:


Score_data,Non_churners = train_test_split(non_churners,test_size=0.5,random_state=42)


# In[102]:


Non_churners['Churn'].value_counts()


# In[103]:


# This is my hold out sample for scoring/deployment
Score_data['Churn'].value_counts()


# In[104]:


Score_data.head(5)


# In[105]:


#Append Churners and non churners excluding the hold out sample for scoring
model_data = pd.concat([Churners, Non_churners],ignore_index=True)


# In[106]:


model_data['Churn'].value_counts()


# In[107]:


#Display the distribution of the target variable - 0 = Non Churner - 1 = Churner
sns.countplot(x='Churn', data = model_data, palette = 'hls')
plt.show()
plt.savefig('count_plot')


# ### Measures of central tendencies - investigating the difference between the averages of independent variables vs the target variables

# In[108]:


model_data.groupby('Churn').mean()


# In[109]:


#Most of the customers in this dataset are in the age range of 30–50.
plt.figure(figsize=(10,6))
model_data[model_data['Churn']==1]['Age_years'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Churn=1')
model_data[model_data['Churn']==0]['Age_years'].hist(alpha=0.2,color='green',
                                              bins=30,label='Churn=0')
plt.legend()
plt.xlabel('Age_years')
plt.ylabel('Frequency')


# In[110]:


#Investigating Churn versus Tenure Months.
plt.figure(figsize=(10,6))
model_data[model_data['Churn']==1]['Tenure_Months'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Churn=1')
model_data[model_data['Churn']==0]['Tenure_Months'].hist(alpha=0.2,color='green',
                                              bins=30,label='Churn=0')
plt.legend()
plt.xlabel('Tenure_Months')
plt.ylabel('Frequency')


# In[111]:


#No of children does not seem to be a good predictor of the outcome variable.
pd.crosstab(model_data.TotalChildren,model_data.Churn).plot(kind='bar')
plt.title('TotalChildren vs Churn')
plt.xlabel('TotalChildren')
plt.ylabel('Volume of Customers')
plt.savefig('pur_fre_pout_bar')


# In[112]:


#numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#newdf = model_data.select_dtypes(include=numerics)
x = model_data[['SalesAmount_sum', 'OrderQuantity_sum', 'YearlyIncome', 'TotalChildren', 'NumberChildrenAtHome', 'NumberCarsOwned', 'Tenure_Months', 'Age_years', 'Churn']]
sns.pairplot(x)


# In[113]:


#check the correlation of the input features with one another.
print(model_data.corr())


# In[114]:


#Identify missing values
model_data.isnull().mean().sort_values(ascending=False)*100


# In[115]:


#replace missing values with most frequent values
model_data = model_data.fillna(model_data.mode().iloc[0])


# In[116]:


#Carry out a statistical test to understand key statistical elements
model_data.describe()


# ## 4.0 Variable Selection - Vote based approach
# 
# We are going to use 6 means to select the best variable since we are not going to work with all the variables in our dataset. In selecting the variables we are going to using a vote based approach in which all the variables will be scored and the best score will be used.

# In[117]:


# Data transformation
# Convert categorical values to numeric using label encoder
from sklearn import preprocessing
from collections import defaultdict
d = defaultdict(preprocessing.LabelEncoder)

# Encoding the categorical variable
fit = model_data.select_dtypes(include=['object']).fillna('NA').apply(lambda x: d[x.name].fit_transform(x))

#Convert the categorical columns based on encoding
for i in list(d.keys()):
    model_data[i] = d[i].transform(model_data[i].fillna('NA'))


# In[118]:


model_data.head(5) #Display the new table


# In[119]:


#Select features and target thereby building our feature table
features = model_data[model_data.columns.difference(['Churn','CustomerKey','OrderDate_max','SalesAmount_max', 'SalesAmount_min','SalesAmount_mean', 'SalesAmount_median','OrderQuantity_min','OrderQuantity_max', 'OrderQuantity_mean', 'OrderQuantity_median','Cuttoff_date','Last_purchase_date','DateFirstPurchase','BirthDate'])]
labels = model_data['Churn']


# In[120]:


features.head()


# In[121]:


labels.head()


# # 4.1 Weight of Evidence (WOE) and Information Value (IV)

# In[122]:


# import packages
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin = 20
force_bin = 3

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv) 


# In[123]:


#Let get the independent variables 
final_iv, IV = data_vars(model_data[model_data.columns.difference(['Churn','CustomerKey','OrderDate_max','SalesAmount_max', 'SalesAmount_min','SalesAmount_mean', 'SalesAmount_median','OrderQuantity_min','OrderQuantity_max', 'OrderQuantity_mean', 'OrderQuantity_median','Cuttoff_date','Last_purchase_date','DateFirstPurchase','BirthDate'])],model_data.Churn)


# ### We want to know the relationship between an independent variables and a dependent variable (“Churn”).

# In[124]:


IV = IV.rename(columns={'VAR_NAME':'index'})
IV.sort_values(['IV'],ascending=0)


# # 4.2 Variable Importance using Random Forest

# In[125]:


#Display our label table 
labels.head(10)


# In[126]:


#import library
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(features,labels) #Fit features and labels

preds = clf.predict(features) #Predict the classification score of features

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(preds,labels)#Get the classification accuracy level/score
print(accuracy)


# >From the above code With 99% accuracy, the classfication score is very accurate.

# In[127]:


#import library
from pandas import DataFrame
VI = DataFrame(clf.feature_importances_, columns = ["RF"], index=features.columns) # RF stands for Random Forest


# In[128]:


VI = VI.reset_index()
VI.sort_values(['RF'],ascending=0)


# # 4.3 Recursive Feature Elimination

# In[129]:


# Get features that are very important 
#import library
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 20)
fit = rfe.fit(features, labels)


# In[130]:


#Import Library
from pandas import DataFrame
Selected = DataFrame(rfe.support_, columns = ["RFE"], index=features.columns)# RFE stands for Recursive Feature Elimination
Selected = Selected.reset_index()


# In[131]:


Selected[Selected['RFE'] == True] #Display list of very important features 


# # 4.4 Variable Importance using Extratrees Classifier
# 
# Extratrees classifier perform similarly to RandomForest, the only major difference is the level of variance, ofcourse extratrees show the lowest variance.
# 
# Let us start by importing the library

# In[132]:


from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(features, labels)

print(model.feature_importances_)


# In[133]:


from pandas import DataFrame #import pandas from DATAfRAME 
FI = DataFrame(model.feature_importances_, columns = ["Extratrees"], index=features.columns)


# In[134]:


FI = FI.reset_index()


# In[135]:


FI.sort_values(['Extratrees'],ascending=0)


# # 4.5 Chi Square
# 
# We will use chi square to determine which of the features are very relevant to the outcome that will be predicted and remove those that are unimportant for this model

# In[136]:


#Import libraries 
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2

model = SelectKBest(score_func=chi2, k=5) #Let select the best the features and expected frequancy of 5
fit = model.fit(features.abs(), labels) 


# In[137]:


#Let create a contegancy table from the dataframe table, but first lets import the library from pandas
from pandas import DataFrame
pd.options.display.float_format = '{:.2f}'.format
chi_sq = DataFrame(fit.scores_, columns = ["Chi_Square"], index=features.columns)


# In[138]:


chi_sq = chi_sq.reset_index()


# In[139]:


#display the created table 
chi_sq.sort_values('Chi_Square',ascending=0)


# # 4.6 L1 feature selection
# 
# We are going to use regularisation to add penalty in every parameters in the machine learning model, the reason for this is to eliminate overfitting in the model.

# In[140]:


#Import the libraries
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels) #let set the panelty 
model = SelectFromModel(lsvc,prefit=True)


# In[141]:


#Let test the effectness of various features
from pandas import DataFrame
l1 = DataFrame(model.get_support(), columns = ["L1"], index=features.columns)
l1 = l1.reset_index()


# In[142]:


#display the performance of all the feature
l1[l1['L1'] == True]


# ### From the above code we could get only 5 features that are selected

# # 5 Combine all together
# 
# let combine all the six models  we have built above, the reason is for effective scoring and easy comparism amonge them using all the features.

# In[143]:


from functools import reduce # import library
dfs = [IV, VI, Selected, FI, chi_sq, l1] #Create a new table for the models built
final_results = reduce(lambda left,right: pd.merge(left,right,on='index'), dfs) #Let get our final table 


# # Vote each variable
# 
# From the code below we will score all the variables using all the models we have built above. But before we can do that, we need to create columns for all the models and bring them into score_table. Meanwhile, it is very important for us to convert some of the models like RFE and L1 that came out as string to integer for easy scoring.

# In[144]:


columns = ['IV', 'RF', 'Extratrees', 'Chi_Square'] #let create a column for all the models built above that are integer 

score_table = pd.DataFrame({},[]) #Create the scoring table fom pandas 
score_table['index'] = final_results['index'] # create Index column that has all the variables

for i in columns:
    score_table[i] = final_results['index'].isin(list(final_results.nlargest(5,i)['index'])).astype(int)
    
score_table['RFE'] = final_results['RFE'].astype(int) # converting and creating a column for RFE from string to integer
score_table['L1'] = final_results['L1'].astype(int) # Converting and creating a column for L1 from string to integer


# In[145]:


score_table['final_score'] = score_table.sum(axis=1)#Sum the scoring from each row into final score column


# In[146]:


score_table.sort_values('final_score',ascending=0)# Display the table and sort it by final_score in ascending order


# From the above we have suceesfully scored and rank all the variables based on the final_score. OrderQuantity_sum has the highest score among all the varaibles, all the models above scored it only Chi_sq that didnt. This is great because our concern is to get the best model with the most important variables for our Label (Churn)

# # Multicollinearity
# 
# Let us get the level of inter-relationship among all the independent variables, if is high is an issue for our model because the outcome wont be correct. Be as it may, it will increase our confidence in our models and variables.

# In[147]:


#import the library
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[148]:


def calculate_vif(features): #Let def the function
    vif = pd.DataFrame() #Build a new a dataframe called varieance inflation factor 
    vif["Features"] = features.columns #Create a column for Features 
    vif["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])] #Add column for VIF   
    return(vif) # Return the vif


# In[149]:


#Generate a feature table for variables with final score equal or greater than 2
features = features[list(score_table[score_table['final_score'] >= 2]['index'])]


# In[150]:


vif = calculate_vif(features)
while vif['VIF'][vif['VIF'] > 10].any(): #Get VIF greater than 10 
    remove = vif.sort_values('VIF',ascending=0)['Features'][:1] #Remove when the VIF is greater than 10
    features.drop(remove,axis=1,inplace=True) #Drop all variables that fall under the category 
    vif = calculate_vif(features)


# In[151]:


#List all the independent variables in VIF.
list(vif['Features'])


# In[152]:


#Describe the new table 
features.describe()


# In[153]:


#Display the first rows in the new table
features.head(5)


# From all our coding activities above we have identified and removed all the possible variables that have high inter-relationship which may likly lead to wrong outcome. Now we are left with 12 variables in our feature tables.

# # Descriptive Stats and Visualization - For selected variables
# 
# We have gotten our truested and reliable independable variables (Features), lets merge this table with our dependable variable Churn. Thereafter we do some cleaning prior visualizing of the table.

# In[154]:


#Merge the best list to churn variable
final_vars = list(vif['Features']) + ['Churn']


# In[155]:


#fill columns with NAN with and integer 0
df1 = model_data[final_vars].fillna(0)


# In[156]:


#Discribe the dataframe to see if any abnormality still exist before visualization
df1.describe()


# In[157]:


bar_color = '#058caa'
num_color = '#ed8549'

final_iv,_ = data_vars(df1,df1['Churn'])
final_iv = final_iv[(final_iv.VAR_NAME != 'Churn')]
grouped = final_iv.groupby(['VAR_NAME'])
for key, group in grouped:
    ax = group.plot('MIN_VALUE','EVENT_RATE',kind='bar',color=bar_color,linewidth=1.0,edgecolor=['black'])
    ax.set_title(str(key) + " vs " + str('target'))
    ax.set_xlabel(key)
    ax.set_ylabel(str('target') + " %")
    rects = ax.patches
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.01*height, str(round(height*100,1)) + '%', 
                ha='center', va='bottom', color=num_color, fontweight='bold')


# ## Cross Validation
# 
# We have successfully clean and selected our vriables, meaning we have Features, Labels and Hold Out sample data ready. it is time to validate all the various to enable us make the right choice of model using Cross validation

# In[158]:


from sklearn.model_selection import train_test_split # Import the library
#Since we are using cross validation here, we will allow almost 100% of the data in train
train, test = train_test_split(df1, test_size=0.001, random_state=0) 
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

features_train = train[list(vif['Features'])]
label_train = train['Churn']
features_test = test[list(vif['Features'])]
label_test = test['Churn']


# In[159]:


#implement cross validation, the cross_val_score method of the sklearn.model_selection library can be used. 
from sklearn.ensemble import RandomForestClassifier  
classifier = RandomForestClassifier(n_estimators=300, random_state=0) #The number of trees in the randomforest is 300


# In[160]:


from sklearn.model_selection import cross_val_score #import the library 
#Lets get accurancies for 5 folds in the cross validation
all_accuracies = cross_val_score(estimator=classifier, X=features_train, y=label_train, cv=5) 


# In[161]:


print(all_accuracies) 


# In[162]:


#find the average of all the accuracies, simple use the mean() method of the object returned by cross_val_score
print(all_accuracies.mean()) 


# In[163]:


#Finally let's find the standard deviation of the data to see degree of variance in the results obtained by our model. To do so, call the std() method on the all_accuracies object
print(all_accuracies.std()) 


# ##  Cross Validation using Logistic Regression 

# In[164]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()


# In[165]:


all_accuracies = cross_val_score(estimator=clf, X=features_train, y=label_train, cv=5)  


# In[166]:


print(all_accuracies) 


# In[167]:


#find the average of all the accuracies, simple use the mean() method of the object returned by cross_val_score
print(all_accuracies.mean()) 


# In[168]:


#Finally let's find the standard deviation of the data to see degree of variance in the results obtained by our model. To do so, call the std() method on the all_accuracies object
print(all_accuracies.std()) 


# ## Grid Search for Parameter Selection
# instead of randomly selecting the values of the parameters, a better approach would be to develop an algorithm which automatically finds the best parameters for a particular model. Grid Search is one such algorithm.

# In[169]:


#Let create the grid_param dictionary with these parameters n_estimators, criterion and bootstrap
grid_param ={
    'n_estimators': [50, 100, 150],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}


# In[170]:


import sklearn #import sklearn
print(sklearn.__version__) #Get the vision of sklearn


# In[171]:


#Import GridSearchCV from SKlearn
from sklearn.model_selection import learning_curve,GridSearchCV
gd_sr = GridSearchCV(estimator=classifier,  
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=10, 
                     n_jobs=-1) #We are going to use all the computing power of the CPU which stands for -1


# In[172]:


#Once the GridSearchCV class is initialized, the last step is to call the fit method of the class and pass it the training and 
#test set, as shown in the following code:
gd_sr.fit(features_train, label_train)


# In[173]:


#the next step is to check the parameters that return the highest accuracy
best_parameters = gd_sr.best_params_  
print(best_parameters)


# The result shows that the highest accuracy is achieved when the n_estimators are 100, bootstrap is True and criterion is "gini".
# 
# The last and final step of Grid Search algorithm is to find the accuracy obtained using the best parameters. Previously we had a mean accuracy of 66.72% with 300 n_estimators.

# In[174]:


#To find the best accuracy achieved, execute the following code:
best_result = gd_sr.best_score_  
print(best_result)


# The accuracy achieved is: 0.6591 of 65.91% which is only slightly below 69.72%.

# ## Lets check the performance of the models and select the best for the project 

# In[175]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(df1, test_size = 0.4)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

features_train = train[list(vif['Features'])]
label_train = train['Churn']
features_test = test[list(vif['Features'])]
label_test = test['Churn']


# In[176]:


all_features = pd.concat([features_train, features_test])
all_labels = pd.concat([label_train, label_test])


# ## RandomForestClassifier

# In[177]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(features_train,label_train)#Fit in RandomForestClassifier into the features and label data
#Predict the performance of the model
pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,label_train)
accuracy_test = accuracy_score(pred_test,label_test)
#lets plot the AUC and ROC to get the performance of the model
from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
auc_train = metrics.auc(fpr,tpr)

fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
auc_test = metrics.auc(fpr,tpr)

print(accuracy_train,accuracy_test,auc_train,auc_test)#print the out come 


# From the code above we can see that the train accuracy is 99.18%, and AUC is 99.98% which makes it great because the highier the AUC the better for the model

# In[178]:


#Train result using RandomForestClassifier
pd.crosstab(label_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED'])


# In[179]:


pd.crosstab(label_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED'])


# ## Logistic Regression

# In[180]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(features_train,label_train)

pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,label_train)
accuracy_test = accuracy_score(pred_test,label_test)

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
auc_train = metrics.auc(fpr,tpr)

fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
auc_test = metrics.auc(fpr,tpr)

print(accuracy_train,accuracy_test,auc_train,auc_test)


# In[181]:


pd.crosstab(label_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED'])


# In[182]:


pd.crosstab(label_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED'])


# ## Neural Network

# In[183]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()

clf.fit(features_train,label_train)

pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,label_train)
accuracy_test = accuracy_score(pred_test,label_test)

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
auc_train = metrics.auc(fpr,tpr)

fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
auc_test = metrics.auc(fpr,tpr)

print(accuracy_train,accuracy_test,auc_train,auc_test)


# In[184]:


pd.crosstab(label_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED'])


# In[185]:


pd.crosstab(label_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED'])


# ## Naive Bayes

# In[186]:


from sklearn.naive_bayes import GaussianNB 
clf = GaussianNB()

clf.fit(features_train,label_train)

pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,label_train)
accuracy_test = accuracy_score(pred_test,label_test)

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
auc_train = metrics.auc(fpr,tpr)

fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
auc_test = metrics.auc(fpr,tpr)

print(accuracy_train,accuracy_test,auc_train,auc_test)


# In[187]:


pd.crosstab(label_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED'])


# In[188]:


pd.crosstab(label_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED'])


# ## Gradient Boosting

# In[189]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()

clf.fit(features_train,label_train)

pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,label_train)
accuracy_test = accuracy_score(pred_test,label_test)

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
auc_train = metrics.auc(fpr,tpr)

fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
auc_test = metrics.auc(fpr,tpr)

print(accuracy_train,accuracy_test,auc_train,auc_test)


# In[190]:


pd.crosstab(label_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED'])


# In[191]:


pd.crosstab(label_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED'])


# ## Performance Tuning
# 
# Our best model is RandomForestClassifier with an accurancy of 99.18% and AUC 0f 99.98%. Now lets check the performance of the best to model.

# In[192]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(3, 10, num = 1)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 2, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(features_train, label_train)

print(rf_random.best_params_)


# In[193]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(**rf_random.best_params_)

clf.fit(features_train,label_train)

pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,label_train)
accuracy_test = accuracy_score(pred_test,label_test)

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
auc_train = metrics.auc(fpr,tpr)

fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
auc_test = metrics.auc(fpr,tpr)

print(accuracy_train,accuracy_test,auc_train,auc_test)


# In[194]:


metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])


# ## Final Model and Model Performance
# 
# We have gone round all the model and have made our choice based on the performance of every model, we have ended up with RandomForestClassifier because it has accurancy of 99.18% and AUC of 99.98% which is the best among all the report from other models. Now we will apply this to all the dataset.

# In[195]:


#Train the final model on all the dataset
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

clf.fit(all_features,all_labels)

pred_train = clf.predict(all_features)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,all_labels)


from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(np.array(all_labels), clf.predict_proba(all_features)[:,1])
auc_train = metrics.auc(fpr,tpr)


print(accuracy_train,auc_train)


# In[196]:


pd.crosstab(all_labels,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED'])


# In[197]:


pd.crosstab(label_test,pd.Series(pred_test),rownames=['ACTUAL'],colnames=['PRED'])


# In[198]:


import bokeh
#from bokeh.charts import Histogram
from ipywidgets import interact
from bokeh.plotting import figure
from bokeh.io import push_notebook, show, output_notebook
output_notebook()

from sklearn import metrics
preds = clf.predict_proba(features_train)[:,1]

fpr, tpr, _ = metrics.roc_curve(np.array(label_train), preds)
auc = metrics.auc(fpr,tpr)

p = figure(title="ROC Curve - Train data")
r = p.line(fpr,tpr,color='#0077bc',legend = 'AUC = '+ str(round(auc,3)), line_width=2)
s = p.line([0,1],[0,1], color= '#d15555',line_dash='dotdash',line_width=2)
show(p)


# In[199]:


#from bokeh.charts import Histogram
from ipywidgets import interact
from bokeh.plotting import figure
from bokeh.io import push_notebook, show, output_notebook
output_notebook()

from sklearn import metrics
preds = clf.predict_proba(features_test)[:,1]

fpr, tpr, _ = metrics.roc_curve(np.array(label_test), preds)
auc = metrics.auc(fpr,tpr)

p = figure(title="ROC Curve - Test data")
r = p.line(fpr,tpr,color='#0077bc',legend = 'AUC = '+ str(round(auc,3)), line_width=2)
s = p.line([0,1],[0,1], color= '#d15555',line_dash='dotdash',line_width=2)
show(p)


# ## Save Model for future use

# In[200]:


import pandas
import joblib

filename = 'final_model.model'
i = [d,clf]
joblib.dump(i,filename)


# ## Score New Data

# In[201]:


all_features.dtypes


# In[202]:


#Prepare score data to ensure the attributes of the modelling data matches that of the scoring data
#Score_data

#replace missing values with most frequent values
Score_data_sel = Score_data.fillna(Score_data.mode().iloc[0])

#Select features and target
Score_data_sel = Score_data_sel[['Age_years',
'CommuteDistance_0-1 Miles',
'EnglishEducation_Bachelors',
'MaritalStatus_S',
'NumberChildrenAtHome',
'OrderQuantity_sum',
'Quality',
'SalesAmount_sum',
'Television  Advertisement',
'YearlyIncome',
'diff_median']]


# In[203]:


#Select features and target
Score_data_sel.head(5)


# In[204]:


# Use the code to load the model
filename = 'final_model.model'

import joblib
d,clf=joblib.load(filename)


# In[205]:


#Score the final model (Logistic regression) 
pred_train = clf.predict(Score_data_sel)


# In[206]:


def score_new(features,clf):
    score = pd.DataFrame(clf.predict_proba(features)[:,1], columns = ['SCORE'])
    score['DECILE'] = pd.qcut(score['SCORE'].rank(method = 'first'),10,labels=range(10,0,-1))
    score['DECILE'] = score['DECILE'].astype(float)
    return(score)


# In[207]:


scores = score_new(Score_data_sel,clf)


# In[208]:


# How many 0s and 1s are there

print('There are', np.count_nonzero(pred_train == 0), '0s')

print('There are', np.count_nonzero(pred_train == 1), '1s')


# In[209]:


Score_data.head(5)


# In[210]:


# Put the result in a datafram
#, 'Score':scores.SCORE, 'Decile':scores.DECILE
Final = pd.DataFrame({'CustomerKey':Score_data.CustomerKey,'Churn_Flag':pred_train})


# In[211]:


# Merge the Final data to the custtbl to do some demographics profiling
Final_ouput = pd.merge(custtbl,
                 Final,
                 on='CustomerKey',how = 'inner')
Final_ouput.head()


# In[212]:


Final_ouput.count()


# In[213]:


prob = scores[['SCORE']].values.squeeze()


# In[214]:


#, 'Score':scores.SCORE, 'Decile':scores.DECILE
Final_ = pd.DataFrame({'CustomerKey':Score_data.CustomerKey,'Churn_Probability':prob})


# In[215]:


Final_.sort_values(['Churn_Probability']).head()


# In[216]:


# Merge the probabilities to the
Final = pd.merge(Final_ouput,
                 Final_,
                 on='CustomerKey',how = 'inner')
Final.sort_values(['Churn_Probability']).head(50)


# In[217]:


# Store the result in a csv file
Final.to_csv('C:\\Users\\Hp\Predicted Churners.csv',index=False)


# In[220]:


labels = 'Churned', 'Retained'
sizes = [Final.Churn_Flag[Final['Churn_Flag']==1].count(), Final.Churn_Flag[Final['Churn_Flag']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proposition of Possible Churners and Retain", size = 20)
plt.show()


# In[ ]:




