#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import re
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ## Data Collection

# ## 3.2 
# ### Define the data sources, parameters and quantity of raw datasets (0.4 point) 

# In[2]:


# Loading the dataset
data = pd.read_csv('crime_dataset.csv')


# In[3]:


# Displaying information about the Dataset
print(data.info())


# ### Collect necessary and sufficient raw datasets; Show samples from raw datasets. (0.1 point)

# In[4]:


data.head(5)


# In[5]:


data.sample(5)


# ### EDA before pre-processing the data

# ### 1. Crime type frequencies and Top 10 Crime types

# In[6]:


# Import necessary libraries for EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for plots
sns.set(style="whitegrid")

# Visualize the distribution of the target variable "Primary Type"
plt.figure(figsize=(12, 6))
sns.countplot(x='Primary Type', data=data, order=data['Primary Type'].value_counts().index)
plt.title('Distribution of Primary Type')
plt.xticks(rotation=90)
plt.show()

# Print the top 10 Primary Types
top_primary_types = data['Primary Type'].value_counts().nlargest(10).index
print("Top 10 Primary Types:")
print(top_primary_types)


# ## Data Pre-processing

# ## 3.3

# ### Pre-process collected raw data with cleaning and validation tools;  (0.4 point) 

# #### 1. Checking the Missing values

# In[7]:


# Checking for missing values
print("Missing values:\n", data.isnull().sum())


# #### 2. Handling missing values

# ###  do count plot if possible

# #### Imputing the missing values using mean and mode for numerical and categorical columns respectively

# In[8]:


num_cols = data.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='mean')
data[num_cols] = imputer.fit_transform(data[num_cols])

cat_cols = data.select_dtypes(include='object').columns
imputer = SimpleImputer(strategy='most_frequent')
data[cat_cols] = imputer.fit_transform(data[cat_cols])


# In[9]:


# Checking for missing values after imputing
print("Missing values:\n", data.isnull().sum())


# #### 3.Checking for Duplicate values

# In[10]:


# Checking for duplicate rows
print("Duplicate rows:", data.duplicated().sum())

# Dropping duplicate rows (if necessary)
data = data.drop_duplicates()


# #### 4. Printing Basic Statistics

# In[11]:


# Displaying basic statistics of the numerical columns
print("Basic Statistics:\n", data.describe())


# #### 5.Printing unique values in categorical columns

# In[12]:


# Displaying unique values in categorical columns
for column in data.select_dtypes(include=['object']).columns:
    print(f"Unique values in {column}:", data[column].unique())


# #### Some sample of data

# In[13]:


# Displaying samples from the pre-processed dataset
print("Samples from the pre-processed dataset:\n", data.sample(5))


# ### Performing EDA  after pre-processing to analyze the data more

# ### 1. Crime type frequencies and Top 10 Crime types

# In[14]:


# Import necessary libraries for EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for plots
sns.set(style="whitegrid")

# Visualize the distribution of the target variable "Primary Type"
plt.figure(figsize=(12, 6))
sns.countplot(x='Primary Type', data=data, order=data['Primary Type'].value_counts().index)
plt.title('Distribution of Primary Type')
plt.xticks(rotation=90)
plt.show()

# Print the top 10 Primary Types
top_primary_types = data['Primary Type'].value_counts().nlargest(10).index
print("Top 10 Primary Types:")
print(top_primary_types)


# ### 2. Top 10 Community areas where the crime is majorly happening

# In[15]:


import matplotlib.pyplot as plt

# Select the top N community areas with the highest crime frequency
top_community_areas = data['Community Area'].value_counts().nlargest(10).index
data_top_community_areas = data[data['Community Area'].isin(top_community_areas)]

# Grouping data by 'Primary Type' and 'Community Area' and counting the frequency
crime_counts_by_community = data_top_community_areas.groupby(['Community Area', 'Primary Type']).size().unstack(fill_value=0)

# Plotting the bar chart
plt.figure(figsize=(15, 8))  # Provide both width and height values
crime_counts_by_community.plot(kind='bar', stacked=True, colormap='viridis')

# Setting x and y labels and title for the plot
plt.xlabel('Community Area')
plt.ylabel('Frequency')
plt.title('Frequency of Top Crime Types by Top 10 Community Areas')

# Customizing the legend
plt.legend(title='Primary Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ### 3.Top 10 Crime types and top 10 location descriptions

# In[16]:


# Select the top N primary crime types and top N location descriptions
top_primary_types = data['Primary Type'].value_counts().nlargest(10).index
top_location_descriptions = data['Location Description'].value_counts().nlargest(10).index

data_top_primary_types_locations = data[data['Primary Type'].isin(top_primary_types) & data['Location Description'].isin(top_location_descriptions)]

# Grouping data by 'Location Description' and 'Primary Type' and counting the frequency
crime_counts_by_location = data_top_primary_types_locations.groupby(['Location Description', 'Primary Type']).size().unstack(fill_value=0)

# Plotting the bar chart
plt.figure(figsize=(15, 15))
crime_counts_by_location.plot(kind='bar', stacked=True, colormap='viridis')

# Setting x and y labels and title for the plot
plt.xlabel('Location Description')
plt.ylabel('Frequency')
plt.title('Frequency of Top 10 Primary Crime Types within Top 10 Location Descriptions')

# Customizing the legend
plt.legend(title='Primary Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# ## Data Transformation

# ## 3.4

# ### Transform pre-processed datasets to desired formats , show the related tools , scripts or formulas, methods;  (0.4 point)

# In[17]:


# Label encoding for categorical variables
label_encoder = LabelEncoder()
for col in cat_cols:
    data[col] = label_encoder.fit_transform(data[col])


# In[18]:


data.head(5)


# ### EDA after Label Encoding

# In[19]:


# Visualize the correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# ## Feature Selection and Engineering

# ### Using Lasso Regression (L1 Regularization) and Domain knowledge for feature selection

# In[20]:


from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

X = data.drop(['Primary Type'], axis=1)

# Target variable
y = data['Primary Type']

# Assuming 'X' and 'y' are your feature matrix and target variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply L1 regularization with cross-validated selection of the best alpha
lasso = LassoCV(cv=5)
lasso.fit(X_scaled, y)

# Get selected features with non-zero coefficients
selected_features = X.columns[lasso.coef_ != 0]

print("Selected Features:", selected_features)


# In[21]:


# Loading the selected features based on Lasso Regression (L1 Regularization) and Domain knowledge into a new CSV file
selected_features = [
    'Date','Year','Longitude','Latitude','Location Description','Primary Type','Description'
]

data_selected = data[selected_features]


# In[22]:


# Save the new CSV file as preprocessed_crimes_data.csv
data_selected.to_csv('preprocessed_crimes_data.csv', index=False)


# In[23]:


# Load the preprocessed_crimes_data.csv file
data_selected = pd.read_csv('preprocessed_crimes_data.csv')


# In[24]:


data_selected.head(10)


# In[25]:


# Renaming the 'Primary Type' column to 'Crime_Type'
data_selected.rename(columns={'Primary Type': 'Crime_Type'}, inplace=True)


# In[26]:


data_selected.head(10)


# In[27]:


# Checking for missing values after feature engineering
print("Missing values:\n", data_selected.isnull().sum())


# ## Data Preparation

# ### 3.5

# #### Preparing Training, Validation and testing datasets from Transformed dataset (preprocessed_crimes_data.csv)

# In[ ]:




