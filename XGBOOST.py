#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import GridSearchCV


# In[2]:


# Load the preprocessed dataset
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/preprocessed_crimes_data.csv')


# In[3]:


data.isnull()


# In[4]:


# Separate features and target variable
X = data.drop('Primary Type', axis=1)
y = data['Primary Type']


# In[5]:


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Verify the unique classes in the new sets
unique_classes_train = set(y_train)
unique_classes_val = set(y_val)

print(f'Unique classes in train set: {unique_classes_train}')
print(f'Unique classes in validation set: {unique_classes_val}')


# In[6]:


# Initialize XGBoost classifier
model = XGBClassifier(
    objective='multi:softmax',
    eval_metric='mlogloss',
    n_estimators=75,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1
)


# In[7]:


#Train the model
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)


# In[8]:


# Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')


# In[9]:


# Evaluate the model on the test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# ROC 

# In[10]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)

# Choose the number of classes to display ROC curves for
n_classes = 10

plt.figure(figsize=(8, 8))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (class {i}) - AUC = {roc_auc:.2f}')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for the first {n_classes} classes')
plt.legend(loc='best')
plt.show()



# In[11]:


#test confusion matrix

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Set background style for the plot
sns.set_style("dark")  # Change the style here

# Plot confusion matrix as a heatmap
plt.figure(figsize=(14, 14))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_classes_train)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=100)  # Rotate x labels for better readability
plt.yticks(rotation=0)   # Keep y labels horizontal for better readability
plt.tight_layout()
plt.title('Confusion Matrix for Original Data\'s Prediction')
plt.show()


# In[12]:


#validation confusion matrix


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)

# Set background style for the plot
sns.set_style("dark")  # Change the style here

# Plot confusion matrix as a heatmap
plt.figure(figsize=(14, 14))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=unique_classes_train)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=100)  # Rotate x labels for better readability
plt.yticks(rotation=0)   # Keep y labels horizontal for better readability
plt.tight_layout()
plt.title('Confusion Matrix for Original Data\'s Prediction')
plt.show()


# In[13]:


#validation data classification report

from sklearn.metrics import classification_report

# Replace X_val or X_test with the respective validation or test set
y_pred = model.predict(X_val)  # Or X_test for test set

# Generate classification report
class_report = classification_report(y_val, y_pred)
print("Classification Report:")
print(class_report)


# In[14]:


#test data classification report

from sklearn.metrics import classification_report

# Replace X_val or X_test with the respective validation or test set
y_pred = model.predict(X_test)  # Or X_test for test set

# Generate classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# Hypertuning the model
# 

# In[15]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 125],  # Fixed value for number of estimators
    'learning_rate': [0.01] ,  # Fixed value for learning rate
}

# Initialize XGBoost classifier
model = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', n_jobs=-1)

# Initialize StratifiedKFold
stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV with StratifiedKFold
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='accuracy', cv=stratified_cv)


# In[16]:


# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Display the best parameters and corresponding accuracy
print(f'Best Parameters: {grid_search.best_params_}')

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions on the validation set with the best model
y_val_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy with Best Model: {accuracy * 100:.2f}%')

# Make predictions on the test set with the best model
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy with Best Model: {test_accuracy * 100:.2f}%')

# Display the classification report for the test set
report = classification_report(y_test, y_test_pred)
print(f'{report}')


# In[18]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = best_model.predict_proba(X_test)

# Choose the number of classes to display ROC curves for
n_classes = 10

plt.figure(figsize=(8, 8))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (class {i}) - AUC = {roc_auc:.2f}')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve after hypertuning for the first {n_classes} classes')
plt.legend(loc='best')
plt.show()


# In[20]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming best_model is your XGBoost model obtained after GridSearchCV and X_test, y_test are your test data
#y_test_pred = best_model.predict(X_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Display the confusion matrix as a heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Test Data')
plt.show()

