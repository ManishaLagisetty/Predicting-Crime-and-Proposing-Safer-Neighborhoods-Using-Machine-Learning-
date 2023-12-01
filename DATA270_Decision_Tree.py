#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install joblib


# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import time

file_path = 'preprocessed_crimes_data.csv'
df = pd.read_csv(file_path)
df.rename(columns={'Primary Type': 'Crime_Type'}, inplace=True)

print("Missing values:\n", df.isnull().sum())

#columns_to_convert = ['Year', 'Location Description', 'Crime_Type', 'Description']
#df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
#df[columns_to_convert] = df[columns_to_convert].fillna(0).astype(int)

df.head(5)


# ## Common functions for metrics

# In[9]:


# precision, recall, and F1 score
def print_metrics(y_test, y_test_pred):
    precision = precision_score(y_test, y_test_pred, average='weighted')  # Use 'weighted' for multiclass
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')


# In[10]:


def print_class_report_conf_matrix(y_test, y_test_pred):
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    #np.set_printoptions(threshold=np.inf)
    print('Confusion Matrix:')
    print(conf_matrix)
    #np.set_printoptions(threshold=1000)

    # Classification Report
    class_report = classification_report(y_test, y_test_pred)
    print('Classification Report:')
    print(class_report)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Accuracy Score: {accuracy * 100:.2f}%')
    print_metrics(y_test, y_test_pred)


# In[11]:


def show_roc_auc_curve(X_test, y_test, model):
    y_test_bin = label_binarize(y_test, classes=model.classes_)
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model.predict_proba(X_test)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
    plt.show()


# In[12]:


df.info()


# ## Splitting into training, testing and validation datasets

# In[13]:


target_col = 'Crime_Type'
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split the data into training, testing, and validation sets
# 70% training, 20% testing, 10% validation
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

print(len(X_train), len(y_train), len(X_test), len(y_test), len(X_val), len(y_val))


# ## Baseline Model

# In[14]:


start_time = time.time()
classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Now make predictions on the test set
y_test_pred = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")


# In[15]:


print_metrics(y_test, y_test_pred)


# In[16]:


print_class_report_conf_matrix(y_test, y_test_pred)


# In[17]:


show_roc_auc_curve(X_test, y_test, classifier)


# In[5]:


# Save the trained model
model_filename = 'dt_model_baseline.joblib'
joblib.dump(classifier, model_filename)
print(f'Model saved to {model_filename}')


# ## Do GridSearch to find best hyperparameters for decision tree classifier

# In[10]:


dtree = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Use GridSearchCV to do grid search
grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_

accuracy = best_model.score(X_test, y_test)
print("Accuracy on Test Set:", accuracy)


# ## Hyperparameter tuned model

# In[18]:


start_time = time.time()
best_clssifier = DecisionTreeClassifier(criterion='entropy',
                                        max_depth=15,
                                        max_features=None,
                                        min_samples_leaf=4,
                                        min_samples_split=5)

best_clssifier.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = best_clssifier.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Now make predictions on the test set
y_test_pred = best_clssifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")


# In[19]:


print_metrics(y_test, y_test_pred)


# In[20]:


print_class_report_conf_matrix(y_test, y_test_pred)


# In[21]:


show_roc_auc_curve(X_test, y_test, best_clssifier)


# In[9]:


# Save the trained model
model_filename = 'dt_hyperparameter_tuning_model.joblib'
joblib.dump(best_clssifier, model_filename)
print(f'Model saved to {model_filename}')


# In[ ]:




