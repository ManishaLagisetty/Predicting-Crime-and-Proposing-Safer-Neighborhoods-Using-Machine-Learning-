#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
data_selected = pd.read_csv('/Users/sarahsha/Downloads/preprocessed_crimes_data.csv')


# In[2]:


data_selected.head(5)


# In[3]:


# Renaming the 'Primary Type' column to 'Crime_Type'
data_selected.rename(columns={'Primary Type': 'Crime_Type'},inplace = True)


# In[4]:


#non_criminal_group=[20,21,22]
#narcotics = [19,25]
#public_peace_violation = [28,29]
#other_offense = [26,30]
#data_selected['Crime_Type'] = data_selected['Crime_Type'].replace(non_criminal_group, 20)
#data_selected['Crime_Type'] = data_selected['Crime_Type'].replace(narcotics,19)
#data_selected['Crime_Type'] = data_selected['Crime_Type'].replace(public_peace_violation,29)
#data_selected['Crime_Type'] = data_selected['Crime_Type'].replace(other_offense,26)
#drop the outliers domestic violence
#data_new = data_selected[data_selected['Crime_Type'] != 10]


# In[5]:


data_selected


# In[6]:


X = data_selected.drop('Crime_Type', axis=1)
y = data_selected['Crime_Type']


# In[7]:


# scale the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[8]:


from sklearn.model_selection import train_test_split
# Split the data into training and temporary (validation and testing) sets.
# X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# Split the remaining data into validation and testing sets.
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)


# In[9]:


from tensorflow.keras.utils import to_categorical
num_classes=len(y_train.unique())
y_train_onehot = to_categorical(y_train, num_classes)
y_val_onehot = to_categorical(y_val, num_classes)
y_test_onehot = to_categorical(y_test, num_classes)


# In[10]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[11]:


model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(num_classes, activation='softmax'))
#from tensorflow.keras.optimizers import Adam
# Compile the model
#optimizer = Adam(learning_rate=10) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train_onehot, epochs=5, batch_size=32, validation_data=(X_val,y_val_onehot))

# Evaluate the model on the test set


# In[12]:


# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test_onehot)


# In[16]:


# Predict the probabilities of instances belonging to each class.
y_pred_probabilities = model.predict(X_test)
# Make the final Prediction
y_pred = np.argmax(y_pred_probabilities,axis=-1)


# In[17]:


# Evaluate the performance
from sklearn.metrics import precision_score, recall_score, f1_score
Precision = precision_score(y_test, y_pred, average='weighted')
Recall = recall_score(y_test, y_pred, average='weighted')
F1_score = f1_score(y_test, y_pred, average='weighted')
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'One Hidden Layer ANN Precision: {Precision:.2f}' )
print(f'One Hidden Layer ANN Recall: {Recall:.2f}' )
print(f'One Hidden Layer ANN F1_score: {F1_score:.2f}' )


# In[13]:


training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']


# In[19]:


y_pred_probabilities = model.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score, classification_report
y_pred = np.argmax(y_pred_probabilities,axis=-1)
report = classification_report(y_test, y_pred)
print(f'One Hidden Layer Classification Report:\n'
      f'{report}')


# In[21]:


from sklearn.metrics import precision_score, recall_score, f1_score

Precision = precision_score(y_test, y_pred, average='weighted')
Recall = recall_score(y_test, y_pred, average='weighted')
F1_score = f1_score(y_test, y_pred, average='weighted')


# In[22]:


print(f'One Hidden Layer ANN Precision: {Precision:.2f}' )
print(f'One Hidden Layer ANN Recall: {Recall:.2f}' )
print(f'One Hidden Layer ANN F1_score: {F1_score:.2f}' )


# In[23]:


crime_type_mapping = {
    0: 'ARSON',
    1: 'ASSAULT',
    2: 'BATTERY',
    3: 'BURGLARY',
    4: 'CONCEALED CARRY LICENSE VIOLATION',
    5: 'CRIM SEXUAL ASSAULT',
    6: 'CRIMINAL DAMAGE',
    7: 'CRIMINAL SEXUAL ASSAULT',
    8: 'CRIMINAL TRESPASS',
    9: 'DECEPTIVE PRACTICE',
    10: 'DOMESTIC VIOLENCE',
    11: 'GAMBLING',
    12: 'HOMICIDE',
    13: 'HUMAN TRAFFICKING',
    14: 'INTERFERENCE WITH PUBLIC OFFICER',
    15: 'INTIMIDATION',
    16: 'KIDNAPPING',
    17: 'LIQUOR LAW VIOLATION',
    18: 'MOTOR VEHICLE THEFT',
    19: 'NARCOTICS',
    20: 'NON - CRIMINAL',
    21: 'NON-CRIMINAL',
    22: 'NON-CRIMINAL (SUBJECT SPECIFIED)',
    23: 'OBSCENITY',
    24: 'OFFENSE INVOLVING CHILDREN',
    25: 'OTHER NARCOTIC VIOLATION',
    26: 'OTHER OFFENSE',
    27: 'PROSTITUTION',
    28: 'PUBLIC INDECENCY',
    29: 'PUBLIC PEACE VIOLATION',
    30: 'RITUALISM',
    31: 'ROBBERY',
    32: 'SEX OFFENSE',
    33: 'STALKING',
    34: 'THEFT',
    35: 'WEAPONS VIOLATION'
}


# In[24]:


y_test_pred = model.predict(X_test)


# In[25]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np


n_classes = y_test_onehot.shape[1] if len(y_test_onehot.shape) > 1 else 1

class_counts = np.sum(y_test_onehot, axis=0)
top_10_classes = np.argsort(class_counts)[-10:][::-1]
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    if i != 10:
        fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_test_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(),  y_test_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure(figsize=(8, 6))

# Plot individual class ROC curves
for i in top_10_classes:
    if i != 10:
        plt.plot(fpr[i], tpr[i], label=f'{crime_type_mapping[i]} (AUC = {roc_auc[i]:.5f})')

# Plot micro-average ROC curve
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average (AUC = {0:0.5f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

# Plot random guessing
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='right', bbox_to_anchor=(1.6, 0.5))
plt.show()

# Calculate the overall ROC AUC score
#macro_roc_auc = roc_auc_score(y_test_onehot, y_test_pred, average='macro')
#print(f'Macro-average ROC AUC: {macro_roc_auc:.2f}')


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

mask = cm == 0
# Display the confusion matrix using a heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', linewidths=.5, square=True, annot_kws={"size": 9}, mask = mask,cbar_kws={"shrink": 0.8})
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=2, square=False, annot_kws={"size": 10}, cbar_kws={"shrink": 0.6},mask =mask)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
plt.savefig('confusion_matrix.png')


# In[ ]:





# In[ ]:




