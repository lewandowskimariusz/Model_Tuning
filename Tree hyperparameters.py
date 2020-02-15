#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import pandas as pd


# In[2]:


#Load, change target and split data
df = pd.read_csv('indian_liver_patient_preprocessed.csv')
df.head()


# In[3]:


#Split data on data and target
X = df.drop(columns='Liver_disease')
y = df.Liver_disease
#Import libraries for preprocessing
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.4, random_state = 1 )


# In[5]:


# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Define params_dt
params_dt = {
             'max_depth': [2, 3, 4],
             'min_samples_leaf': [0.12, 0.14, 0.16, 0.18]
            }


# In[6]:


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator=dt,
                       param_grid=params_dt,
                       scoring='roc_auc',
                       cv=5,
                       n_jobs=-1)


# In[7]:


# Grid_d fit model
grid_dt.fit(X_train, y_train)


# In[9]:


# y_pred
y_pred = grid_dt.predict(X_test)


# In[11]:


# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_


# In[16]:


# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))


# In[ ]:




