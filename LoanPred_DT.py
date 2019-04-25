#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
np.random.seed(1)
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# # Input Dataset 

# In[2]:


le = LabelEncoder()
df = pd.read_csv('./Data.csv',index_col=0)


# In[3]:


df =  shuffle(df).reset_index(drop=True)


# # Data Exploration

# In[4]:


df.head()


# In[5]:


print('Missing Values: ')


# In[6]:


df.isnull().sum()


# In[7]:


print('Data Description: ')


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


for cols in ['sex','region','married','car','saving_acc', 'current_acc', 'mortgage', 'loan','region']:
    df[cols] = df[cols].astype('category')
df['age'] = df['age'].astype('int32')
df['income'] = df['income'].astype('float64')
df['children'] = df['children'].astype('int32')


# In[11]:


df.info()


# # Contigency tables for catagorical attributes

# In[12]:


pd.crosstab(df['sex'],df['loan'])


# In[13]:


pd.crosstab(df['region'],df['loan'])


# In[14]:


pd.crosstab(df['married'],df['loan'])


# In[15]:


pd.crosstab(df['children'],df['loan'])


# In[16]:


pd.crosstab(df['car'],df['loan'])


# In[17]:


pd.crosstab(df['saving_acc'],df['loan'])


# In[18]:


pd.crosstab(df['current_acc'],df['loan'])


# In[19]:


pd.crosstab(df['mortgage'],df['loan'])


# In[20]:


df.corr()


# ## Distribution of Age & Income with Loan

# In[21]:


x1 = x2 = []


# In[22]:


x1 = df.loc[df.loan == 'YES','age']
x2 = df.loc[df.loan == 'NO','age']


# In[60]:


from matplotlib import pyplot as plt 
fig = plt.figure(figsize = (25,10))
ax = fig.add_subplot(121)
legend = ['YES','NO']
ax.set(title = "Distribution of Age with Loan", ylabel = 'Proportion', xlabel = 'AGE')
ax.hist([x1,x2] ,color = ['blue','orange'], bins = 30, normed = False, label=['blue','orange'])
plt.legend(legend)
plt.show()


# In[24]:


k1 = k2 = []


# In[25]:


k1 = df.loc[df.loan == 'YES','income']
k2 = df.loc[df.loan == 'NO','income']


# In[26]:


fig1 = plt.figure(figsize = (25,10))
ax1 = fig1.add_subplot(121)
legend = ['YES','NO']
ax1.set(title = "Distribution of Income with Loan", ylabel = 'Proportion', xlabel = 'Income')
ax1.hist([k1,k2] ,color = ['blue','orange'], bins = 30, normed = False)
plt.legend(legend)
plt.show()


# ## Age & Income Bucketing

# In[61]:


y = df.iloc[:,[10]]
X = df.iloc[:,0:10]
z = list(X.columns.values)
z.remove('income')
y = le.fit_transform(y)
for name in z:
    X[name] = le.fit_transform(X[name])


# In[28]:


def age_convert(age):
    if (age < 35):
        return 1
    elif (35<= age <= 55):
        return 2
    else:
        return 3


# In[29]:


def income_convert(income):
    if (income < 5000):
        return 1
    elif (5000 <= income <= 33000):
        return 2
    elif (33000 < income < 61000):
        return 3
    else:
        return 4


# In[30]:


X['age_conv'] = X['age'].apply(lambda k:age_convert(k))
X = X.drop(labels = ['age'], axis = 1)


# In[31]:


import math
X['income'] = X['income'].apply(lambda X:math.floor(X))
X['income_conv'] = X['income'].apply(lambda k:income_convert(k))
X = X.drop(labels = ['income'], axis = 1)


# # Feature Selection

# In[32]:


feature_cols = list(X.columns.values)


# In[33]:


print(feature_cols)


# # Model Construction

# In[34]:


from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# ### Preprocessing on catagorical variables

# In[35]:


from sklearn import preprocessing
pp = preprocessing.LabelEncoder()
for cols in ['sex', 'region', 'married', 'children', 'car', 'saving_acc', 'current_acc', 'mortgage']:
    X[cols] = pp.fit_transform(X[cols])
y = pp.fit_transform(y)


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1) 


# ## Choosing Decision Tree parameters

# In[37]:


dtparams = {"criterion": ["gini", "entropy"], 
              'max_depth':range(4,16), 
              'min_samples_leaf' : range(4,20), 
              'max_leaf_nodes':range(10,15)}


# In[38]:


gscv = GridSearchCV(DecisionTreeClassifier(), dtparams, n_jobs = 4, cv = 10, scoring = 'f1', refit = 'f1')
gscv.fit(X = X_train, y = y_train)


# In[39]:


print (gscv.best_score_, gscv.best_params_)


# In[62]:


res_df = pd.DataFrame(gscv.cv_results_)


# In[41]:


res_df['mean_train_score'].max()


# In[42]:


model = DecisionTreeClassifier(criterion = "entropy", random_state = 1, max_depth = 6, 
                               max_leaf_nodes = 14, min_samples_leaf = 8 )


# In[43]:


clf = model.fit(X_train, y_train)


# In[44]:


y_pred = model.predict(X_test)


# In[45]:


confusion_matrix(y_test, y_pred)


# # Model Visualization

# In[46]:


from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO  


# In[47]:


from IPython.display import Image  
dot_data = StringIO()
export_graphviz(model, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# # Model Evaluation

# In[48]:


from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score


# In[72]:


accuracy = accuracy_score(y_test, y_pred)
print (str(round(accuracy * 100,1) ) + '% accuracy')


# In[50]:


rascore = roc_auc_score(y_test, y_pred)
print(rascore)


# ### Precision vs Recall plot

# In[51]:


cls_report = classification_report(y_test,y_pred)
print cls_report 


# In[52]:


precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.step(recall, precision, color='g', alpha=0.2, where='post')
plt.xlabel('Recall value')
plt.ylabel('Precision value')
plt.ylim([0.0, 1.1])
plt.xlim([0.0, 1.0])
plt.title('Precision vs Recall plot: ')


# # ROC Curve Plotting

# In[53]:


def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[54]:


model.fit(X_train, y_train) 


# In[55]:


probs = model.predict_proba(X_test)  


# In[56]:


probs = probs[:, 1]


# In[57]:


auc = roc_auc_score(y_test, y_pred)  
print('AUC: %.2f' % auc)


# In[58]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)  


# In[59]:


plot_roc_curve(fpr, tpr)  


# In[ ]:




