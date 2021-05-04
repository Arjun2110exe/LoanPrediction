#!/usr/bin/env python
# coding: utf-8

# # Classifier for Loan Deafult IBM-Coursera

# ###### This is a peer graded assessment test/examination of IBM-COursera certification Machine Learning With Pyton. We will be doing classification problem on Loan_train dataset to predict the defaulting and genuine customers.
# ###### Logistic regression, Support Vector Machine, K-Nearest Neighbour and Decision Tree models are built , applied on the data set and evaluated using Jaccard index ,F1-score ,LogLoass as mentioned in the test question.

# In[1]:


loan_train_csv= "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv"
loan_test_csv = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv"


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#reading loan_train_csv

df_train= pd.read_csv(loan_train_csv,  parse_dates=True)
df_test = pd.read_csv(loan_test_csv,  parse_dates=True)


# In[4]:


df_train.shape


# In[5]:


df_train=df_train.append(df_test)


# In[6]:


df_train.shape


# In[7]:


#rename columns
columns = df_train.columns
columns


# In[8]:


#rename columns 
df_train.rename(columns = { columns[0]:"one" , columns[1]:"two"}, inplace = True)
df_train.columns


# In[9]:


#check data types
df_train.dtypes


# In[10]:


#converting string dates  to datetime 
df_train['effective_date'] = pd.to_datetime(df_train['effective_date'])
df_train['due_date'] = pd.to_datetime(df_train['due_date'])
df_train.dtypes


# In[11]:


#check null values
df_train.isna().sum()


# In[12]:


df_train.head(10)


# In[13]:


#checking types of unique values 
df_train.education.unique() # 4 unique
df_train.loan_status.unique() #two uniques


# In[14]:


#convrt gender colum to numeric values -- 0- male, 1- female
from sklearn.preprocessing import LabelEncoder
genderEncoder= LabelEncoder()
df_train.Gender = genderEncoder.fit_transform(df_train.Gender)
df_train.head()


# In[15]:


edu_dummy= pd.get_dummies(df_train.education)
edu_dummy
df_train.drop(axis=1, columns= "education", inplace= True)
df_train = pd.concat( [df_train,edu_dummy ], axis=1 )


# In[16]:


#convert loan_status  to numeric values 1- paid, 0- default
loan_status_encoder =LabelEncoder()
df_train.loan_status = loan_status_encoder.fit_transform(df_train.loan_status)
df_train.head()


# In[17]:


# create method to add date parts . This method is a part of fast ai library
from pandas.api.types import *
import re
def add_datepart(df, fldname, drop=True, time=False, errors="raise"):	
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    
    """
    
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

def is_date(x): return np.issubdtype(x.dtype, np.datetime64)


# In[18]:


#separating target varible and independent avriables
X = df_train.loc[:, df_train.columns != 'loan_status']
Y = df_train.loan_status


# In[19]:


add_datepart(X, fldname= 'effective_date')
add_datepart(X, fldname= 'due_date')
X


# In[20]:


X.dtypes


# In[21]:


from sklearn.preprocessing import StandardScaler
X_normalized = StandardScaler().fit(X).transform(X)
X_normalized[0:3]


# # Feature Selection PCA

# In[22]:


from sklearn import decomposition
pca = decomposition.PCA(n_components=4)
X = pca.fit_transform(X)


# # Train Test Split

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split (X, Y, test_size= 0.2, random_state= 0)


# In[24]:


#class counts in target variable
print(df_train.loan_status.value_counts())

df_train.loan_status.value_counts().plot.bar(rot=0)
plot.show()

print(Y_valid.value_counts())
Y_valid.value_counts().plot.bar(rot=0)
plot.show()


# # Building model using Logistic Regression

# In[25]:


#Logistic model
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,Y_train)

#
y_pred_LR =logreg.predict(X_valid)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_valid, y_pred_LR))

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_valid, y_pred_LR)


# # Building model using Decision Tree

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
DT.fit(X_train,Y_train)
#DT

y_pred_DT=DT.predict(X_valid)
print(accuracy_score(Y_valid, y_pred_DT))


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_valid, y_pred_DT)


# ##### This is the best model

# # Building model using K- Nearest Neighbour

# In[27]:


from sklearn.neighbors import KNeighborsClassifier

kmax=20
mean_accuracy=np.zeros(kmax-1)

for k in range(1,kmax):
    
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train,Y_train)
    ypred = knn.predict(X_valid)
    mean_accuracy[k-1]=np.mean(ypred==Y_valid);
    
    


plot.plot(np.arange(19),mean_accuracy)
plot.xlabel("K value")
plot.ylabel("mean accuracy")
plot.show()
mean_accuracy


# K=4 is the best K value

# In[29]:


knn = KNeighborsClassifier(n_neighbors=5).fit(X_train,Y_train)
y_pred_knn = knn.predict(X_valid)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_valid, y_pred_knn))

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_valid, y_pred_knn)


# # Building model using Support Vector Machine

# In[30]:


from sklearn import svm
svm = svm.SVC(probability=True)
svm.fit(X_train, Y_train)

y_pred_svm = svm.predict(X_valid)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_valid, y_pred_svm))

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_valid, y_pred_svm)


# # Finding the best Model . Evaluation Using Jaccard Index, F1- Score and and Log Loss

# In[31]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# # Jaccard Similarity

# In[36]:


# Jaccard for LR 
jcLR = round(jaccard_similarity_score(Y_valid, y_pred_LR),2)
# Jaccard Decision Trees
jcDT = round(jaccard_similarity_score(Y_valid, y_pred_DT),2)
#Jaccard for KNN
jcKNN = round(jaccard_similarity_score(Y_valid, y_pred_knn),2)
# Jaccard for SVM
jcSVM = round(jaccard_similarity_score(Y_valid, y_pred_svm),2)

list_jaccard_similarity  = [jcLR, jcDT, jcKNN, jcSVM]
list_jaccard_similarity


# # F1_score

# In[37]:


fLR = round(f1_score(Y_valid, y_pred_LR, average='weighted'), 2)
fDT = round(f1_score(Y_valid, y_pred_DT, average='weighted'), 2)
fKNN = round(f1_score(Y_valid, y_pred_knn, average='weighted'), 2)
fSVM = round(f1_score(Y_valid, y_pred_svm, average='weighted'),2 )

list_f1_score = [fLR, fDT, fKNN, fSVM]
list_f1_score


# # Log Loss

# In[38]:


lgloss_LR =log_loss(Y_valid, logreg.predict_proba(X_valid))
lgloss_DT = round(log_loss(Y_valid, DT.predict_proba(X_valid)), 2)
lgloss_KNN =round(log_loss(Y_valid, knn.predict_proba(X_valid)), 2)
lgloss_SVM =round(log_loss(Y_valid, svm.predict_proba(X_valid,)), 2)


list_log_loss = [lgloss_LR, lgloss_DT, lgloss_KNN, lgloss_SVM]
list_log_loss


# # Final Report on Evaluation

# In[39]:


ind = np.arange(4) 
width = 0.35       
plot.bar(ind, list_jaccard_similarity, width, label='Jaccard Index')
plot.bar(ind + width,  list_f1_score, width,
    label='F1-Score')

plot.ylabel('Scores')
plot.title('Comparision between Jacard similarity and F1- Score')

plot.xticks(ind + width / 2, ('LR', 'DT', 'KNN', 'SVM'))
plot.legend(loc='best')
plot.show()


# In[40]:


width = 0.35       
plot.bar(ind, list_log_loss, width, label='Log Loss Report')

plot.ylabel('Scores')
plot.title('Log Loss Report')

plot.xticks(ind , ('LR', 'DT', 'KNN', 'SVM'))
plot.legend(loc='best')
plot.show()


# ##### The lowest log loss, highest jaccard and F1-score in DT. LR stands at second rank. Highest log loss and lowest jaccard similary,f1-score in SVM. Hence, decision Tree is thebest model for the given problem. AFter that LR is the second best.

# # Precision- Recall Curve

# The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall).

# In[41]:


#method to plot - precision- recall curve for model evaluation
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
def plot_prec_recall_curv(ytrue, ypred, title):
    #for a good model  precision and recall both should be high. 
    precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
    auc1 = auc(recall, precision)
    # calculate average precision score
    ap = average_precision_score(ytrue, ypred)
    print(' auc=%.3f ap=%.3f' % ( auc1, ap))
    # plot the precision-recall curve for the model
    plot.plot(recall, precision, marker='.')
    # show the plot
    plot.xlabel("recall")
    plot.ylabel("precision")
    plot.ylim([0.0, 1.05])
    plot.xlim([0.0, 1.0])
    plot.title(title+" PRECISION-RECALL CURVE , AUC: "+ str(auc1) )
    plot.fill_between(recall, precision, alpha=0.2, color='b')
    plot.show()


# In[42]:


#plot recall precision curve
plot_prec_recall_curv(Y_valid, y_pred_LR, title="Logistic Regression Model")

plot_prec_recall_curv(Y_valid, y_pred_DT, title="Decision Tree")

plot_prec_recall_curv(Y_valid, y_pred_knn, title="KNN")

plot_prec_recall_curv(Y_valid, y_pred_svm ,title="SVM")


# ##### We can observe from all curves that decision tree has the highest area under precision-recall curve. It gives the best performance. Second best model is the linear regression.

# # F1-score, Jaccard similary and log loss together:

# In[43]:


# printing F1-score, Jaccard similary and log loss together:
df = pd.DataFrame(
    list(zip(
        ['LR','DT','KNN','SVM'],list_jaccard_similarity, list_f1_score, list_log_loss)),
    columns = ['Model', 'Jaccard','F1-Score','Log-Loss']
)

df = pd.DataFrame.from_dict(df)

print(df)


# In[ ]:




