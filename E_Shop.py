""""CA2 for B8IT106
by 
Pedro Mesquita Vasconcelos
and 
Rodolfo Ferreira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from sklearn.ensemble import RandomForestClassifier

# Importing dataset and examining it
dataset = pd.read_csv("E-Shop.csv")
pd.set_option('display.max_columns', None) # Will ensure that all columns are displayed
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# 3 Variables contain binomial information: Weekend, Transaction and VisitorType
# Creation of function to convert categorical features into Numerical features
def converter(column):
    if str(column).lower() == 'true' or str(column).lower() == 'new_visitor':
        return 1
    else:
        return 0

dataset['Weekend'] = dataset['Weekend'].apply(converter)
dataset['Transaction'] = dataset['Transaction'].apply(converter)
dataset['VisitorType'] = dataset['VisitorType'].apply(converter)
print(dataset.info())

#One of the categorical variables (Month) is not binomail, so we will create a new column for each month
categorical_features = ['Month']
final_data = pd.get_dummies(dataset, columns = categorical_features)
print(final_data.info())
print(dataset.head(10))

# Dividing dataset into label and feature sets. Transaction is our target variable
X = final_data.drop('Transaction', axis = 1) 
Y = final_data['Transaction'] 
print(type(X))
print(type(Y))
print(X.shape)
print(X.info())
print(Y.shape)

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

#Analyse balancing of data
print(pd.Series(Y).value_counts())
#In this analysis we will be focusing on the reduction of False Positives: meaning that we couldn't predict users that didn't buy 

# Dividing dataset into training and test sets: 70% training and 30% Test
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)

print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

# Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
smote = SMOTE(random_state = 101) #Selected one integer on random_state to get same results
X_train,Y_train = smote.fit_sample(X_train,Y_train)

print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())

# Tuning the random forest parameter 'n_estimators' and implementing cross-validation using Grid Search
rfc = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
grid_param = {'n_estimators': [50, 100, 150, 200, 250, 300]}

gd_sr = GridSearchCV(estimator=rfc, param_grid=grid_param, scoring='precision', cv=5)

gd_sr.fit(X_train, Y_train)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)

# Building random forest using the tuned parameter
rfc = RandomForestClassifier(n_estimators=150, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)
featimp = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])

# Selecting features with higher sifnificance and redefining feature set
X = final_data[['PageValue', 'ExitRate', 'ProductRelated_Duration', 'Administrative', 'ProductRelated', 'Administrative_Duration', 'BounceRate','Month_Nov','Informational','Informational_Duration','Month_May','VisitorType']]

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Dividing dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

rfc = RandomForestClassifier(n_estimators=150, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)

Y_pred = rfc.predict(X_test)
conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])
