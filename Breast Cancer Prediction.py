# Data Preprocessing

# Dataset Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Importing the libraries and dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/content/breast_cancer.csv')

dataset.head()

## Data Exploration

dataset.shape

dataset.info()

dataset.select_dtypes(include='object').columns

len(dataset.select_dtypes(include='object').columns)

dataset.select_dtypes(include=['float64', 'int64']).columns

len(dataset.select_dtypes(include=['float64','int64']).columns)

# statistical summary
dataset.describe()

## Dealing with the missing values

dataset.isnull().values.any()

dataset.isnull().values.sum()

dataset.columns[dataset.isnull().any()]

len(dataset.columns[dataset.isnull().any()])

dataset['Unnamed: 32'].count()

dataset = dataset.drop(columns='Unnamed: 32')

dataset.shape

dataset.isnull().values.any()

## Dealing with categorical data

dataset.select_dtypes(include='object').columns

dataset['diagnosis'].unique()

dataset['diagnosis'].nunique()

# One hot encoding
dataset = pd.get_dummies(dataset, drop_first=True)

dataset.head()

## Countplot

sns.countplot(dataset['diagnosis_M'], label='Count')
plt.show()

# B (0) values
(dataset.diagnosis_M == 0).sum()

# M (1) values
(dataset.diagnosis_M == 1).sum()

## Correlation matrix and heatmap

dataset_2 = dataset.drop(columns='diagnosis_M')

dataset_2.head()

dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
    figsize=(20,10),
    title='Correlation with diagnosis_M',
    rot=45,
    grid=True
)

# correlation matrix
corr =  dataset.corr()

# heatmap
plt.figure(figsize = (20,10))
sns.heatmap(corr, annot=True)
plt.show()
#

corr

## Splitting this dataset train and test set

dataset.head()

# matrix of features / independent variables
x = dataset.iloc[:, 1:-1].values

x.shape

# target variable / dependent variable
y = dataset.iloc[:, -1].values

y.shape

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train.shape

x_test.shape

y_test.shape

## Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_test



# Building the model

## Logistic Regression

from sklearn.linear_model import LogisticRegression

classifir_lr = LogisticRegression(random_state=0)

classifir_lr.fit(x_train, y_train)

y_pred = classifir_lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, f1_score, precision, recall]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

results

cm = confusion_matrix(y_test, y_pred)

print(cm)



## Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifir_lr, X=x_train, y=y_train, cv=10)

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

## Random Forest

from sklearn.ensemble import RandomForestClassifier

classifier_rm = RandomForestClassifier(random_state=0)
classifier_rm.fit(x_train, y_train)

y_pred = classifier_rm.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
model_results = pd.DataFrame([['Random Forest', acc, f1, precision, recall]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
results = pd.concat([results, model_results], ignore_index=True)
results

cm = confusion_matrix(y_test, y_pred)
print(cm)

## Cross validation

from sklearn.model_selection import cross_val_score

accu = cross_val_score(estimator=classifier_rm, X=x_train, y=y_train, cv=10)

print("Accuracy is {:.2f} %".format(accu.mean()*100))
print("Standard Deviation is {:.2f} %".format(accu.std()*100))


# Randomized search to find the best parameters

from sklearn.model_selection import RandomizedSearchCV

parameters = {
    'penalty': ['l1', 'l2', 'elasticnet'],  # 'none' penalty is less commonly used
    'C': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    'solver': ['liblinear', 'saga']  # 'liblinear' supports l1 and l2, 'saga' supports l1, l2, and elasticnet
}

parameter_grid = [
    {
        'penalty': ['l2'],
        'solver': ['newton-cg', 'lbfgs', 'sag'],
        'C': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        'max_iter': [100, 200, 300]  # Increase the iterations for convergence
    },
    {
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'C': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        'max_iter': [100, 200, 300]  # Increase the iterations for convergence
    },
    {
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'C': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        'l1_ratio': [0.25, 0.5, 0.75],  # Only include l1_ratio with elasticnet
        'max_iter': [100, 200, 300]  # Increase the iterations for convergence
    }
]

random_search = RandomizedSearchCV(
    estimator=classifir_lr,
    n_iter=10,
    param_distributions=parameter_grid,
    scoring='roc_auc',
    n_jobs=-1,
    cv=10,
    verbose=3,
    error_score='raise'
)

random_search.fit(x_train, y_train)



random_search.best_params_



# Predicting a single observation

dataset.head()

single_obs = [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]

classifir_lr.predict(sc.transform(single_obs))

