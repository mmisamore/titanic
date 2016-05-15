import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import radviz

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

### Begin Feature Engineering
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# Code gender feature
sexMap = {'male': 0, 'female': 1}
train.replace({'Sex': sexMap}, inplace=True)
test.replace({'Sex': sexMap}, inplace=True)

# Code embarked feature
embarkedMap = {'S': 0, 'C': 1, 'Q': 2}
train.replace({'Embarked': embarkedMap}, inplace=True)
test.replace({'Embarked': embarkedMap}, inplace=True)
train['Embarked'].fillna(0, inplace=True)
test['Embarked'].fillna(0, inplace=True)

# Match last part of ticket number
ticketRe = r"(\d+)$"
def regexHelper(regex, s):
    if re.search(regex, s):
        match = re.search(regex, s)
        return float(match.group(1))
    else:
        return None 

# Ticket number as a Feature
train['Ticket'] = [regexHelper(ticketRe,ticket) for ticket in train['Ticket']]
test['Ticket']  = [regexHelper(ticketRe,ticket) for ticket in test['Ticket']]
train['Ticket'].fillna(pd.Series.mean(train['Ticket']), inplace=True)
test['Ticket'].fillna(pd.Series.mean(test['Ticket']), inplace=True)

# Cabin section as a Feature
def cabinHelper(c):
    if c == 'n':
        return 'NaN'
    else:
        return c

cabinMap = {'NaN': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}
train['Cabin'] = [cabinHelper(str(c)[0]) for c in train['Cabin']]
test['Cabin']  = [cabinHelper(str(c)[0]) for c in test['Cabin']]
train.replace({'Cabin': cabinMap}, inplace=True)
test.replace({'Cabin': cabinMap}, inplace=True)

# Fix the Age feature to remove NaN
avgAgeTrain = pd.Series.mean(train['Age'])
avgAgeTest  = pd.Series.mean(test['Age'])
train['Age'].fillna(avgAgeTrain, inplace=True)
test['Age'].fillna(avgAgeTest, inplace=True)

# Finalize before scaling
xTrain = train.drop(['Survived','PassengerId'],axis=1)
yTrain = train['Survived']

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Debugging: value counts for all attributes
# print [(attr, xTrain[attr].value_counts(dropna=False)) for attr in xTrain.columns.values]

# Rescale without losing column headers
for attr in xTrain.columns.values:
    xTrain[attr] = scaler.fit_transform(xTrain[attr].reshape(-1,1))

# Add back PassengerId after rescaling
xTrain['PassengerId'] = train['PassengerId']

# Begin analysis
fullSet = xTrain.copy()
fullSet['Survived'] = yTrain

# Strong correlations with Survived: Sex, Pclass, Cabin, Fare, Embarked
# print fullSet.corr()

# Deploy a classifier
from sklearn.grid_search import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
c = KNeighborsClassifier() # F-score = 0.746, Accuracy = 0.81
clf = GridSearchCV(c, {'n_neighbors': range(1,40)})

# from sklearn.tree import DecisionTreeClassifier
# c = DecisionTreeClassifier() # F-Score = 0.74 
# clf = GridSearchCV(c, {'min_samples_split': range(1,40)})

from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier() # F-Score = 0.75 
from sklearn.ensemble import AdaBoostClassifier 
# clf = AdaBoostClassifier() # F-Score = 0.75 
from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB() # F-Score = 0.73
from sklearn.linear_model import RidgeClassifierCV
# clf = RidgeClassifierCV() # F-score = 0.74
from sklearn.ensemble import BaggingClassifier
# clf = BaggingClassifier() # F-score = 0.75
from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression() # F-score = 0.75

# 5-fold cross-validation score
from sklearn.cross_validation import cross_val_score
print "Cross val score: ", cross_val_score(clf, 
    xTrain.drop('PassengerId',axis=1), yTrain, cv=5
)

# Split into artificial train and test sets
from sklearn.cross_validation import train_test_split
featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
    xTrain.drop('PassengerId',axis=1), yTrain, train_size=0.80, random_state=42
)

# Fit and predict
clf.fit(featuresTrain,labelsTrain)
pred = clf.predict(featuresTest)

# Print feature importances for DecisionTreeClassifier
# print "Feature importances: ", zip(featuresTrain.columns.values,
#        clf.feature_importances_)

# Scoring via accuracy
from sklearn.metrics import accuracy_score
print "Accuracy: ", accuracy_score(labelsTest,pred)

# Scoring via precision and recall
from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, _ = precision_recall_fscore_support(
    labelsTest,pred,average='binary'
)

# Precision and recall for the artificial train/test split
print "Precision: ", precision
print "Recall: ", recall
print "F-Score: ", fscore

