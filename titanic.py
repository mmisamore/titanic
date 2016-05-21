import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import radviz

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

### Begin Feature Engineering

# Code titles from Names
def title(name):
    if 'Mr.' in name:
        return 0
    elif 'Mrs.' in name or 'Mme.' in name:
        return 1
    elif 'Miss.' in name or 'Ms.' in name or 'Mlle.' in name:
        return 2
    elif 'Master.' in name:
        return 3
    elif 'Dr.' in name:
        return 4
    elif 'Countess' in name:
        return 5
    elif 'Rev.' in name:
        return 6
    elif 'Major.' in name or 'Col.' in name:
        return 7
    elif 'Sir.' in name or 'Lady.' in name or 'Don.' in name:
        return 8
    elif 'Capt.' in name:
        return 9
    else:
        return 10 

titlesTrain = pd.DataFrame({'titles': [title(n) for n in train['Name']]})
train['titles'] = titlesTrain['titles']

titlesTest = pd.DataFrame({'titles': [title(n) for n in test['Name']]})
test['titles'] = titlesTest['titles']

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# Code gender feature
sexMap = {'male': 0, 'female': 1}
train.replace({'Sex': sexMap}, inplace=True)
test.replace({'Sex': sexMap}, inplace=True)

# Embarked feature
embarkedMap = {'S': 0, 'C': 1, 'Q': 2}
train.replace({'Embarked': embarkedMap}, inplace=True)
train['Embarked'].fillna(0, inplace=True)
test.replace({'Embarked': embarkedMap}, inplace=True)
test['Embarked'].fillna(0, inplace=True)

# Match last part of ticket number
ticketNum = r"(\d+)$"
ticketPrefix = r"^([^\d\s]+)"
def regexHelper(regex, s):
    if re.search(regex, s):
        match = re.search(regex, s)
        return match.group(1)
    else:
        return None 

# Ticket number as a Feature
train['Ticket'] = pd.to_numeric([regexHelper(ticketNum,ticket) for ticket in
    train['Ticket']])
test['Ticket']  = pd.to_numeric([regexHelper(ticketNum,ticket) for ticket in
    test['Ticket']])

# Impute gaps
trainMin = pd.Series.min(train['Ticket'])
trainMax = pd.Series.max(train['Ticket'])
train['Ticket'].fillna((trainMax-trainMin)/2.0+trainMin,inplace=True)
testMin = pd.Series.min(test['Ticket'])
testMax = pd.Series.max(test['Ticket'])
test['Ticket'].fillna((testMax-testMin)/2.0+testMin,inplace=True)

# Cabin section as a Feature
cabinMap = {'n': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'T': 9 }
train['Cabin'] = [str(c)[0] for c in train['Cabin']]
train.replace({'Cabin': cabinMap}, inplace=True)
test['Cabin']  = [str(c)[0] for c in test['Cabin']]
test.replace({'Cabin': cabinMap}, inplace=True)

# Simplify to "had cabin or not"
train['Cabin'] = train['Cabin'] == 1
test['Cabin']  = test['Cabin'] == 1

# Fix the Age feature to remove NaN
train['Age'].fillna(28.0, inplace=True)
test['Age'].fillna(28.0, inplace=True)

# Fix the Fare feature to remove NaN/None
train['Fare'].fillna(15.0, inplace=True)
test['Fare'].fillna(15.0, inplace=True)

# Finalize before scaling
xTrain = train.drop(['Survived','PassengerId'],axis=1)
yTrain = train['Survived']
xTest  = test.drop(['PassengerId'],axis=1)

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Rescale without losing column headers
for attr in xTrain.columns.values:
    xTrain[attr] = scaler.fit_transform(xTrain[attr].reshape(-1,1))
for attr in xTest.columns.values:
    xTest[attr]  = scaler.fit_transform(xTest[attr].reshape(-1,1))

# Add back PassengerId after rescaling
xTrain['PassengerId'] = train['PassengerId']
xTest['PassengerId']  = test['PassengerId']

# Deploy a classifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()
# params = {
#     'p':         [1,2],
#     'leaf_size': [1,2,4,8,16,32,64],
#     'algorithm': ['ball_tree','kd_tree','brute'],
#     'weights':   ['uniform','distance'],
#     'n_neighbors': [1,2,4,6,8,10,15,20]
# }
# gs = GridSearchCV(clf, params, cv=5, scoring='accuracy')

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
params = {
    'n_estimators': [50],
    'max_features': [1],
    'max_depth':    [10],
    'random_state': [1,2,4,8,16,32],
    'n_jobs':       [-1]
}
gs = GridSearchCV(clf, params, cv=5, scoring='accuracy') 

# Split into artificial train and test sets
from sklearn.cross_validation import train_test_split
featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
    xTrain.drop('PassengerId',axis=1), 
    yTrain, 
    train_size=0.80,
    random_state=42
)

# Fit and predict
gs.fit(featuresTrain,labelsTrain)
pred = gs.predict(featuresTest)
 
# Manually built decision tree with a decent score. Tuned RF is competitive with
# this. TODO: Factor in new "title" feature.
# def predictManual(record):
#     if record.Sex == 0:
#         return 0
#     else: # Females
#         if record.Pclass == 0:
#             return 1
#         elif record.Pclass == 0.5:
#             return 1
#         else:
#             if record.Fare > 0.05:
#                 return 0
#             else:
#                 if record.Embarked == 0:
#                     return 0
#                 else:
#                     return 1

# Put together a list of predictions 
# pred = [predictManual(r) for r in xTest.itertuples()]
 
from sklearn.metrics import classification_report  
print classification_report(labelsTest,pred)
exit(0)
 
# SUBMISSIONS
# gs.fit(xTrain.drop('PassengerId',axis=1),yTrain)
# pred = gs.predict(xTest.drop('PassengerId',axis=1))

# Create and write the submission to disk
# submission = pd.DataFrame({'Survived': pred}, index=xTest['PassengerId'])
# submission.to_csv("submission.csv")

'''
# Scoring for training before submission:

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
'''
