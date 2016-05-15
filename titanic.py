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

# Embarked feature
embarkedMap = {'S': 1, 'C': 1, 'Q': 2}
train.replace({'Embarked': embarkedMap}, inplace=True)
train['Embarked'].fillna(1.0, inplace=True)
test.replace({'Embarked': embarkedMap}, inplace=True)
test['Embarked'].fillna(1.0, inplace=True)

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

# Begin analysis
# fullSet = xTrain.copy()
# fullSet['Survived'] = yTrain

# Strong correlations with Survived: Sex, Pclass, Cabin, Fare, Embarked
# print fullSet.corr()

# Try PCA in the pipeline
# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# xTrain = pd.DataFrame(pca.fit_transform(xTrain.drop('PassengerId',axis=1)))
# xTrain['PassengerId'] = train['PassengerId']

# Deploy a classifier
from sklearn.grid_search import GridSearchCV

# from sklearn.neighbors import KNeighborsClassifier
# c = KNeighborsClassifier() # Accuracy = 0.80 
# clf = GridSearchCV(c, {
#    'n_neighbors': range(1,40),
#    'weights': ['uniform','distance']
# })
 
from sklearn.tree import DecisionTreeClassifier
c = DecisionTreeClassifier() # Accuracy = 0.82 
clf = GridSearchCV(c, {
    'min_samples_split': range(1,20), 
    'max_features': range(1,len(xTrain.columns.values)-1),
    'random_state': [1,10,12,23,42]
})

# from sklearn.ensemble import RandomForestClassifier
# c = RandomForestClassifier() # Accuracy = 0.81
# clf = GridSearchCV(c, {
#    'min_samples_split': range(1,20), 
#     'n_estimators': range(5,15),
#     'random_state': [42]
# })

# from sklearn.ensemble import AdaBoostClassifier 
# c = AdaBoostClassifier() # Accuracy = 0.79 
# clf = GridSearchCV(c, {
#    'random_state': [1,10,12,23,42],
#    'learning_rate': [0.2,0.4,0.6,0.8,1.0]
# })

# Vanilla Linear SVC
# from sklearn.svm import LinearSVC
# c = LinearSVC() # Accuracy = 0.79 
# clf = GridSearchCV(c, {
#    'random_state': [1,10,12,23,42]
# })

# from sklearn.linear_model import LogisticRegression
# c = LogisticRegression() # Accuracy = 0.799
# clf = GridSearchCV(c, {
#    'random_state': [1,10,12,23,42]
# })

# from sklearn.svm import SVC
# c = SVC() # Accuracy = 0.793
# clf = GridSearchCV(c, {
#    'C': [0.5,1.0,5.0,10.0,20.0],
#    'kernel': ['linear','poly','rbf'],
#    'random_state': [1,10,12,23,42]
# })

# 5-fold cross-validation score
from sklearn.cross_validation import cross_val_score
print "Cross val score: ", cross_val_score(
    clf, 
    xTrain.drop('PassengerId',axis=1),
    yTrain, 
    cv=5
)

# Split into artificial train and test sets
from sklearn.cross_validation import train_test_split
featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
    xTrain.drop('PassengerId',axis=1), 
    yTrain, 
    train_size=0.80, 
    random_state=42
)

# Fit and predict
clf.fit(featuresTrain,labelsTrain)
pred = clf.predict(featuresTest)

# SUBMISSIONS
# clf.fit(xTrain.drop('PassengerId',axis=1),yTrain)
# pred = clf.predict(xTest.drop('PassengerId',axis=1))
# Create and write the submission to dis
# submission = pd.DataFrame({'Survived': pred}, index=xTest['PassengerId'])
# submission.to_csv("submission.csv")

# Scoring for training before submission:

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

