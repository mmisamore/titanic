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

# Try PCA in the pipeline
# from sklearn.decomposition import PCA
# pca = PCA(n_components=5,whiten=True)
# xTrain = pd.DataFrame(pca.fit_transform(xTrain.drop('PassengerId',axis=1)))
# xTrain['PassengerId'] = train['PassengerId']

# Deploy a classifier
# from sklearn.grid_search import GridSearchCV

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=10) # Accuracy = 0.81 
 
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier() # Accuracy = 0.78
 
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=1000,criterion='entropy') # 0.79

# Split into artificial train and test sets
# from sklearn.cross_validation import train_test_split
# featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(
#     xTrain.drop('PassengerId',axis=1), 
#     yTrain, 
#     train_size=0.20 
# )

# Fit and predict
# clf.fit(xTrain.drop('PassengerId',axis=1),yTrain)
# print zip(xTrain.columns.values, clf.feature_importances_)
# exit(0)

# clf.fit(featuresTrain,labelsTrain)

xTrain['Survived'] = train['Survived']
# print xTrain.corr() - Tells us that females is the first split, more likely to survive
# xTrain = xTrain[xTrain['Sex'] == 1].drop('Sex',axis=1)
# Class 0 more likely to have survived than not for Females
# print len(xTrain[(xTrain['Pclass'] == 0) & (xTrain['Survived'] == 0)])
# Class 0.5 mostly survived as well for females
# print len(xTrain[(xTrain['Pclass'] == 0.5) & (xTrain['Survived'] == 0)])
# Class 1 is a completely even split for females: just as likely to survive or perish
# print len(xTrain[(xTrain['Pclass'] == 1) & (xTrain['Survived'] == 0)])
# xTrain = xTrain[xTrain['Pclass'] == 1].drop('Pclass',axis=1)
# If Embarked = 0, more likely to perish than survive
# print len(xTrain[(xTrain['Embarked'] == 0) & (xTrain['Survived'] == 0)])
# If Embarked = 0.5, more likely to survive than perish
# print len(xTrain[(xTrain['Embarked'] == 0.5) & (xTrain['Survived'] == 1)])
# If Embarked = 1, more likely to survive than perish
# print len(xTrain[(xTrain['Embarked'] == 1) & (xTrain['Survived'] == 1)])

# Suggests that cabin = 1 for Men correlated with lower survival rate
# xTrain = xTrain[xTrain['Sex'] == 0].drop('Sex',axis=1)
# print xTrain.corr()

def predictManual(record):
    if record.Sex == 0:
        return 0            
    else: # Females
        if record.Pclass == 0:
            return 1
        elif record.Pclass == 0.5:
            return 1
        else:
            if record.Embarked == 0:
                return 0
            elif record.Embarked == 0.5:
                return 1
            else:
                return 1

# Put together a list of predictions 
pred = [predictManual(r) for r in xTest.itertuples()]

# pred = clf.predict(featuresTest)


# SUBMISSIONS
# clf.fit(xTrain.drop('PassengerId',axis=1),yTrain)
# pred = clf.predict(xTest.drop('PassengerId',axis=1))

# Create and write the submission to disk
submission = pd.DataFrame({'Survived': pred}, index=xTest['PassengerId'])
submission.to_csv("submission.csv")

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
