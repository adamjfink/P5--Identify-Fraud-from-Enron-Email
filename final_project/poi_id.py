#!/usr/bin/python

import sys
import pickle
import matplotlib
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from copy import copy
from numpy import mean
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.svm import SVC
from tester import dump_classifier_and_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi']

POI_label = "poi" 

features_list = [POI_label] + financial_features + email_features

### Exploring Dataset ###

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Number of People
print ("Number of People in Dataset:")
print len(data_dict)    
    
### Number of Features
print ("Number of Features per Person:")
no_of_features = len(data_dict[data_dict.keys()[0]])     
print (no_of_features)

### Number of POI in dataset
print ("Number of POI in DataSet:")
i = 0
for key in data_dict:
    if data_dict[key]['poi'] == True:
        i = i + 1
print i  

### Missing Features
featureList = data_dict['ALLEN PHILLIP K'].keys()
print('Each person has this many features avalible:')
print len(featureList)
missingList = {}
for feature in featureList:
    missingList[feature] = 0
for person in data_dict.keys():
    records = 0
    for feature in featureList:
        if data_dict[person][feature] == 'NaN':
            missingList[feature] += 1
        else:
            records += 1
print('Number of Missing Values for Each Feature:')
for feature in featureList:
    print("%s: %d" % (feature, missingList[feature]))

### Task 2: Remove outliers ###

### Plot to find outliers
for dic in data_dict.values():
    matplotlib.pyplot.scatter(dic['salary'] , dic['bonus'])
matplotlib.pyplot.xlabel("Salary")
matplotlib.pyplot.ylabel("Bonus")
matplotlib.pyplot.show()

### remove outliers
def outlier_remove(dict_object, keys):
    for key in keys:
        dict_object.pop(key, 0)

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
outlier_remove(data_dict, outliers)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
### my_feature_list = copy(features_list)

### Calculate Persons Wealth
for item in my_dataset:
    worker = my_dataset[item]
    if (all([worker['salary'] != 'NaN',
                worker['total_stock_value'] != 'NaN',
                worker['exercised_stock_options'] != 'NaN',
                worker['bonus'] != 'NaN'])):
        worker['wealth'] = sum([worker[field] for field in ['salary',
                                                            'total_stock_value',
                                                        'exercised_stock_options',
                                                                        'bonus']])
    else:
        worker['wealth'] = 'NaN'

my_feature_list = copy(features_list)+ ['wealth']
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

print ("The Best Features Are:")

from sklearn.feature_selection import SelectKBest
kNum = 10

### Find K Best
def k_best_find(data_dict, features_list, k):

    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    return k_best_features

best_features = k_best_find(my_dataset, my_feature_list, kNum)

my_feature_list = [POI_label] + best_features.keys()

print ("The Selected Features Are:")
print (len(my_feature_list) - 1, my_feature_list[1:])

### take out and split features
data = featureFormat(my_dataset, my_feature_list)
labels, features = targetFeatureSplit(data)

### Scale Features
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.linear_model import LogisticRegression

l_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])

from sklearn.naive_bayes import GaussianNB

g_clf = GaussianNB()

from sklearn.cluster import KMeans

k_clf = KMeans(n_clusters=2, tol=0.001)

s_clf = SVC(kernel='rbf', C=1000)

from sklearn.ensemble import RandomForestClassifier

r_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels,test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        
    print "Precision:"
    print (mean(precision))
    print "Recall:"
    print (mean(recall))
    return mean(precision), mean(recall)

evaluate_clf(g_clf, features, labels)
evaluate_clf(l_clf, features, labels)
evaluate_clf(k_clf, features, labels)
evaluate_clf(s_clf, features, labels)
evaluate_clf(r_clf, features, labels)

l_tuning_parameters = {'tol': [10**-1,10**-5,10**-10], 'C': [ 0.5,1,10,10**5,10**10]}
LF = GridSearchCV(LogisticRegression(), l_tuning_parameters, scoring = 'recall')
LF.fit(features, labels)
print("Best parameters are for Logistic Regression Are:")
print(LF.best_params_)

r_tuning_parameters = {'n_estimators': [2,3,5,10], 'criterion': ['gini','entropy']}
RF = GridSearchCV(RandomForestClassifier(), r_tuning_parameters, scoring = 'recall')
RF.fit(features, labels)
print("Best parameters are for Random Forest Are:")
print(RF.best_params_)

clf = l_clf
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)