
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


data_summary = pd.DataFrame.from_dict(data_dict,orient='index')
data_summary.replace('NaN',np.nan, inplace =True)
print data_summary.info()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options','other', 'long_term_incentive',
                 'restricted_stock', 'director_fees','from_poi_to_this_person',
              'from_this_person_to_poi','to_messages','from_messages']
email_list = ['email_address']


print "no. of selected features:",len(features_list)-1

### Task 2: Remove outliers

print "no. of obsevations in data:",len(data_dict)
print "person names in dataset:"

name = []
for person in data_dict.keys():
    name.append(person)
    if len(name) == 4:
        print '{:<40}{:<40}{:<40}{:<40}'.format(name[0],name[1],name[2],name[3])
        name = []
print '######################################################'
print '{:<50}{:<50}'.format(name[0],name[1])
print " so, total & the travel agency in the park is an outlier so we will remove from list"

non_poi =0
for x in data_dict.values():
    if x["poi"]:
         non_poi +=1
print "no.of persons in the data:", len(data_summary)
print "no.of POI's :",non_poi
print "no.of Non_POI's", len(data_dict) - non_poi         
#print len(data_summary[data_summary['poi']])

print "missing values in each features"
from sklearn import preprocessing

NaNs = [0 for  a in range(len(features_list))]
for a, p in enumerate(data_dict.values()):
    for b,q in enumerate(features_list):
        if p[q] == 'NaN':
            NaNs[b] +=1
print "##################"
for a,q in enumerate(features_list):
    print q,NaNs[a]
print "###################################################"
print "obseravtions in Total"

for name,x in data_dict.iteritems():
    if name == 'TOTAL':
        print x
print "##################################"
 
data_dict.pop('TOTAL',0)
data_dict.pop('The Travel Agency In the Park',0)
data_dict.pop('LOCKHART EUGENE E',0)
print "The outliers are removed"

print "no.of obsevations without TOTAL:",len(data_dict)     
####removing Nan's from each feature
for i in data_dict:
    for j in data_dict[i]:
        if data_dict[i][j] == 'NaN':
            data_dict[i][j] =0

print "######################################"
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset= data_dict
features_list=features_list

#new features
print 'emails from poi to these person new feature is created'

for employee in my_dataset.values():
    employee['fraction_from_poi']=0
    if float(employee['to_messages'])>0:
        employee['fraction_from_poi'] = float(employee['from_poi_to_this_person'])/float(employee['to_messages'])
        
print "@@@@@@@@@@@@@@@@@"
print 'emails to poi from the person new feature is createdsfl'
for employee in my_dataset.values() :
    employee['fraction_to_poi']=0
    if float(employee['from_messages'])>0:
        employee['fraction_to_poi'] = float(employee['from_this_person_to_poi'])/float(employee['from_messages'])
    

features_list.extend(['fraction_from_poi','fraction_to_poi'])
print "@@@@@@@@"
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest,f_classif


###cross validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection  import StratifiedShuffleSplit,GridSearchCV,cross_val_score
cvv=StratifiedShuffleSplit(n_splits=10, test_size=0.1,random_state=0)
cv_eva= StratifiedShuffleSplit(labels,100,random_state =42)
#print cv.get_n_splits(labels, features)

random_acc = []
random_pre = []
random_recall = []
ad_acc = []
ad_pre = []
ad_recall = []


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics
from time import time
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from tester import test_classifier
from feature_format import featureFormat,targetFeatureSplit

##please don't replace evaluation function any other test_classifiers

#here evaluation is best set one.
#evaluating and training the classifiers using cv.
def evaluation(grid,features,labels,cv,folds=1000):
    true_nve = 0
    false_nve = 0
    true_pve = 0
    false_pve =0
    for train_index, test_index in cv.split(features,labels):
        fea_train = []
        fea_test = []
        lab_train = []
        lab_test = []
        for k in train_index:
            fea_train.append(features[k])
            lab_train.append(labels[k])
        for l in test_index:
            fea_test.append(features[l])
            lab_test.append(labels[l])
        grid.fit(fea_train,lab_train)
        prediction = grid.predict(fea_test)
        for p,t in zip(prediction,lab_test):
            if p == 0 and t == 0:
                true_nve +=1
            elif p ==0 and t ==1:
                false_nve +=1
            elif p ==1 and t ==0:
                false_pve +=1
            elif p ==1 and t ==1:
                true_pve +=1
    predictions_t = true_nve +false_nve+false_pve+true_pve                     
    acc = round(1.0*(true_pve+true_nve)/predictions_t,2)
    pre = round(1.0*true_pve/(true_pve + false_pve),2)
    recall = round(1.0*true_pve/(true_pve+false_nve),2)
    return acc,pre,recall



print features_list
print "#################################"
print "processing ....."
for p in range(len(features[0])):
    t0 = time()
    select = SelectKBest(f_classif, k=p+1)
    select.fit(features,labels)
    min_features = select.fit_transform(features,labels)
    drop = np.sort(select.scores_)[::-1][p]
    sfl = [a for b, a in enumerate(features_list[1:])if select.scores_[b] >= drop]
    sfl = ['poi'] + sfl
    rc = RandomForestClassifier(random_state =1200)
    aB = AdaBoostClassifier(random_state=1200)
    acc,pre,recall = evaluation(rc,min_features,labels,cvv)
    random_acc.append(acc)
    random_pre.append(pre)
    random_recall.append(recall)
    acc,pre,recall = evaluation(aB,min_features,labels,cvv)
    ad_acc.append(acc)
    ad_pre.append(pre)
    ad_recall.append(recall)
    print "fitting time for k = {0} : {1}".format(p+1, round(time()-t0,3))
    print "Random Forest accuracy:{0} precision:{1} recall:{2}".format(random_acc[-1],random_pre[-1],random_recall[-1])
    print "ADABOOST accuracy:{0} precision:{1} recall:{2}".format(ad_acc[-1],ad_pre[-1],ad_recall[-1])


randomforest_df = pd.DataFrame({'RandomForest_accuracy':random_acc,'RandomForest_precision':random_pre,'RandomForest_recall':random_recall})    
adaboost_df = pd.DataFrame({'AdaBoost_accuracy':ad_acc,'AdaBoost_precision':ad_pre,'AdaBoost_recall':ad_recall})


randomforest_df.plot()
plt.show()

adaboost_df.plot()
plt.show()




# adaboost having high recall and random forest having high precision


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html





# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
#labels_train = np.reshape(labels_train,(99,1))

sk_best = SelectKBest(f_classif, k = random_recall.index(max(random_recall))+1)
sk_best.fit(features,labels)
borders = np.sort(sk_best.scores_)[::-1][random_recall.index(max(random_recall))+1]
sfl = [i for j,i in enumerate(features_list[1:]) if sk_best.scores_[j] > borders]
sfl= ['poi']+sfl

#sfl-> selecting some of the best features from the list and adding poi
# Now the selected features will be
sf=sk_best.fit_transform(features,labels)


print "@@@@@@@@@@@@@@"
print len(sfl)-1

for i in sfl[1:]:
    print i

random_forest = RandomForestClassifier(random_state=1200)
random_forest.fit(sf,labels)
print "@@@@@@@@@using SelectKBest@@@@@@@@@"
print random_forest.feature_importances_
for i in  sfl[1:]:
    print i,"score:",sk_best.scores_[features_list[1:].index(i)]

print "!!!!!!!!!!!!!!!!!"
print len(sf)
print len(labels)


# tunning the classifiers
print "please wait classifiers are tunning...."

#test_classifier is used below which is imported from test.py


t0 =time()
tunning = {'n_estimators':[1,50,100],'min_samples_split':[2,8],'max_features':[2,3]}

rf_clf = GridSearchCV(RandomForestClassifier(random_state=42),tunning,cv= cvv,scoring = 'recall')
rf_clf.fit(sf,labels)
print (rf_clf.best_params_)
print "time:{0}".format(round(time()-t0,3))
random_clf = rf_clf.best_estimator_
test_classifier(random_clf,my_dataset,sfl,folds = 1000)
print "Now, It is tunned with the random classifier and let tune the ADABOSST clf:"




t0 =time()
tunning = {'n_estimators':[5,40,80],'learning_rate':[0.2,0.5,1]}
print("#tunning hyper-parameters for recall")


Ada_clf = GridSearchCV(AdaBoostClassifier(random_state=42),tunning,cv=cvv,scoring='recall')
Ada_clf.fit(sf,labels)
print (Ada_clf.best_params_)
print "time:{0}".format(round(time()-t0,3))

print "wait........."

Adaboost_classifier=Ada_clf.best_estimator_
test_classifier(Adaboost_classifier,my_dataset,sfl,folds=1000)

print "It's Done!!!"


pickle.dump(rf_clf,open("my_classifier.pkl","w"))
pickle.dump(data_dict,open("my_dataset.pkl","w"))
pickle.dump(features_list,open("my_features_list.pkl","w"))

#exported pkl files for further classification.
clf=rf_clf.best_estimator_
features_list=sfl
                  







### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
