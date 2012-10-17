#/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 02:57:24 2012

@author: tsundokum
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from extractfeatures import extract_features, get_dataframe

GENERATE_SUBMISSION = True
K_FOLD = 10

N_ESTIMATORS = 100
MAX_DEPTH = 11

OUT_PATH = '../results/'

feature_names = [
"pclass",
"age",
#"sibsp",
#"parch",
"fare",
"embarked_code",
"sex_code", #sex is replaced by sex_code
"ticket_number", # cleaned-up number of the ticket
]

raw_data = get_dataframe("train.csv")
data = extract_features(feature_names, raw_data)
Y = extract_features(["survived"], raw_data).values.T[0]

skf = StratifiedKFold(Y, K_FOLD)
sum_score = 0
for train, cv in skf:
#train, cv = skf.__iter__().next()

    train_x, cv_x = data.take(train), data.take(cv)
    train_y, cv_y = Y.take(train), Y.take(cv)
    
    classifier = RandomForestClassifier(n_estimators=N_ESTIMATORS, oob_score=True, verbose=1, max_depth=MAX_DEPTH, compute_importances=True)
    classifier.fit(train_x, train_y)
    
    predicted_cv = classifier.predict(cv_x)
    
    score = sum(predicted_cv==cv_y)/float(cv_y.shape[0])
    sum_score += score
    print "score on cross-validation is %.3f" % score
    print
    print "Feature importances:"
    for i, feature in enumerate(feature_names):
        print "%s \t %.4f" % (feature, classifier.feature_importances_[i])
    print    
    
print "="*40    
print "average score in %0.4f" % (sum_score/K_FOLD)    
print "="*40    
    
    
if GENERATE_SUBMISSION:
    raw_data = get_dataframe("train.csv")
    data = extract_features(feature_names, raw_data)
    Y = extract_features(["survived"], raw_data).values.T[0]    
    
    classifier = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)
    classifier.fit(data, Y)
    
    test_raw_data = get_dataframe("test.csv")    
    test_data = extract_features(feature_names, test_raw_data)
    
    prediction = classifier.predict(test_data)
    
    from datetime import datetime
    np.savetxt(OUT_PATH + "submission_"+ 
           datetime.now().strftime("%Y%m%d_%H%M") + "_"  # date and time
           + str(int(sum_score*10000/K_FOLD))            # average score (0.8181 -> 8181)
           +".csv", prediction, fmt="%d")

