#/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 01:28:08 2012
"""

import pandas as pd

INITIAL_FEATURES = [
"survived",
"pclass",
"name",
"sex",
"age",
"sibsp",
"parch",
"ticket",
"fare",
"cabin",
"embarked"]

######################################################################
# Add your feature extracting functions here
######################################################################

def sex_code_func(s_sex):
    if s_sex=="male":
        return 1
    elif s_sex=="female":
        return 0
    else:
        raise Exception("Unknown sex value (%s)" %s_sex)

def sex_code(data):
    return pd.DataFrame.from_dict({"sex_code": data["sex"].apply(sex_code_func)})
    
def embarked_code_func(s_embarked):
    if s_embarked=="C":
        return 0
    elif s_embarked=="S":
        return 1
    elif s_embarked=="Q":
        return 2
    elif pd.isnull(s_embarked):
        return 3 #if nan
    else:
        raise Exception("Unknown embarked value (%s)" %s_embarked)
    
def embarked_code(data):
    return pd.DataFrame.from_dict({"embarked_code": data["embarked"].apply(embarked_code_func)})