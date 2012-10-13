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
        raise Exception("Unknown sex")

def sex_code(data):
    return pd.DataFrame.from_dict({"sex_code": data["sex"].apply(sex_code_func)})