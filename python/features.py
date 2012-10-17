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

def ticket_number_func(s_ticket):
    """ Extracts actual number of the ticket, skipping the optional string prefix.
        Examples:
        >>> extract_digits("C.A./SOTON 34068")
        34068
        >>> extract_digits("34568")
        36568
    """
    ends_with = s_ticket.split()[-1]
    if ends_with.isdigit():
        return int(ends_with)
    else:
        return 0

def ticket_number(data):
    df = pd.DataFrame.from_dict({"ticket_number": data["ticket"].apply(ticket_number_func)})
    # replace missing values with the mean 
    mean = df['ticket_number'].mean()
    df = df['ticket_number'].apply(lambda i: i if i!=0 else mean)    
    return df
