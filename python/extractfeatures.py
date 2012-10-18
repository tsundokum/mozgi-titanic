#/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 23:51:07 2012

Idea (and code) is taken from
https://github.com/benhamner/Stack-Overflow-Competition
"""

import pandas as pd
import os
import features
from collections import Counter

DATA_PATH = "../data/"


def get_dataframe(file_name="train.csv"):
    return pd.io.parsers.read_csv(os.path.join(DATA_PATH, file_name))


def extract_features(feature_names, data):
    fea = pd.DataFrame(index=data.index)
    for name in feature_names:
        if name in data:
            fea = fea.join(data[name])
        else:
            #load function from features.py and apply it
            fea = fea.join(getattr(features, name)(data))
    return fea


if __name__ == "__main__":

    feature_names = [
    "survived",
    "pclass",
    "name",
    "age",
    "sibsp",
    "parch",
    "ticket",
    "fare",
    "cabin",
    "embarked_code",  # embarked is replaced by embarked_code
    "sex_code",  # sex is replaced by sex_code
    "ticket_number",  # cleaned-up number of the ticket
    "cabin_code",  # cabin is replaced with cabin_code
    "title"
    ]

    raw_data = get_dataframe("train.csv")
    data = extract_features(feature_names, raw_data)

    #manipulate data as you want
    print Counter(data["title"])
