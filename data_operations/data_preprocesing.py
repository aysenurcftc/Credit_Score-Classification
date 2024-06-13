import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def read_data(datadir):
    
    data=pd.read_csv(datadir)
    return data
  
  
  
"""   

def split_data(data):
    

    split = int(len(data) * config.split_ratio)
    
    train_data = data[:split]
    test_data = data[split:]
    
    if config.dataset == "uci_credit_approval":
        x_train = train_data.drop('A16', axis=1)
        y_train = train_data[['A16']]
        x_test = test_data.drop('A16', axis=1)
        y_test = test_data[['A16']]
    
    return x_train, y_train, x_test, y_test   
"""  


def split_data(data):
    if config.dataset == "uci_credit_approval":
        X = data.drop('A16', axis=1)
        y = data[['A16']]
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - config.split_ratio, random_state=42)
    
    return x_train, y_train, x_test, y_test
        
   

def data_imputing_numerical(X_train, y_train, X_test, y_test):
    num_var = X_train.select_dtypes(include=[np.float64, np.int64]).columns
    
    if config.imputing_type_numerical == "median":
        for var in num_var:
            deger=X_train[var].median()
            X_train[var].fillna(deger, inplace=True )
            X_test[var].fillna(deger, inplace=True)
        
    elif config.imputing_type_numerical == "mean":
        for var in num_var:
            deger=X_train[var].mean()
            X_train[var].fillna(deger, inplace=True )
            X_test[var].fillna(deger, inplace=True)
        
    return X_train, y_train, X_test, y_test

def data_imputing_categorical(X_train, y_train, X_test, y_test):
    cat_val = X_train.select_dtypes(exclude=[np.float64, np.int64]).columns
    
    if config.imputing_type_categorical == "mode":
        for var in cat_val:
            deger=X_train[var].mode()[0]
            X_train[var].fillna(deger, inplace=True )
            X_test[var].fillna(deger, inplace=True)
      
    if config.dataset == "uci_credit_approval":
        y_train.replace(to_replace={'+':1, '-':0}, inplace=True)
        y_test.replace(to_replace={'+':1, '-':0}, inplace=True)

    return X_train, y_train, X_test, y_test
    
    

def encode_features(X_train, X_test):
    
    cat_vars = X_train.select_dtypes(exclude=[np.float64, np.int64]).columns
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoder.fit(X_train[cat_vars])

    X_train_encoded = encoder.transform(X_train[cat_vars])
    X_test_encoded = encoder.transform(X_test[cat_vars])

    # Drop original categorical columns and concatenate encoded features
    X_train = X_train.drop(cat_vars, axis=1).reset_index(drop=True)
    X_test = X_test.drop(cat_vars, axis=1).reset_index(drop=True)

    X_train = pd.concat([X_train, pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(cat_vars))], axis=1)
    X_test = pd.concat([X_test, pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(cat_vars))], axis=1)

    return X_train, X_test
    
    
    





