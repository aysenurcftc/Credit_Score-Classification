import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import KBinsDiscretizer



def read_data(datadir):
    
    data=pd.read_csv(datadir, low_memory=False)
    return data
  
  

def split_data(data):
    
    if config.dataset == "uci_credit_approval":
        X = data.drop('A16', axis=1)
        y = data[['A16']]
        
    elif config.dataset == "statlog_german_credit_data":
        X = data.drop('class', axis=1)
        y = data[['class']]
        
    elif config.dataset == "loan":
        X = data.drop(columns=["loan_status"], axis=1)
        y = data["loan_status"].values
        
    elif config.dataset == "statlog_australian_credit_approval":
        X=data.drop('A15', axis=1)
        y=data[['A15']].values
        
    elif config.dataset == "default_of_credit_card_clients":
        X=data.drop('Y', axis=1)
        y=data[['Y']].values
        
   
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1-config.split_ratio, random_state=42)
    
    if config.dataset == "statlog_german_credit_data":
       y_train = (y_train == 2).astype(int)
       y_test = (y_test == 2).astype(int)
    
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
    
    
    
    
def label_encoding(data):
    
    for col in data.select_dtypes(object):
        data[col] = LabelEncoder().fit_transform(data[col])
        
    return data


def min_max_scaler(X_train, X_test):
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)
    return X_train, X_test

def encode_features(X_train, X_test):
    
    cat_vars = X_train.select_dtypes(exclude=[np.float64, np.int64]).columns
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoder.fit(X_train[cat_vars])

    X_train_encoded = encoder.transform(X_train[cat_vars])
    X_test_encoded = encoder.transform(X_test[cat_vars])

    X_train = X_train.drop(cat_vars, axis=1).reset_index(drop=True)
    X_test = X_test.drop(cat_vars, axis=1).reset_index(drop=True)

    X_train = pd.concat([X_train, pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(cat_vars))], axis=1)
    X_test = pd.concat([X_test, pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(cat_vars))], axis=1)
    
    
    return X_train, X_test
    
    
    
def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled



def uniform_discretization_transform(data):
    if config.dataset == "statlog_australian_credit_approval":
        columns_to_discretize = ['A1', 'A2', 'A3', 'A7', 'A10', 'A13', 'A14']
        data_to_discretize = data[columns_to_discretize]

        # Perform a uniform discretization transform
        trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        data_dis = trans.fit_transform(data_to_discretize)

        data_dis = pd.DataFrame(data_dis, columns=columns_to_discretize)
        data[columns_to_discretize] = data_dis
        
        
        return data
    
    

