import config
from sklearn.metrics import classification_report, accuracy_score
from models.logistic_regression_nn import LogisticRegression
from models.nn_model import NnModel
import numpy as np



def train_model(X_train, y_train, x_test, y_test):

    y_train = y_train.values.reshape(1, -1)
    y_test = y_test.values.reshape(1, -1)
 

    if config.model_type == 'logistic regression':
        model = LogisticRegression()
        trained_model = model.model(X_train.T, y_train, x_test.T, y_test,num_iterations=4000, learning_rate=0.1, print_cost=True) 
       
        
    elif config.model_type == "NN":
        model = NnModel()
       
        trained_model = model.nn_model(X_train.T, y_train, n_h = 3, num_iterations = 10000, print_cost=True)
        
       
    else:
        raise ValueError("Unsupported model type")
    
    return trained_model


