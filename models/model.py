
import config
from sklearn.metrics import classification_report, accuracy_score
from models.logistic_regression_nn import LogisticRegression
import numpy as np



def train_model(X_train, y_train, x_test, y_test):
    
    """Train a classification model."""
    
    y_train = y_train.values.reshape(1, -1)
    y_test = y_test.values.reshape(1, -1)
    
    if config.model_type == 'logistic regression':
        model = LogisticRegression()
        trained_model = model.model(X_train.T, y_train, x_test.T, y_test,num_iterations=4000, learning_rate=0.01, print_cost=True) 
    else:
        raise ValueError("Unsupported model type")
    
    return trained_model



def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


