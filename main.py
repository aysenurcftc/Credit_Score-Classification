import config
from data_operations.data_preprocesing import  data_imputing_categorical, data_imputing_numerical,   encode_features, label_encoding, min_max_scaler, read_data, split_data, standardize_features, uniform_discretization_transform
from models.logistic_regression_nn import LogisticRegression
from models.model import train_model
from models.nn_model import NnModel
from sklearn.metrics import accuracy_score
import numpy as np


def main():
    
    if config.run_mode == "training":
         
         print("Training model\n")
         
         if config.dataset == "uci_credit_approval":
            data= read_data(config.filepath)
            x_train,y_train, x_test,  y_test = split_data(data)
            x_train, y_train, x_test, y_test = data_imputing_categorical(x_train, y_train ,x_test, y_test)
            x_train, y_train, x_test, y_test = data_imputing_numerical(x_train,y_train, x_test,  y_test)
            
            x_train, x_test = encode_features(x_train, x_test)
            x_train, x_test = standardize_features(x_train, x_test)
            
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            print(counts_train)
            print(counts_test)
            
            
            model = train_model(x_train, y_train.values, x_test, y_test.values)
            
            

         
         elif config.dataset == "statlog_german_credit_data":
            data = read_data(config.filepath)
            x_train, y_train, x_test, y_test = split_data(data)

            x_train, x_test = encode_features(x_train, x_test)
            
            x_train, x_test = standardize_features(x_train, x_test)
            
            
            
            model = train_model(x_train, y_train.values, x_test, y_test.values)
            
            
            
         elif config.dataset == "loan":
            data = read_data(config.filepath)
            data = label_encoding(data)
            x_train, y_train, x_test, y_test = split_data(data)
            
            x_train, x_test = min_max_scaler(x_train,x_test)
            
            model = train_model(x_train, y_train, x_test, y_test)
            
            
           
            
         elif config.dataset == "statlog_australian_credit_approval":
            data = read_data(config.filepath)
            data = uniform_discretization_transform(data)
            
            x_train, y_train, x_test, y_test = split_data(data)
            
            
         
            #print(x_train.head())
            x_train, x_test = standardize_features(x_train, x_test)
            
            model = train_model(x_train, y_train, x_test, y_test)
            
         elif config.dataset == "default_of_credit_card_clients":
            
            data = read_data(config.filepath)
   
            x_train, y_train, x_test, y_test = split_data(data)
            
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            
   
            
            x_train, x_test = standardize_features(x_train, x_test)
            model = train_model(x_train, y_train, x_test, y_test)
            
            
            
            
         else:
            raise ValueError("Unsupported dataset")
         
            
         #evaluate model
         if config.model_type == "NN":
            nn_model = NnModel()
            y_pred = nn_model.predict(model, x_test.T)
            
            if config.dataset == "loan" or config.dataset == "statlog_australian_credit_approval" or config.dataset == "default_of_credit_card_clients":
               accuracy = accuracy_score(y_test.reshape(-1), y_pred.reshape(-1))
         
            else:
               accuracy = accuracy_score(y_test.values.reshape(-1), y_pred.reshape(-1))
            print(f"Test Accuracy: {accuracy}")
         
         
            
         
         
if __name__ == '__main__':
    main()

        
        
        