
import config
from data_operations.data_preprocesing import  data_imputing_categorical, data_imputing_numerical,  encode_features, read_data, split_data
from models.model import evaluate_model, train_model


def main():
    
    if config.run_mode == "training":
         
         print("Training model\n")
         
         if config.dataset == "uci_credit_approval":
            
            
            data= read_data(config.filepath)
            x_train,y_train, x_test,  y_test = split_data(data)
            
            x_train, y_train, x_test, y_test = data_imputing_categorical(x_train, y_train ,x_test, y_test)
            x_train, y_train, x_test, y_test = data_imputing_numerical(x_train,y_train, x_test,  y_test)
            
            
            x_train, x_test = encode_features(x_train, x_test)
            
        
            print(f"x_train shape: {x_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            print(f"x_test shape: {x_test.shape}")
            print(f"y_test shape: {y_test.shape}")
            
            model = train_model(x_train, y_train, x_test, y_test)
            #accuracy, report = evaluate_model(model, x_test, y_test)
            #print(f"Random Forest Model Accuracy: {accuracy}")
            #print(f"Random Forest Model Report:\n{report}")
            
          
            
            
         
if __name__ == '__main__':
    main()

        
        
        