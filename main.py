
import config
from data_operations.data_preprocesing import data_imputing, dataset_split, encode_features, read_data
from models.model import evaluate_model, train_model


def main():
    
    if config.run_mode == "training":
         
         print("Training model\n")
         
         if config.dataset == "uci_credit_approval":
            
            filepath = "data/uci_credit_approval.csv"
            _, X, y = read_data(filepath)
            
            X_train, X_test, y_train, y_test = dataset_split(X, y, split=0.2)
            X_train, X_test, y_train, y_test = data_imputing(X_train, X_test, y_train, y_test)
            X_train, X_test = encode_features(X_train, X_test)
            
            # Train Random Forest model
            model = train_model(X_train, y_train)
            accuracy, report = evaluate_model(model, X_test, y_test)
            print(f"Random Forest Model Accuracy: {accuracy}")
            print(f"Random Forest Model Report:\n{report}")
            
          
            
            
         
if __name__ == '__main__':
    main()

        
        
        