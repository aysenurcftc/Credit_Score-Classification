import copy
from data_operations.helper_functions import sigmoid
import numpy as np
from sklearn.metrics import classification_report


class NnModel():
    
    
    def layer_sizes(self, X, Y):
      
        n_x = X.shape[0]
        n_h = 4
        n_y = Y.shape[0]
      
        return (n_x, n_h, n_y)
    
    
    
    def initialize_parameters(self, n_x, n_h, n_y):
      
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters


    def forward_propagation(self, X, parameters):
    
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
       
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        
     
        
        assert(A2.shape == (1, X.shape[1]))
        
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        
        return A2, cache
    
    
    def compute_cost(self, A2, Y):
       
       
        Y = np.array(Y)
        
        m = Y.shape[1] 
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        cost = - np.sum(logprobs) / m  
        
        cost = float(np.squeeze(cost))  
                                       
        
        return cost
    
    def backward_propagation(self, parameters, cache, X, Y):
      
        m = X.shape[1]
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

       
        A1 = cache['A1']
        A2 = cache['A2']
        
       
        dZ2 = A2 -Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        
         
        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads
    
    
    def update_parameters(self, parameters, grads, learning_rate = 0.01):
      
        
        W1 = copy.deepcopy(parameters["W1"])
        b1 = copy.deepcopy(parameters["b1"])
        W2 = copy.deepcopy(parameters["W2"])
        b2 = copy.deepcopy(parameters["b2"])
        
       
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        
      
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        
         
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters
        
    
    def nn_model(self, X, Y, n_h, num_iterations = 10000, print_cost=False):
       
        
        #np.random.seed(5)
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]
        
       
        parameters = self.initialize_parameters(X.shape[0], n_h, Y.shape[0])
        
        for i in range(0, num_iterations):
            
          
            A2, cache = self.forward_propagation(X, parameters)
            cost = self.compute_cost(A2, Y)
            grads = self.backward_propagation(parameters, cache, X, Y)
            parameters = self.update_parameters(parameters, grads)
            
           
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                
                predictions = self.predict(parameters, X)
                print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

                
        return parameters
    
    
    
    def evaluate_model(self, parameters, X_train, Y_train, X_test, Y_test):
        Y_pred_train = self.predict(parameters, X_train)
        Y_pred_test = self.predict(parameters, X_test)
        
        train_report = classification_report(Y_train.T.flatten(), Y_pred_train.flatten(), zero_division=0)
        test_report = classification_report(Y_test.T.flatten(), Y_pred_test.flatten(), zero_division=0)
        
        print("Train classification report:")
        print(train_report)
        
        print("Test classification report:")
        print(test_report)

        return {"train_report": train_report, "test_report": test_report}
    
    
    
    def predict(self, parameters, X):
       
        A2, cache =  self.forward_propagation(X, parameters)
        predictions = (A2 > 0.5).astype(int)
       
        return predictions
    
    
    
    