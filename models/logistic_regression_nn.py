import copy
from data_operations.helper_functions import sigmoid
import numpy as np


class LogisticRegression():
    
    def propagate(self, w, b, X, Y):
      
        
        m = X.shape[1]
        z = np.dot(w.T, X) + b
        A = sigmoid(z)
        
        #cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        
        epsilon = 0.00001
        cost = - (1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
        
        dz = A - Y
        dw = (1 / m) * np.dot(X, dz.T)
        db = (1 / m) * np.sum(dz)
        
        
        cost = np.squeeze(np.array(cost))

        grads = {"dw": dw,
                "db": db}
        
        return grads, cost
    
    
   
    def optimize(self, w, b, X, Y, num_iterations=1000, learning_rate=0.1, print_cost=False):
       
    
        w = copy.deepcopy(w)
        b = copy.deepcopy(b)
        
        costs = []
        
        for i in range(num_iterations):
          
            grads, cost = self.propagate(w, b, X, Y)
            
           
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            
            
            w = w - learning_rate * dw
            b = b - learning_rate * db
            
            
            if i % 100 == 0:
                costs.append(cost)
            
              
                if print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        
        params = {"w": w,
                "b": b}
        
        grads = {"dw": dw,
                "db": db}
        
        return params, grads, costs
    
    
    
    def predict(self, w, b, X):
   
        
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)
        
        A = sigmoid(np.dot(w.T, X)+b)
        
       
        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                Y_prediction[0 ,i] = 1
            else:
                Y_prediction[0,i] = 0
            
    
        return Y_prediction
    
    
    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
        
        
        w, b = np.zeros((X_train.shape[0], 1)), 0.0
        
        params, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        w = params["w"]
        b = params["b"]
        
        Y_prediction_train = self.predict(w, b, X_train)
        Y_prediction_test = self.predict(w, b, X_test)
        
      
        if print_cost:
            print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
            print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        
        d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations}
        
        return d
    
    
    