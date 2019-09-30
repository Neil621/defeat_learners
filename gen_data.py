"""  		   	  			  	 		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: nwatt3 (replace with your User ID)  		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903476861 (replace with your GT ID)  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import math  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		   	  			  	 		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		   	  			  	 		  		  		    


#Create a Python program called gen_data.py that implements two functions. The two functions should be named as follows, and support the following API:
#X1, Y1 = best4LinReg(seed = 5)
#X2, Y2 = best4DT(seed = 5)

def best4LinReg(seed=5):  		   	  			  	 		  		  		    	 		 		   		 		  
    np.random.seed(seed)  		   	  			  	 		  		  		    	 		 		   		 		  
    #X = np.zeros((100,2))  		   	  			  	 		  		  		    	 		 		   
    #Y = np.random.random(size = (100,))*200-100  		   	  			  	 		  		  		    	 		 		   		 		  
    # Here's is an example of creating a Y from randomly generated  		   	  			  	 		  		  		    	 		 		   		 		  
    # X with multiple columns  		   	  			  	 		  		  		    	 		 		   		 		  
    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3  		   	  			  	 		  		  	
    
    p = np.random.randint(2, 10+1)
    n = np.random.randint(10, 1000+1)
    bias = np.random.randn(1)
    
    X = 1000 * np.random.randn(n, p) + np.random.randint(1, 10)
    Y = np.dot(X, np.random.randn(p)) + bias 
    
    
    
    return X, Y  		   	  			  	 		  		  		    	 		 		   		 	





  		   	  			  	 		  		  		    	 		 		   		 		  
#def best4DT(seed=5):  		   	  			  	 		  		  		    	 		 		   		 		  
#    np.random.seed(seed)  		   	  			  	 		  		  		    	 		 		   		 		  
#    X = np.zeros((100,2))  		   	  			  	 		  		  		    	 		 		   		 		  
#    Y = np.random.random(size = (100,))*200-100  		   	  			  	 		  		  		    	 		 		   		 		  
 #   return X, Y  		   	  			  	 		  		  		    	 		 		   		 	


def best4DT(seed=5):
    np.random.seed(seed)
    p = np.random.randint(2, 10+1)
    n = np.random.randint(10, 1000+1)
    X1 = np.random.randn(n, p) + np.random.randint(0, 10)
    Y1 = np.random.randn(n) + 2
    X2 = np.random.randn(n, p) - np.random.randint(15, 20)
    Y2 = np.random.randn(n) - 20
    X3 = np.random.randn(n, p) - np.random.randint(30, 35)
    Y3 = np.random.randn(n) + 100
    X = np.vstack((X1, X2, X3))
    Y = np.vstack((Y1, Y2, Y3)).reshape(-1)
    return X, Y


  		   	  			  	 		  		  		    	 		 		   		 		  
def author():  		   	  			  	 		  		  		    	 		 		   		 		  
    return 'nwatt3' #Change this to your user ID  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    print("they call me Tim.")  		   	  			  	 		  		  		    	 		 		   		 		  
