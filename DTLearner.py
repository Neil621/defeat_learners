"""  		   	  			  	 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		   	  			  	 		  		  		    	 		 		   		 		  
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
import warnings  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class DTLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    #def __init__(self, leaf_size=1, verbose = False):  		   	  			  	 		  		  		    	 		 		   		 		  
      #  warnings.warn("\n\n  WARNING! THIS IS NOT A CORRECT DTLearner IMPLEMENTATION! REPLACE WITH YOUR OWN CODE\n")  		   	  			  	 		  		  		    	 		 		   		 		  
     #   pass # move along, these aren't the drones you're looking for  		   	  			  	 		  		  		    	 		 		   		 		  
    
class DTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
        if leaf_size < 1:
            raise Exception("Must have at least one leaf. It's a tree!")
        self.leaf_size = leaf_size


    def author(self):
        return "nwatt3"

    
    
    
    
    def train_tree(self, Xtrain, Ytrain):


        if np.unique(Ytrain).shape[0] == 1:
            
            #if the data has only one yvalue then this is returned as 1 leaf
            
            return np.asarray([[-1, Ytrain[0], np.nan, np.nan]])
        
        
        if self.leaf_size >= Xtrain.shape[0]:
            
            
            #if there is less than leaf size , still return one leaf as lead size has to be 1 at least
            return np.asarray([[-1, np.mean(Ytrain), np.nan, np.nan]])
        
        

        # identify best feature to split based on correlation with target Ytrain
        correlation_array = []
        for i in range(Xtrain.shape[1]):
            
            
            variance_matrix = np.var(Xtrain[:, i])
            correlation_matrix = np.corrcoef(Xtrain[:, i], Ytrain)[0, 1] if variance_matrix > 0 else 0
            correlation_array.append(correlation_matrix)
        bestFeature = np.argsort(correlation_matrix)[::-1][0]

        # split threshold for splitting
        #I'm basing this on the median of best feature
        split_threshold = np.median(Xtrain[:, bestFeature])
        
        
        #left tree defined as less than or equal to the threshold
        left_values = Xtrain[:, bestFeature] <= split_threshold

        if np.median(Xtrain[left_values, bestFeature]) == split_threshold:
            
            
            # only makes sense to split on a feature if x actually varies
            
            return np.asarray([[-1, np.mean(Ytrain), np.nan, np.nan]])

        
        
        # train right tree, this is just the opposite of left. so used ~
        RHS_tree = self.train_tree(Xtrain[~left_values], Ytrain[~left_values])
        
        
        # train tree left
        LHS_tree = self.train_tree(Xtrain[left_values], Ytrain[left_values])

        #root_node node of the tree
        root_node = np.asarray([[bestFeature, split_threshold, 1, LHS_tree.shape[0]+1]])

        # create the tree with root, LHS and right
        return np.vstack((root_node, LHS_tree, RHS_tree))


    def query_tree(self, tree, Xtrain):
        root_node = tree[0]
        if int(root_node[0]) == -1:
            #return leaf value
            return root_node[1]
        elif Xtrain[int(root_node[0])] <= root_node[1]:
            # left tree
            LHS_tree = tree[int(root_node[2]):,:]
            return self.query_tree(LHS_tree, Xtrain)
        else:
            # go right if value is neither equal to or less than split value
            RHS_tree = tree[int(root_node[3]):,:]
            return self.query_tree(RHS_tree, Xtrain)


    def addEvidence(self, Xtrain, Ytrain):

        #add x train and y train data to model
        
        self.tree = self.train_tree(Xtrain, Ytrain)


    def query(self, data):

        
        Y = []
        for X in data:
            Y.append(self.query_tree(self.tree, X))
        return np.asarray(Y)


