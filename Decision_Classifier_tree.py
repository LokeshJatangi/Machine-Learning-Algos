import math
import numpy as np

# Creating a class to store the Learnings from data in terms of decision node.
class DecisionNode():
    """
    Class represents either a Leaf node or Internal decision node of tree
    
    :param threshold       (float)          : Threshold value comparing against data
    :param attribute_index (int)            : Column on which data is being split
    :param value           (float)          : Class prediction
    :param left_child      (DecisionNode)  : Decision node when condition on threshold is true
    :param right_child     (DecisionNode)  : Decision node when condition on threshold is false
    """
    
    def __init__(self,threshold = None,attribute_index = None,value = None,left_child=None,right_child=None):
       
        self.threshold = threshold                  # Threshold for Best split
        self.attribute_index = attribute_index      # Column 
        self.value = value                          # Leaf value
        self.left_child = left_child                # Left subtree
        self.right_child = right_child              # Right subtree




class DecisionTree():
    """
    Class represnting the Classification decision tree
    """
    
    
    tree = {}
    
    def __init__(self,min_samples_split = 5):
   
        self.min_samples_split  = min_samples_split
        self.root =None # to build tree

    def _entropy(self,target_attribute,entropy = 0.0):
        """
         Entropy of feature is calculated based on the
        Target_attribute for its corresponding rows/instance/data
        
        :param target_attribute (array) : Label array (Y)
        :return entropy (float)
        """
        
        # Find no of classes present in Target_attribute
        unique_labels = np.unique(target_attribute)
        
        for label in unique_labels:
            
            # Counts no of 1's in the bool array for each label
            count = np.sum(target_attribute == label) 
            p = count / len(target_attribute) 
            
            entropy += -p * math.log(p,2)
            
        return entropy
    
        
    
    def _impurity_after_split(self,parent,left_child,right_child,
                                impurity_children=0.0):
        """
        Assuming univariate binary split, input is corresponding target attribute values
        
        :param  parent_node(array)  : data before splitting
        :param left_child (array)  : array based on splitting decision
        :param right_child (array) : array based on splitting decision
        :return impurity_children (float)  : weighted average of entropy of children nodes            
        """
        p1 = len(left_child)/len(parent)
        impurity_children =   ( p1*(self._entropy(left_child)) \
                                 + (1-p1)*(self._entropy(right_child))  ) #p2 = 1-p1 becuase its binary split
        
        return impurity_children
        
    
    
    
    def _information_gain(self,parent_node,left_child,right_child, info_gain=0.0):
        """
        Calculate Information gain after splitting based on condition
        
        :param  parent_node (array) : data before splitting
        :param left_child (array)  : array based on splitting decision
        :param right_child (array) : array based on splitting decision 
        :return info_gain (float) : Information gain after splitting
        
        """
        
        entropy_parent_node = self._entropy(parent_node)
        Impurity_after_split = self._impurity_after_split(parent_node,left_child,right_child)
        info_gain = entropy_parent_node - Impurity_after_split
       
        return info_gain
    
    
    def _split_data(self,XY,feature_index,threshold):
        """
        Splits the combined data into two parts based threshold comparison 
        which in turn forms left subtree and right subtree
        
        :param feature_index (int) 
        :param threshold     (float)
        :return              (tuple)  
        """
        
        if isinstance(threshold,int) or isinstance(threshold,float):
            
            # Extract data from XY where bool condn satisfies
            xy_1 = XY[ XY[:,feature_index] <= threshold ] # similar to np.extract
            xy_2 = XY[ XY[:,feature_index] > threshold  ]
            
        
        return xy_1 , xy_2
    
    
    def _majority_vote(self,target_attr):
        
        """
        At leaf node the leaf should represent a class.
        The class of leaf is determined by majority_vote of data grouping 
        """
        
        majority = None
        max_count = 0
        
        # Find no of classes present in Target_attribute
        unique_labels = np.unique(target_attr)
        
        for label in unique_labels:
            count = np.sum(target_attr == label) 
            if count > max_count :
                majority = label
                max_count = count
            
        return majority
    
    def _build_tree(self , data ,target_attribute):
        """
        Its the main function and the one which gave headache.
        
        Recursively calls the left, right subtree and base condition is a Leaf node
    
        
        """
        
        Max_info_gain = 0.0
    
        # Combine target with data 
        
        if len(np.shape(target_attribute)) == 1: # if its single column
            #expand it to matrix for concatenation
            target_attribute = np.expand_dims(target_attribute,axis =1) 
            
        combined_data = np.concatenate((data,target_attribute),axis = 1)
        n_samples , n_attributes = np.shape(data)
        
        
        # Finding best-split data
        
        # Avoid overfitting by early stopping for min_samples
        if n_samples >= self.min_samples_split :
            
            #columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","Alcohol"]
            # Impurity_after_split calculation for all features
            for attribute_i in range(n_attributes):
               # print("---------------------------------------")
               # print("Attribute : "+str(columns[attribute_i]))
                # Select all rows/data of feature column and create unique values array  
                unique_values = np.unique(data[:, attribute_i]) 
                
                # Calcualte impurity w.r.t unique values for each feature
                for threshold in unique_values :
                    
                   # print("Threshold value : "+str(threshold))

                    # Calculate data_1 and data_2 after the split based on threshold 
                    combined_data_1 , combined_data_2 = self._split_data(combined_data,attribute_i,threshold)
                    
                    
                    # calculate target_1 and target_2 corresponding to data_1 and data_2 for entropy
                    if len(combined_data_1) > 0 and len(combined_data_2) > 0 :
                        
                        
                        # Select all rows of target_attribute for entropy calculation
                        target_1 = combined_data_1[:,-1] # remember shape is 1D (x,)
                        # Calculate information_gain 
                        info_gain = self._information_gain(target_attribute,target_1,target_2)
                        
                        # Save threshold value , attribute_index  and branches
                        if info_gain > Max_info_gain :
                            Max_info_gain = info_gain
                            best_split = [attribute_i,threshold]
                            # Conditions are true at decision node
                            left_x = combined_data_1[:,:-1]
                            left_y = combined_data_1[:,-1]
                            #Conditons are false
                            right_x = combined_data_2[:,:-1]
                            right_y = combined_data_2[:,-1]
                        #print("Info_gain : "+ str(info_gain) + ", Max_info_gain : "+str(Max_info_gain))


        if Max_info_gain > 1e-5 :
            # Internal node
            # Recursively building tree
            left_subtree = self._build_tree(left_x,left_y) 

            right_subtree = self._build_tree(right_x,right_y)
            return DecisionNode(attribute_index = best_split[0],threshold = best_split[1],left_child = left_subtree,right_child = right_subtree)
        
        leaf_value = self._majority_vote(target_attribute)
        return DecisionNode(value = leaf_value)

    def learn(self, X,Y):
        
        """
        Tree learnt from data starting with root
        """
        # implement this function
        self.root = self._build_tree(X,Y)

    # implement this function
    def classify(self, test_instance,tree = None):
        """
        Recursively going down the tree till a leaf is found
        """

        # Start from root
        if tree is None :
            tree = self.root
        
        # At leaf ( Base condition)
        if tree.value is not None:
            return tree.value
        
        # At internal node
        
        attribute_value = test_instance[tree.attribute_index]
         
        subtree = tree.left_child
        if isinstance(attribute_value,int) or isinstance(attribute_value,float):
            if attribute_value > tree.threshold :
                subtree = tree.right_child

        
        # Search in Subtree
        return self.classify(test_instance, subtree)
   
