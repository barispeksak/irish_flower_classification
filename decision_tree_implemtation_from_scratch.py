"""
Decision Tree Classifier from Scratch
=====================================
Implementation of decision tree algorithm for Iris flower classification.
Uses entropy, information gain, and recursive tree building.

Author: Baris Peksak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


class TreeNode:
    """Tree node for decision tree structure."""
    
    def __init__(self, feature=None, threshold=None, predicted_class=None, is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.predicted_class = predicted_class
        self.is_leaf = is_leaf
        self.left_child = None
        self.right_child = None


class DecisionTreeClassifier:
    """Decision Tree Classifier built from scratch."""
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None
    
    def load_data(self, filename="IRIS.csv"):
        """Load and preprocess the iris dataset."""
        df = pd.read_csv(filename)
        X = df.drop('species', axis=1).values
        y_text = df['species'].values
        
        species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        y = np.array([species_mapping[species] for species in y_text])
        
        return X, y
    
    def compute_entropy(self, y):
        """Calculate entropy for given labels."""
        if len(y) <= 0:
            return 0.
        
        entropy = 0.
        p0 = np.sum(y == 0) / len(y) 
        p1 = np.sum(y == 1) / len(y) 
        p2 = np.sum(y == 2) / len(y) 

        if p0 > 0:
            entropy -= p0 * np.log2(p0)
        if p1 > 0:
            entropy -= p1 * np.log2(p1)
        if p2 > 0:
            entropy -= p2 * np.log2(p2)

        return entropy

    def split_data(self, X, node_indices, feature, threshold):
        """Split data based on feature threshold."""
        left_indices = []
        right_indices = []

        for i in node_indices:
            if X[i][feature] <= threshold:
                left_indices.append(i)
            else:
                right_indices.append(i)
        
        return left_indices, right_indices

    def compute_information_gain(self, X, y, node_indices, feature, threshold):
        """Calculate information gain for a potential split."""
        left_indices, right_indices = self.split_data(X, node_indices, feature, threshold)

        X_node, y_node = X[node_indices], y[node_indices] 
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        w_left = len(y_left) / len(y_node)
        w_right = len(y_right) / len(y_node)

        entropy_node = self.compute_entropy(y_node)
        entropy_left = self.compute_entropy(y_left)
        entropy_right = self.compute_entropy(y_right)

        information_gain = entropy_node - ((w_left * entropy_left) + (w_right * entropy_right))
        return information_gain

    def get_best_split(self, X, y, node_indices):
        """Find the best feature and threshold for splitting."""
        unique_thresholds_list = []
        midpoint_list = []
        
        n_features = X.shape[1]
        subset_X = X[node_indices]

        # Get unique values for each feature
        for feature in range(n_features):
            unique_values = subset_X[:, feature]
            unique_values = np.unique(unique_values)
            unique_thresholds_list.append(unique_values.tolist())
        
        # Sort unique values
        for i in range(len(unique_thresholds_list)):
            unique_thresholds_list[i].sort()

        # Calculate midpoints between consecutive values
        for i in range(len(unique_thresholds_list)):
            midpoint_threshold_list = []
            for j in range(len(unique_thresholds_list[i]) - 1):
                midpoint_threshold_value = (unique_thresholds_list[i][j + 1] + unique_thresholds_list[i][j]) / 2
                midpoint_threshold_list.append(midpoint_threshold_value)
            midpoint_list.append(midpoint_threshold_list)   

        best_feature = -1
        max_info_gain = -1
        node_y = y[node_indices]
        
        # Check if node is pure
        if len(np.unique(node_y)) <= 1:
            return -1, -1  
        
        # Test all feature-threshold combinations
        for feature in range(n_features):
            for threshold in midpoint_list[feature]:
                info_gain = self.compute_information_gain(X, y, node_indices, feature, threshold)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def build_tree_recursive(self, X, y, node_indices, current_depth):
        """Recursively build the decision tree."""
        node_y = y[node_indices]
        
        # Stop if max depth reached
        if current_depth == self.max_depth:
            majority_class = np.bincount(node_y).argmax()
            return TreeNode(is_leaf=True, predicted_class=majority_class)
            
        best_feature, best_threshold = self.get_best_split(X, y, node_indices)
        
        # Stop if node is pure
        if best_feature == -1:
            pure_class = node_y[0]  
            return TreeNode(is_leaf=True, predicted_class=pure_class)
        
        # Create internal node and split
        left_indices, right_indices = self.split_data(X, node_indices, best_feature, best_threshold)
        node = TreeNode(feature=best_feature, threshold=best_threshold, is_leaf=False)
        
        # Recursively build children
        node.left_child = self.build_tree_recursive(X, y, left_indices, current_depth + 1)
        node.right_child = self.build_tree_recursive(X, y, right_indices, current_depth + 1)
        
        return node

    def fit(self, X, y):
        """Train the decision tree."""
        root_indices = list(range(len(y)))
        self.root = self.build_tree_recursive(X, y, root_indices, current_depth=0)
        return self

    def predict_single(self, sample):
        """Predict class for a single sample."""
        current_node = self.root
        while not current_node.is_leaf:
            if sample[current_node.feature] <= current_node.threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        return current_node.predicted_class

    def predict(self, X_test):
        """Predict classes for multiple samples."""
        predictions = []
        for sample in X_test:
            pred = self.predict_single(sample)
            predictions.append(pred)
        return predictions

    def accuracy(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return correct / len(y)
