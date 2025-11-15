import numpy as np

class decision_tree:
    def __init__(self, max_depth = 5, min_leaf_size = 2):
        self._max_depth = max_depth
        self._min_leaf_size = min_leaf_size
        self._tree = None
        self._n_leaf = 0
        self._features_used = []
    
    def fit(self, X, y):
        self._tree = self._tree_formation(X,y,depth = 0)

    def predict(self, x):
        x = np.asarray(x)
        predicted_classes = np.zeros((len(x),),int)
        for i,point in enumerate(x):
            predicted_classes[i] = self._predict(point,self._tree)
        return predicted_classes
    
    def _gini_index(self, y):                                   # returns the gini index of inputted node (any root/decision/leaf)
        _, freq = np.unique(y, return_counts = True)
        p = freq / sum(freq)                                    # array of probability of getting a particular label/class
        return 1 - sum(p**2)

    def _best_split_line(self, X, y):
        n_samples, n_features = np.shape(X)
        best_gain = -1
        best_line = None
        parent_gini_index = self._gini_index(y)
        for feature in range(n_features):
            vals = X[:,feature]
            for support_point in vals:
                left_daughter = vals[vals <= support_point]
                y_left = y[vals <= support_point]
                right_daughter = vals[vals > support_point]
                y_right = y[vals > support_point]
                if len(left_daughter) == 0 or len(right_daughter) == 0:
                    continue
                weight_left_daughter = len(left_daughter)/n_samples     # proportion of data points in left daughter node compared to the parent node
                weight_right_daughter = len(right_daughter)/n_samples   # proportion of data points in right daughter node compared to the parent node
                information_gain = parent_gini_index - (weight_left_daughter*self._gini_index(y_left)+weight_right_daughter*self._gini_index(y_right))
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_line = {'feature' : feature, 'threshold' : support_point}
        return best_line           # returns the best splitting line available, if not then returns 'None'

    def _decide_label(self, y):
        labels, freq = np.unique(y, return_counts = True)
        return labels[np.argmax(freq)]              # return the label with maximum frequency

    def _tree_formation(self, X, y, depth):
        if len(X) <= self._min_leaf_size or depth > self._max_depth or len(set(y)) == 1:
            self._n_leaf += 1
            pure = False
            if len(set(y)) == 1:
                pure = True
            return {'leaf' : True, 'label' : self._decide_label(y), 'size' : len(X), 'Pure' : pure}
        best_split_line = self._best_split_line(X,y)
        if best_split_line == None:
            self._n_leaf += 1
            return {'leaf' : True, 'label' : self._decide_label(y), 'size' : len(X), 'Pure' : False}
        feature, threshold = best_split_line['feature'], best_split_line['threshold']
        self._features_used.append(feature)
        # bifurcation into left and right daughter nodes
        X_left, y_left = X[X[:,feature] <= threshold], y[X[:,feature] <= threshold]
        X_right, y_right = X[X[:,feature] > threshold], y[X[:,feature] > threshold]
        left_daughter = self._tree_formation(X_left,y_left,depth+1)
        right_daughter = self._tree_formation(X_right,y_right,depth+1)
        return {'leaf' : False, 'feature' : feature, 'threshold' : threshold, 'left' : left_daughter, 'right' : right_daughter}
    
    def get_used_features(self):
        return np.sort(np.array(list(set(self._features_used))))

    def accuracy_score(self, y_test, y_pred):
        if len(y_pred) != len(y_test):
            return 'Arrays with different lengths encountered'
        if len(y_pred) == 0:
            return 'Arrays with zero length encountered'
        N = len(y_test)
        correct_vals = 0
        for i in range(N):
            if y_pred[i] == y_test[i]:
                correct_vals+=1
        return correct_vals/N

    def _predict(self, x, tree):
        if tree['leaf'] == True:
            return tree['label']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict(x,tree['left'])
        else:
            return self._predict(x,tree['right'])

    def _print_tree(self, node=None, depth=0):
        if node is None:
            node = self._tree

        indent = "\t" * depth
        if node['leaf']:
            return f"{indent}Leaf: label = {node['label']}, size = {node['size']}, pure leaf = {node['Pure']}\n"
        else:
            s = f"{indent}Node: Feature {node['feature']} <= {node['threshold']:.3f}\n"
            s += self._print_tree(node['left'], depth + 1)
            s += self._print_tree(node['right'], depth + 1)
            return s
    
    def __str__(self):
        if self._tree == None:
            return 'Use the fit function first to construct a tree'
        else:
            return f'The decision tree is given as below :\n..............................................\n\n{self._print_tree()}'

if __name__ == '__main__':
    # this is a sample example
    # we have considered the iris dataset available in the sklearn library
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X,y = iris.data, iris.target
    # splitting the dataset into training and testing set
    ratio = 0.4
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio ,random_state=31)
    model = decision_tree()
    # training the model using training set of data
    model.fit(X_train,y_train)
    # making predictions for X_test classes using the trained model
    y_pred = model.predict(X_test)
    accuracy = model.accuracy_score(y_pred,y_test)
    feat_used = model.get_used_features()
    print(model)
    print(f'..............................................\nRatio of size of training dataset to total size of available data : {ratio}')
    print('Accuracy Score :',accuracy)
    print('Features used :',feat_used)
