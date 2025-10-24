import numpy as np
import numpy.random as r

class NN:
    def __init__(self, n_nodes = [100], activation_function = None, max_iter = 10**3, learning_rate = 0.005):
        self._params = None
        self._n_hidden_layers = len(n_nodes)
        self._n_nodes = np.asarray(n_nodes)
        self._activation_function = activation_function
        self._activations = None
        if activation_function is None:
            self._activation_function = self._ReLU
            self._d_activation_function = self._d_ReLU
        else:
            self._activation_function = activation_function
            self._d_activation_function = self._d_ReLU if activation_function == self._ReLU else self._d_sigmoid
        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self._cost_arr = None
        self._max = None
        self._min = None
    
    # ReLU activation
    def _ReLU(self, x):
        return np.maximum(0,x)
    
    # sigmoid activation
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    # derivatives of activation functions
    def _d_ReLU(self, x):
        return np.where(x<=0,0,1)   # returns 0 if true, else 1

    def _d_sigmoid(self, x):
        s = self._sigmoid(x)
        return s*(1-s)

    # Softmax function
    def _softmax(self, x):
        z = np.exp(x - np.max(x))
        return z / np.sum(z, axis=0, keepdims=True)

    # # to scale the input data
    # def _scale(self, X, fit=False):
    #     if fit:
    #         self._mean = X.mean(axis=0)
    #         self._std = X.std(axis=0) + 1e-8
    #     return (X - self._mean) / self._std

    # to scale the input data
    def _scale(self, X, fit=False):
        if fit==True:
            self._max = np.max(X)
            self._min = np.min(X)
        return (X - self._min) / (self._max - self._min)

    # defining the cost function
    def _cost(self, y_estim, y_true):
        return np.sum((y_estim-y_true)**2)

    # feeding forward (till last hidden layer)
    def _forward_feed_1(self, a, w, b):
        return self._activation_function(w @ a + b)
    
    # feeding forward (from last hidden layer to output layer)
    def _forward_feed_2(self, a, w, b):
        return self._softmax(w @ a + b)

    # feeding forward (from input layer to output layer)
    def _forward_feed_complete(self, a, W, B, y_true):
        activations = [a]  # to store the activations in each layer
        
        # propagation through hidden layers
        for i in range(self._n_hidden_layers):
            a = self._activation_function(W[i] @ a + B[i])
            activations.append(a)
        
        # propagation to the output layer
        a = self._softmax(W[-1] @ a + B[-1])
        activations.append(a)
        cost = self._cost(a, y_true)
        return activations, cost

    # knonecker delta function
    def _delta(self, i, j):
        return int(i == j)

    # back propagation (Output layer)
    def _back_propagate_output_layer(self, y_true, activations):
        a_out = activations[-1]
        x = 2 * (a_out - y_true)
        n = len(a_out)
        s = a_out.reshape(-1, 1)
        J = np.diagflat(s) - s @ s.T
        dCdb = J.T @ x
        dCdw = np.tensordot(dCdb, activations[-2], axes = 0)
        return dCdb, dCdw
    
    # back propagation (other than the output layer)
    def _back_propagate_hidden_layer(self, b, w1, w2, a, dCdb):
        z = w1 @ a + b
        dCda = w2.T @ dCdb
        dCdb = dCda * self._d_activation_function(z)
        dCdw = np.tensordot(dCdb,a,axes = 0)
        return dCdb, dCdw, dCda

    # back propagation (complete)
    def _back_propagate_complete(self, W, B, activations, y_true):
        dCda_arr = [0] * self._n_hidden_layers
        dCdb_arr = [0] * (self._n_hidden_layers + 1)
        dCdw_arr = [0] * (self._n_hidden_layers + 1)
        bias, weight = self._back_propagate_output_layer(y_true, activations)
        dCdb_arr[-1] = bias
        dCdw_arr[-1] = weight
        for i in range(self._n_hidden_layers-1,-1,-1):
            d_bias, d_weight, d_activ = self._back_propagate_hidden_layer(B[i], W[i], W[i+1], activations[i], dCdb_arr[i+1])
            dCdb_arr[i] = d_bias
            dCdw_arr[i] = d_weight
            dCda_arr[i] = d_activ
        return dCdb_arr, dCdw_arr, dCda_arr

    # defining the fit method
    def _fit(self, X, y):
        # skeleton for the NN
        n,m = X.shape
        n_targets = len(np.unique(y))
        self._n_nodes = np.append([m],self._n_nodes)
        self._n_nodes = np.append(self._n_nodes,[n_targets])

        # scaling the data
        # X_scaled = self._scale(X.T, fit = True).T
        X_scaled = self._scale(X, fit = True)
        
        # initializing the weights and biases
        W, B = [],[]
        for i in range(len(self._n_nodes)-1):
            W.append(r.randn(self._n_nodes[i+1], self._n_nodes[i]) * np.sqrt(2 / self._n_nodes[i]))
            B.append(np.zeros((self._n_nodes[i+1],), float))

        # using Stochastic gradient descent
        cost_arr = []
        p = np.linspace(0,self._max_iter-1,11)
        p = list(map(int,p))
        c = 0
        for i in range(self._max_iter):
            indices = np.arange(n)
            r.shuffle(indices)
            X_shuffled = X_scaled[indices]
            y_shuffled = y[indices]
            cost = 0.0
            for j in range(n):
                y_true = np.zeros((self._n_nodes[-1],),float)
                y_true[y[j]] = 1
                activations, cost_per_item = self._forward_feed_complete(X_shuffled[j], W, B, y_true)
                dCdb_arr, dCdw_arr, dCda_arr = self._back_propagate_complete(W, B, activations, y_true)
                for k in range(self._n_hidden_layers + 1):
                    W[k] -= self._learning_rate * dCdw_arr[k]
                    B[k] -= self._learning_rate * dCdb_arr[k]
                cost += cost_per_item
            cost_arr.append(cost/n)
            if i==p[c]:
                print(f'{c*10} % complete --> cost : {cost_arr[-1]}')
                c+=1
            
            # if i % 100 == 0:
            #     print(f'Epoch {i}, cost : {cost_arr[-1]}')
        self._params = [W, B]
        self._cost_arr = cost_arr

    # fitting function (public)
    def fit(self, X, y):
        self._fit(X, y)

    # prediction
    def _predict(self, X):
        W, B = self._params
        predictions = []
        probs = []
        # X_scaled = self._scale(X.T).T
        X_scaled = self._scale(X)
        for x in X_scaled:
            a = x
            for i in range(self._n_hidden_layers):
                a = self._activation_function(W[i] @ a + B[i])
            a = self._softmax(W[-1] @ a + B[-1])
            predictions.append(np.argmax(a))
            probs.append(a)
        return np.array(predictions), np.array(probs)
    
    # prediction method (public)
    def predict(self, X):
        return self._predict(X)