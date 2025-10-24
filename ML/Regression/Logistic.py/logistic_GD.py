# THE PARAMETERS FOR LOGISTIC REGRESSION IN THIS CODE ARE OBTAINED USING 'GRADIENT DESCENT'
import numpy as np

# this class deals with a two class logistic regression problem only
class Logistic_Regression:
    def __init__(self):
        self._parameters = None
        self._cost = None
        self._X_max = None
    
    # normalizing the data against individual features' maxima
    def _normalize(self, X):
        return X / self._X_max
    
    # Sigmoid function
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Probability of class 1
    def _p(self, theta, x):
        return self._sigmoid(theta @ x)

    # defining the cost function and its gradient for 2-class logistic regression
    def _cost_func(self, theta, X, y):
        preds = self._sigmoid(X @ theta)
        epsilon = 1e-15
        # to avoid log(0)
        preds = np.clip(preds, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
    
    def _grad_cost(self, theta, X, y):
        preds = self._sigmoid(X @ theta)
        return X.T @ (preds - y) / len(y)
    
    # Gradient descent algorithm
    def fit(self, X, y, theta_ini, learning_rate = 0.1 , max_iter = 1000):
        self._X_max = X.max(axis=0)
        X = self._normalize(X)
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding the bias term
        theta = np.array(theta_ini)
        cost_arr = []

        for _ in range(max_iter):
            grad = self._grad_cost(theta, X_b, y)
            theta -= learning_rate * grad
            cost = self._cost_func(theta, X_b, y)
            cost_arr.append(cost)
        
        self._parameters = theta
        self._cost = np.array(cost_arr)
    
    # predictor function
    def pred(self, X):
        X = self._normalize(X)
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term
        probs = self._sigmoid(X_b @ self._parameters)
        return (probs >= 0.5).astype(int)  # convert the boolean mask to class labels
    
    # getting the trained parameters
    def get_params(self):
        return self._parameters
    
    # getting the cost array
    def get_cost(self):
        return self._cost
    
    # accuracy score
    def accuracy(self, X, y):
        y_pred = self.pred(X)
        return np.mean(y == y_pred)
    
if __name__ == '__main__':
    # loading the breast cancer dataset from sklearn
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    print('The model is being trained...will take around 20 seconds')

    data = load_breast_cancer()
    X = data.data
    y = data.target
    # splitting the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)
    # training the logistic model
    model = Logistic_Regression()
    params_ini = np.zeros((len(X[0])+1,),float)
    n_iter = 10**5
    rate = 1
    model.fit(X_train, y_train, params_ini, learning_rate=rate, max_iter=n_iter)
    params = model.get_params()
    cost = model.get_cost()
    accuracy_score = model.accuracy(X_test, y_test)
    print(f'Accuracy : {accuracy_score:.04f}\n\nParameters : \n\n{params}')

    # visualising cost against the number of iterations
    plt.plot(np.arange(n_iter),np.log10(cost))
    plt.title('Evolution of cost')
    plt.xlabel('Iteration')
    plt.ylabel('log10(cost)')
    plt.show()