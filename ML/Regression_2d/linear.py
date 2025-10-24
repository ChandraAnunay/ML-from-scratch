import numpy as np

class LinearRegression:
    def __init__(self, data):
        """
        This code accepts data in the format : [[x1,y1],[x2,y2],[x3,y3],...,[xn,yn]]
        Here, 'n' is the total number of data points available
        Dimension of feature space = 1
        Single Feature : x
        y(s) serve as continuous target values

        Fitting Model : y = m*x+c
        """
        self._data = np.array(data)
        self._x = self._data[:, 0]
        self._y = self._data[:, 1]
        self._params = None

    def fit(self):
        A = np.column_stack((self._x, np.ones_like(self._x)))
        b = self._y
        ATA = A.T @ A
        ATb = A.T @ b
        self._params = np.linalg.solve(ATA, ATb)
        return self._params[0], self._params[1]

    def predict(self, x):
        x = np.asarray(x)
        if self._params is None:
            raise ValueError("Model must be fitted before prediction.")
        return self._params[0] * x + self._params[1]
    
    def SSE(self):  # Sum square error
        residuals = self.predict(self._x) - self._y
        return np.sum(residuals ** 2)
    
    def RMSE(self):  # Mean squared error
        return np.sqrt(self.SSE() / len(self._x))

if __name__ == '__main__':
    import numpy.random as r
    import matplotlib.pyplot as plt

    # Generate sample data
    def f(x):
        return 3 * x + 1

    x = np.linspace(0, 5, 100)
    x += r.standard_normal(len(x))
    y = f(x)
    y += r.standard_normal(len(y))

    data = np.column_stack((x, y))
    model = LinearRegression(data)
    m, c = model.fit()
    SSE = model.SSE()
    RMSE = model.RMSE()

    print('Slope            :', m)
    print('Intercept        :', c)
    print('SSE              :',SSE)
    print('RMSE             :',RMSE)

    # Plotting
    plt.scatter(x, y, label='Data',marker='.')
    x_line = np.array([min(x), max(x)])
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, 'r-', label=f'Fitted Line: {m:.3f}x + {c:.3f}')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.grid(True)
    plt.show()
