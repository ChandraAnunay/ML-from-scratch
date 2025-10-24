import numpy as np

class PolynomialRegression:
    def __init__(self,data):
        """
        This code accepts data in the format : [[x1,y1],[x2,y2],[x3,y3],...,[xn,yn]]
        Here, 'n' is the total number of data points available
        Dimension of feature space = 1
        Single Feature : x
        y(s) serve as continuous target values

        Fitting Model : y = a_0 + a_1*x + a_2*x^2 + ... + a_d*x^d
        where d = degree of the polynomial
        """
        self._data = data
        self._deg = 0
        self._x = self._data[:, 0]
        self._y = self._data[:, 1]
        self._params = None

    def fit(self,deg = 1):  # does a linear fit by default if no value for degree is passed
        self._deg = deg
        l = len(self._data)
        A = np.zeros((l,self._deg+1),float)
        b = self._y
        for i in range(l):
            for j in range(self._deg+1):
                A[i][j] = (self._x[i])**(self._deg-j)