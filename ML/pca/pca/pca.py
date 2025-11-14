import numpy as np

class PCA:
    def __init__(self,n_components):
        # n_components : dimension of the reduced feature space
        self.n_components = n_components
        self.mean = None
        self.cov = None
        self.eigenvec = None
        self.principle_components = None
        self.projection = None
    
    def fit(self,X):
        # mean
        self.mean = np.mean(X,axis = 0)
        # centralizing the data
        X = X - self.mean
        # variance-covariance matrix
        self.cov = np.cov(X.T)
        # eigenvalues and eigenvectors of the var-cov matrix
        eigenval, eigenvec = np.linalg.eig(self.cov)
        # sorting the eigenvectors in decreasing order of eigenvalues
        indices = np.argsort(eigenval)[::-1]
        eigenvec = eigenvec.T
        # self.eigenvec = eigenvec[indices][0:self.n_components].T
        # self.eigenvec = eigenvec[:, indices[:self.n_components]]
        self.eigenvec = eigenvec[indices][0:self.n_components].T
    
    def transform(self,X):
        # centralizing the data
        X = X - self.mean
        self.principle_components = np.matmul(X,self.eigenvec)
        self.projection = np.matmul(self.principle_components,self.eigenvec.T)
        return self.principle_components, self.projection + self.mean
