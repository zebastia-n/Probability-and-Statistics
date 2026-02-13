import numpy as np

class LingerRegressions:

    def __init__(self):
        self.b = None
        self.d = None
        self.n = None       

    def fit(self, X, Y):

        X = np.array(X)
        Y = np.array(Y)
   
        ones = np.ones((X.shape[0], 1))
        X = np.column_stack([ones, X])

        self.n = X.shape[0]     # n = sample size
        self.d = X.shape[1] - 1 # d = number of futurs you are using

        self.b = np.linalg.pinv(X.T @ X) @ X.T @ Y # BETA

    def predict(self, X):
        X = np.array(X)

        ones = np.ones((X.shape[0], 1))
        X = np.column_stack([ones, X])
        return X @ self.b

    def SSE(self, X,Y):
        Y = np.array(Y)
        sse = Y - self.predict(X)
        return  np.sum(sse**2)
    
    def mean(self, Y):
        Y = np.array(Y)
        return np.mean(Y)

    def regression_variance(self, Y):
        return np.sum((Y - (np.mean(Y)))**2) / (len(Y) - 1)

    def standard_deviation(self, Y):
        return np.sqrt(self.regression_variance(Y))

    def rmse(self, X, Y):
        Y = np.array(Y)
        Y_hat = self.predict(X)
        return np.sqrt(np.mean((Y - Y_hat)**2))