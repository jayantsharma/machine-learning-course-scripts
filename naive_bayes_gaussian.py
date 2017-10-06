import numpy as np
from load_data import load_data
from evaluate_model import evaluate_model, error
from numpy.linalg import pinv

class GNB():
    def __init__(self):
        pass

    def fit(self, X, y):
        # convenience variables
        self.D = X.shape[1]

        self.target_names = np.unique(y)
        K = len(self.target_names)
        self.K = K

        t = np.eye(self.K)[y]        # one-hot vector
       
        N = t.sum(axis=0)
        self.priors = N/t.sum()

        self.means = X.T.dot(t) / N

        Ss = np.empty([self.D, self.K])

        for k in self.target_names:
            Ss[:,k] = ((X - self.means[:,k]) ** 2).T.dot(t[:,k]) / N[k]

        self.cov = (Ss * N).sum(axis=1) / N.sum()
        return self

    def predict(self, X):
        from scipy.stats import multivariate_normal as mvnorm
        from functools import reduce
        import operator
        
        def pdf(x,mu,sigma):
            try:
                return mvnorm.pdf(x, mean=mu, cov=sigma, allow_singular=True)
            except:
                print(x, mu, sigma)

        _ypred = np.array(
            [
                [
                    reduce(operator.mul, [
                        pdf(x[i], self.means[i,j], self.cov[i]) 
                        for i in range(self.D)
                    ], 1)
                    for j in range(self.K)
                ]
                for x in X
            ]
        )
        ypred = _ypred * self.priors
        return ypred.argmax(axis=1)
    
    def score(self, X, y):
        y_preds = self.predict(X)
        return error(y, y_preds)

def naiveBayesGaussian(filename, num_splits=10, train_set_percentages=[10, 25, 50, 75, 100]):
    X, y = load_data(filename)
    model = GNB()
    stats = evaluate_model(X, y, model, num_splits, train_set_percentages)
    print("Test set errors for each split for each Training set percentage:")
    print(stats)
    print("Mean test set error for each Training set perentage:")
    print(stats.mean(axis=1))
    return stats

if __name__ == '__main__':
    import sys
    num_splits = 10
    train_set_percentages = [10, 25, 50, 75, 100]

    filename = sys.argv[1]
    try:
        num_splits = int(sys.argv[2])
        train_set_percentages = [ int(x) for x in sys.argv[3].split(',') ]
    except:
        pass
    naiveBayesGaussian(filename, num_splits, train_set_percentages)
