import numpy as np
from numpy.linalg import inv, pinv, eig, norm
from load_data import load_data
import matplotlib.pyplot as plt
from pylab import savefig
from evaluate_model import error
from evaluation_helpers import splice_dataset_randomly, train_test_split_at

class GaussianGM():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.target_names = np.unique(y)

        N = np.array([X[y == i].shape[0] for i in self.target_names])
        self.priors = N / N.sum()

        self.means = np.array(
            [
                (np.sum(X[y == i], axis=0) / N[i]) for i in self.target_names
            ]).T

        Ss = np.array([
            np.sum(
                np.array(
                    [ np.outer(x - self.means[:,i], x - self.means[:,i] ) for x in X[y == i] ]
                    )
                , axis=0 ) / N[i]
            for i in self.target_names])

        self.covar = np.sum(
            np.array(
                [ (N[i] / N.sum()) * Ss[i] for i in self.target_names ]
                )
            , axis = 0)

        return self

    def predict(self, X):
        from scipy.stats import multivariate_normal as mvnorm
        y_probs = np.array(
            [
                mvnorm.pdf(X, mean=self.means[:,i], cov=self.covar)
                for i in self.target_names
                ]
            ).T
        y_probs = y_probs * self.priors
        y_preds = np.apply_along_axis(np.argmax, 1, y_probs)
        return y_preds

    def score(self, X, y):
        y_preds = self.predict(X)
        return error(y, y_preds)

def LDA2dProjection(filename):
    X, y = load_data(filename)
    classes = np.unique(y)

    m = [np.mean(X[y == i], axis = 0, keepdims=True) for i in classes]
    Sw = np.sum ( [np.sum( [ np.outer(x - m[i], x - m[i]) for x in X[y == i] ], axis = 0 ) for i in classes], axis = 0)
    # Sw is singular.
    Sw_inv = pinv(Sw)

    N = np.array([X[y == i].shape[0] for i in classes])
    M = np.mean(X, axis=0)
    Sb = np.sum([N[i] * np.outer(m[i] - M, m[i] - M) for i in classes], axis=0)
    evals, evecs = eig(np.dot(Sw_inv, Sb))

    W = np.array( [evecs[:,j] for j in range(evecs.shape[0]) for eg in np.sort(evals)[-2:] if evals[j] == eg] )
    X_r2 = np.array([np.dot(W, x) for x in X])

    plt.figure()
    for i in classes:
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], label=i)
    plt.legend()
    plt.title('LDA of Digits dataset: 2d projection')
    savefig("LDA2dProjectionDigits", bbox_inches='tight')

    return X_r2, y

def cross_validate_model(X, y, num_crossval):
    dataset = np.column_stack((X,y))
    spliced_dataset = splice_dataset_randomly(dataset, num_crossval)
    training_error = np.empty(num_crossval)
    test_error = np.empty(num_crossval)
    # import pdb; pdb.set_trace()
    for i in range(num_crossval):
        X_train, X_test, y_train, y_test = train_test_split_at(spliced_dataset, i)
        # import pdb; pdb.set_trace()
        gaussian_classifier = GaussianGM().fit(X_train, y_train)
        training_error[i] = gaussian_classifier.score(X_train, y_train)
        test_error[i] = gaussian_classifier.score(X_test, y_test)

    print("--------------------------------")
    print("TRAINING ERROR")
    print(training_error)
    print("--------------------------------")
    print("TEST ERROR")
    print(test_error)
    print("--------------------------------")
    print("Training error mean : ", training_error.mean())
    print("Test error mean : ", test_error.mean())
    print("Test error std : ", test_error.std())

if __name__ == '__main__':
    import sys
    num_crossval = 10
    filename = sys.argv[1]
    # import pdb; pdb.set_trace()
    try:
        num_crossval = int(sys.argv[2])
    except:
        pass
    X_r2, y = LDA2dProjection(filename)
    cross_validate_model(X_r2, y, num_crossval)
#     print(GaussianGM().fit(X_r2, y).score(X_r2, y))
