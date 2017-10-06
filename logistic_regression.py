import numpy as np
from load_data import load_data
from evaluate_model import evaluate_model, error
from numpy.linalg import pinv

class LR():
    def __init__(self):
        pass

    def fit(self, X, y, num_iter=10):
        # convenience variables
        N = y.shape[0]
        D = X.shape[1]

        self.target_names = np.unique(y)
        K = len(self.target_names)
        self.K = K

        t = np.eye(K)[y][:,:K-1].flatten('F')         # 1-hot vectors for each of the target values, eg: [0 0 1 0 0]

        self.W = np.zeros((D+1,K-1))
        # self.W = np.random.rand(D+1,K-1)

        # normalize
        self.compute_mean_std_of_dataset(X)
        # normalize using mean and std, then attach ones; shape - N x (D+1)
        X = self.padded(self.normalized(X))

        # matrix with X at diagonals
        _X = self.construct_X_(X)
        for i in range(num_iter):
#            print("Iter num : ", i)

            y_pred = self.predict(X)
#            print(y_pred[:5,:5])
#            print("Score : ", accuracy_score(y, y_pred.argmax(axis=1), normalize=True))
            y_pred = y_pred[:,:-1].flatten('F')

            difference = y_pred - t
            # print(difference.shape, _X.shape)
            gradient = _X.T.dot(difference).reshape(((D+1)*(K-1),1))
            R = self.construct_R(y_pred, N)
            Hessian = _X.T.dot(R).dot(_X)

            try:
                Winc = pinv(Hessian).dot(gradient).reshape((D+1,K-1), order='F')
                Wnew = self.W - Winc
                # break if Wnew =~ W
                if np.all(np.isclose(self.W, Wnew)):
                    break
                else:
                    self.W = Wnew
            except:
                # import pdb; pdb.set_trace()
                pass
            # print(accuracy_score(y, y_pred, normalize=True))
#            print("Here's what weights look like")
#            print(self.W[:10,:])
#            print("------------------")
        return self

    def compute_mean_std_of_dataset(self, X):
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        vf = np.vectorize(lambda x: x if x != 0 else 1, otypes=[np.float])
        self.X_std = vf(self.X_std)
#        import pdb; pdb.set_trace()
#        print(X_mean[:5], X_std[:5])

    def normalized(self, X):
        return (X - self.X_mean) / self.X_std

    def padded(self, X):
        return np.column_stack((np.ones(X.shape[0]), X))

    def construct_X_(self,X):
        m, n, K = X.shape[0], X.shape[1], self.K
        _X = np.zeros((m*(K-1), n*(K-1)))
        for i in range(K-1):
            _X[m*i:m*(i+1), n*i:n*(i+1)] = X
        return _X

    def Ijk(self,j,k):
        return 1 if k == j else 0

    def construct_R(self, y_pred, N):
        K = self.K
        R = np.zeros((N*(K-1), N*(K-1)))
        for j in range(K-1):
            for k in range(K-1):
                for i in range(N):
                    R[j*N + i, k*N + i] = y_pred[j*N+i] * (self.Ijk(j,k) - y_pred[k*N+i])
        return R

    def softmax(self, A):
        A = np.exp(A)
        L1_norm = A.sum(axis=1)
        return (A.T / (L1_norm)).T

    def predict(self, X):
        prod = X.dot(self.W)
#        print("X looks like:")
#        print(X[:3,:5])
#        print("X.W looks like:")
#        print(prod[:3,:5])
        zero = np.zeros(X.shape[0])
        A = np.column_stack((prod, zero))
        yp = self.softmax(A)
        return yp

#    def _predict(self, X):
#        foo = self.predict(X)
#        return foo[:,:-1]

    def score(self, X, y):
        Xtest = self.padded(self.normalized(X))
        y_preds = self.predict(Xtest).argmax(axis=1)
        return error(y, y_preds)

def logisticRegression(filename, num_splits=10, train_set_percentages=[10, 25, 50, 75, 100]):
    print(filename, num_splits, train_set_percentages)
    X, y = load_data(filename)
    model = LR()
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
    # import pdb; pdb.set_trace()
    try:
        num_splits = int(sys.argv[2])
        train_set_percentages = [ int(x) for x in sys.argv[3].split(',') ]
    except:
        pass
    logisticRegression(filename, num_splits, train_set_percentages)
