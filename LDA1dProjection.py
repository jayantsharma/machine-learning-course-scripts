import numpy as np
from numpy.linalg import inv, pinv, eig, norm
from load_data import load_data
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from pylab import savefig
from evaluation_helpers import splice_dataset_randomly, train_test_split_at

def LDA1dProjection(filename, num_crossval):
    X, y = load_data(filename)
    classes = np.unique(y)
    dataset = filename.split('.')[0]
    data = np.column_stack((X,y))

    spliced_dataset = splice_dataset_randomly(data, num_crossval)
    Xp_train, ys_train = list(), list()
    Xp_test, ys_test = list(), list()
    w = None
    for i in range(num_crossval):
        X_train, X_test, y_train, y_test = train_test_split_at(spliced_dataset, i)
        w = fit(X_train, y_train, w)

        Xp_train.append(transform(X_train, w))
        ys_train.append(y_train)

        Xp_test.append(transform(X_test, w))
        ys_test.append(y_test)

    Xp_train = np.concatenate(Xp_train)
    ys_train = np.concatenate(ys_train)

    Xp_test = np.concatenate(Xp_test)
    ys_test = np.concatenate(ys_test)

    plt.figure()
    plt.title('Training data LDA 1-d projection for the {} dataset'.format(dataset.capitalize()))
    plt.hist( [Xp_train[ys_train == i] for i in classes], bins = 20, normed = 1, histtype='bar')
    savefig("LDA1dProjection_training", bbox_inches='tight')

    plt.figure()
    plt.title('Testing data LDA 1-d projection for the {} dataset'.format(dataset.capitalize()))
    plt.hist( [Xp_test[ys_test == i] for i in classes], bins = 20, normed = 1, histtype='bar')
    savefig("LDA1dProjection_testing", bbox_inches='tight')

def fit(X_train, y_train, w_old):
    classes = [ int(i) for i in np.unique(y_train) ]
    means = [np.mean(X_train[y_train == i], axis = 0, keepdims=True) for i in classes]
    Sw = np.sum ( 
            [
                np.sum ( 
                    [ 
                        np.outer(x - means[i], x - means[i]) for x in X_train[y_train == i] 
                    ]
                    , axis = 0 )
                for i in classes 
            ]
            , axis = 0)
    Sw_inv = inv(Sw)

    N = [X_train[y_train == i].shape[0] for i in classes]
    M = np.mean(X_train, axis=0)
    Sb = np.sum([N[i] * np.outer(means[i] - M, means[i] - M) for i in classes], axis=0)

    evals, evecs = eig(np.dot(Sw_inv, Sb))

    w = evecs[:, np.argmax(evals)]
    if np.any(w_old):
        if(w_old.dot(w)) < 0:
            w = -w

    return w

def transform(X_test, w):
    X_projected = np.dot(X_test, w)
    return X_projected

if __name__ == '__main__':
    import sys
    num_crossval = 10
    filename = sys.argv[1]
    # import pdb; pdb.set_trace()
    try:
        num_crossval = int(sys.argv[2])
    except:
        pass
    LDA1dProjection(filename, num_crossval)
