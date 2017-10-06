import numpy as np
from evaluation_helpers import split_dataset

def evaluate_model(X, y, model, num_splits=10, train_set_percentages=[10, 25, 50, 75, 100]):
    train_score = np.empty(num_splits)
    test_score = np.empty(num_splits)
    error_rates = np.empty((len(train_set_percentages), num_splits))
    for i in range(num_splits):
        X1, X2, y1, y2 = split_dataset(X, y, train_set_size=0.8)
        for j, percent in enumerate(train_set_percentages):
            X_train, _, y_train, _ = split_dataset(X1, y1, train_set_size=percent/100)
            error_rates[j, i]  = model.fit(X_train, y_train).score(X2,y2)
    return error_rates

def error(y, y_pred):
    num_errors = np.count_nonzero(y - y_pred)
    return num_errors / y.size
