import numpy as np

def train_test_split_at(spliced_dataset, i):
    training_data = np.row_stack(np.delete(spliced_dataset, i))
    test_data = spliced_dataset[i]
    X_train, y_train = training_data[:,:-1], training_data[:,-1]
    X_test, y_test = test_data[:,:-1], test_data[:,-1]
    return real(X_train), real(X_test), integer(real(y_train)), integer(real(y_test))

def splice_dataset_randomly(dataset, crossval):
    # import pdb; pdb.set_trace()
    y = dataset[:,-1]
    classes = np.unique(y)
    K = len(classes)
    spliced_by_class = list()
    for k in range(K):
        spliced_class_dataset = _splice_dataset_randomly(dataset[y == k], crossval)
        spliced_by_class.append(spliced_class_dataset)

    spliced_dataset = list()
    for i in range(crossval):
        splice = list()
        for k in range(K):
            splice.append(spliced_by_class[k][i])
        spliced_dataset.append(np.concatenate(np.array(splice)))

    return spliced_dataset

def _splice_dataset_randomly(dataset, crossval):
    N = dataset.shape[0]
    permutation_order = np.eye(N)[np.random.permutation(N)]
    permuted_matrix = permutation_order.dot(dataset)
    # array of crossval datasets
    spliced_dataset = np.array(np.array_split(permuted_matrix, crossval))
    return spliced_dataset

def split_dataset(X, y, train_set_size=0.5):
    dataset = np.column_stack((X,y))
    classes = np.unique(y)
    K = len(classes)
    split_by_class = list()
    for k in range(K):
        split_class_dataset = _split_dataset(dataset[y == k], train_set_size)
        split_by_class.append(split_class_dataset)

    split_dataset = list()
    for i in range(2):
        split = list()
        for k in range(K):
            split.append(split_by_class[k][i])
        split_dataset.append(np.concatenate(np.array(split)))

    X_train, y_train = split_dataset[0][:,:-1], split_dataset[0][:,-1]
    X_test, y_test = split_dataset[1][:,:-1], split_dataset[1][:,-1]

    return real(X_train), real(X_test), integer(real(y_train)), integer(real(y_test))

def _split_dataset(dataset, train_set_size):
    N = dataset.shape[0]
    permutation_order = np.eye(N)[np.random.permutation(N)]
    permuted_matrix = permutation_order.dot(dataset)
    # array of crossval datasets
    split_dataset = np.array(np.array_split(permuted_matrix, [int(train_set_size * N)]))
    return split_dataset

def real(mat):
    if np.any(mat):
        vfunc = np.vectorize(lambda x: np.real(x))
        return vfunc(mat)
    else:
        return mat

def integer(vec):
    if np.any(vec):
        vfunc = np.vectorize(lambda x: int(x))
        return vfunc(vec)
    else:
        return vec

# foo = np.array( [ [ 1, 0 ], [ 3, 1 ], [ 5, 0 ], [ 7, 1 ] ] )
# print(split_dataset(foo, train_set_size=0.5))
