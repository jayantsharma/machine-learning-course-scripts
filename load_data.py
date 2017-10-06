import numpy as np
import re

def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    X, target = data[:,:-1], data[:,-1:]
    target = np.array([ int(t) for t in target ])

    # If Boston dataset, see if there's need to classify
    if re.compile(".*boston.*").match(filename):
        if list(np.unique(target)) == [0,1]:
            pass
        else:
            target_median = np.median(target)
            classify_target = lambda t: 1 if t >= target_median else 0
            target = np.array([classify_target(t) for t in target])
            data[:,-1:] = target.reshape(len(data), 1)

    return X, target

if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    print(load_data(filename)[1][:15])
