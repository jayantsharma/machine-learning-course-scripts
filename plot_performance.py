from pylab import savefig
import matplotlib.pyplot as plt
from naive_bayes_gaussian import naiveBayesGaussian
from logistic_regression import logisticRegression

def plot_gnb_vs_lr_performance(dataset, num_splits=10, train_set_percentages=[10, 25, 50, 75, 100]):
    gnb_stats = naiveBayesGaussian(dataset, num_splits, train_set_percentages)
    lr_stats = logisticRegression(dataset, num_splits, train_set_percentages)
    dataset = dataset.split('/')[-1].split('.')[0]

    y_gnb = gnb_stats.mean(axis=1)
    y_gnb_err = gnb_stats.std(axis=1)

    y_lr = lr_stats.mean(axis=1)
    y_lr_err = lr_stats.std(axis=1)

    plt.figure()
    plt.errorbar(train_set_percentages, y_gnb, yerr=y_gnb_err, fmt='--o', capsize=5)
    plt.errorbar(train_set_percentages, y_lr, yerr=y_lr_err, fmt='--o', capsize=5)
    #x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,y1,0.4))
    plt.ylabel('Test set error')
    plt.xlabel('Training set percentage')
    plt.title("Gaussian Naive-Bayes vs Logistic Regression on the {} dataset".format(dataset))
    plt.legend(('Gaussian NB', 'Logistic Regression',))
    savefig("logistic_vs_gnb_{}".format(dataset), bbox_inches='tight')

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
    plot_gnb_vs_lr_performance(filename, num_splits, train_set_percentages)
