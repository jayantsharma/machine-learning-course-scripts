## Python and libraries' version info:
* python - 3.6.2
* numpy - 1.13.1
* scipy - 0.19.1

## Dataset Load:
Common utility for loading the datasets for all questions. This utility transforms the continuous target variables for the Boston dataset IF required.

Try using: `python load_data boston.csv`

## LDA
Usage: `python LDA1dProjection.py <filename> <num_crossval>`

Eg: `python LDA1dProjection boston.csv 10`

This function will plot and save the histograms in _LDA1dProjection_training.png_ and _LDA1dProjection_testing.png_.

### 2-d LDA with Gaussian generative modeling
Usage: `python LDA2dGaussGM.py <filename> <num_crossval>`

Eg: `python LDA2dGaussGM digits.csv 10`

This function will plot and save the 2-d projections of the digits dataset in _LDA2dProjectionDigits.png_.
  
## Naive-Bayes Gaussian Classifier
To run the Gaussian Naive Bayes classifier, use the command:

`python naive_bayes_gaussian.py <filename> <num_splits> <comma_separated_list_of_percentages>`

Eg: `python naive_bayes_gaussian.py boston.csv 10 10,25,50,75,100`

## Logistic Regression Classifier
To run the Logistic Regression classifier, use the command:

`python logistic_regression.py <filename> <num_splits> <comma_separated_list_of_percentages>`

Eg: `python logistic_regression.py boston.csv 10 10,25,50,75,100`

## Performance Comparison
To get plots comparing the two methods against a dataset, use the command:

`python plot_performance.py <filename> <num_splits> <comma_separated_list_of_percentages>`

Eg: `python plot_performance.py boston.csv 10 10,25,50,75,100`

This function will plot and save the error bar in _logistic\_vs\_gnb\_{dataset}.png_. For eg, for the Boston dataset, this stores it in _logistic\_vs\_gnb\_boston.png_.
  
__NOTE__: Running plot\_performance on the Digits dataset (`python plot_performance.py digits.csv`) takes 30 minutes on my i5-6200U @ 2.30 GHZ machine. Testing this using a smaller number of num\_splits might be more appropriate for grading.
