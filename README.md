## Python and libraries' version info:
* python - 3.6.2
* numpy - 1.13.1
* scipy - 0.19.1

## Dataset Load:
Common utility for loading the datasets for all questions. This utility transforms the continuous target variables for the Boston dataset IF required.
Try using: python load\_data boston.csv

## LDA
  i)  Use: python LDA1dProjection.py filename num\_crossval
        Eg: python LDA1dProjection boston.csv 10
      This function will plot and save the histogram in LDA1dProjection.png.

### 2-d LDA with Gaussian generative modeling
iii)  Use: python LDA2dGaussGM.py filename num\_crossval
        Eg: python LDA2dGaussGM digits.csv 10
      This function will plot and save the 2-d projections of the digits dataset in LDA2dProjectionDigits.png.
  

## Question. 4
To run the Gaussian Naive Bayes classifier, use the function:
  python naive\_bayes\_gaussian.py filename num\_splits comma\_separated\_list\_of\_percentages
  Eg: python naive\_bayes\_gaussian.py boston.csv 10 10,25,50,75,100

Similarly, to run the Logistic Regression classifier, use the function:
  python logistic\_regression.py filename num\_splits comma\_separated\_list\_of\_percentages
  Eg: python logistic\_regression.py boston.csv 10 10,25,50,75,100

To get plots comparing the two methods against a dataset, use the function:
  python plot\_performance.py filename num\_splits comma\_separated\_list\_of\_percentages
  Eg: python plot\_performance.py boston.csv 10 10,25,50,75,100
This function will plot and save the error bar in logistic\_vs\_gnb\_{dataset}.png. For eg, for the Boston dataset, this stores it in logistic\_vs\_gnb\_boston.png.
  
_NOTE_: Running plot\_performance on the Digits dataset (python plot\_performance.py digits.csv) takes 30 minutes on my i5-6200U @ 2.30 GHZ machine. Testing this using a smaller number of num\_splits might be more appropriate for grading.
