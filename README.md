# Project1-Oslo
Project about regression and classification.

This folder is composed of multiple files, here are some information to see clearer:

Our program is composed of two main script corresponding to each one of the task, for each one of the task.
A global boolean variable "final" indicate if we do the testing on our training set (evaluate the error) or on the testing set (final prediction). In addition, another script (loopProgram) allow us to run our predictions multiple times to do some statistics.
* mainClassification.m
* mainRegression.m
* loopProgram: Execute the prediction on the training set multiple time to evaluate the final error.

The training of each model is executed in two functions which compute the best beta parametters, using cross validation.
* trainClassificationModel.m
* trainRegressionModel.m

In addition, we have developped multiple functions to simplified the some process:
* dataTransform: Perform data transformation on the training and testing set (for instance dummy encoding, sqrt, remove categorical variables,...).
* modelSelection: For the regression task, this function will decide in which cluster each testing sample is. It is useful to apply the right model.
* outlierDetection: Not working but was suppose to detect the outliers in our testing set on the classification task

Finaly, we have the low level machine algorithm :
* leastSquare
* leastSquareGD
* ridgeRegression
* logisticRegression
* penLogisticRegression
* IRLS

Some mathematical functions:
* sigmoid
* hessian

Differents cost functions are available:
* costRMSE
* costMSE
* costClass: 1-loss
