# Notes
- Since there are three different algorithms all using the same data, a majority of the data cleaning and preprocessing steps are the same.

  The differences are as follows:

  ## SVM <br>
  - uses standardization with StandardScaler() from Scikitlearn <br>
  - PCA num_components are 2 for Red Wine and 6 for White Wine

  ## Logistic Regression <br>
  - uses normalization with MinMaxScaler() from Scikitlearn
  - PCA num_components are Red-4, White-6
  
  ## Linear Regression <br>
  - data is normalized with MinMaxScaler() <br>
  - quality grades are left as is (scale of 1-10) <br>
  - predictions are rounded to the nearest integer to align with the discrete quality grades
  - PCA num_components are Red- and White- 


# Assessments
## SVM 
While the Red Wine model leaves much to be desired, the White Wine model seems to have the potential to accurately classify wines as good or bad, based on the physicochemical properties included in this study. <br> 

Red Wine Model: <br>
Best Overall: 2 principal components that account for 46% of the variance, 71.6% testing accuracy, 70.0% training accuracy. 1.6% difference between the two, so there is some slight underfitting but not too bad. Increasing the number of principal components at best increased the testing accuracy by 6.7%, but led to training accuracy of 98.7%, grossly overfitting and not worth increasing the included variance. <br>
<br>
White Wine Model: <br>
Best Overall: 6 principal components that account for 82.1% of the variance, 90.0% testing accuracy, 99.8% training accuracy. 9.8% difference between 
the two, but it was the lowest difference for all #'s of PCs, and at relatively high accuracy, this moderate overfitting was accepted as the best option. Increasing the number of principal components led to little change in training or testing accuracy while decreasing the number of principal components dropped testing accuracy by as much as 10% with a little as a 2.0% decrease in training accuracy. This model also has an exceptional recall of 99.0%, so if it were to suggest a wine would be "bad", it would  not often be wrong <br>

## Logistic Regression
Both models fail to impress in terms of accuracy, however, compared to the Support Vector Machine model, the Red Wine Logistic Regression model was able to improve by almost 5%. Some credit for that may be due to the decision to normalize the data instead of standardizing, along with the use of a different algorithm. <br>

Red Wine Model:<br>
Best Overall: 4 principal components that account for 68.5% of the variance, 75% testing accuracy, 70.6% training accuracy. <br>
White Wine Model:<br>
Best Overall: 6 principal components that account for 84.5% of the variance, 72.7% testing accuracy, 71.8% training accuracy.


GridSearchCV was run to tune the hyperparameters of the models, but the model defaults proved to be the best, so the
grid search was removed.

## General
Possible explanations for the difficulty in reducing overfitting in the Red Wine SVM could be due to the lower number of observations relative to the White Wine dataset which was significantly less prone to overfitting. Red Wine had 1,136 observations after downsampling the majority class (1), while White Wine had 4,890 observations after upsampling the minority class (0). The only concern with the upsampling of the White Wine minority class is that the upsampling more than doubled the original sample size of the class, originally 1,158 samples, with 1,287 being added in the upsampling. While such a large injection of "fake" data isn't ideal, I believed it would have been better than the alternatives of either not balancing the classes, or downsampling the majority class and losing the information held by the 1,287 real observations.<br> 

To improve the models, my first suggestion would be to find more wines of grades other than 5 or 6. On a scale of 1-10, 5 and 6 are both average when it comes to grading a beverage; the class separation boundary of 5/6 was a choice made by lack of other grades. It would have been preferred to have enough observations in each class to use each integer grade as its own class, or to be able to create 3 classes of bad (1-3), average (4-6), good (7-10) to produce a more useful result using a more intuitive boundary. Additionally, if there were enough observations belonging to each class, instead of grouping the classes together, each integer could be used as its own class, leading to much more specific classifications. This would most likely lead to the implementation of a different algorithm other than an SVM due to the increase in computational costs since the multi-class SVM approach One vs. One requires n!/2!(n-2)! SVMs to classify and while One vs All is better, only requiring n SVMs be trained, there are both faster and better-suited algorithms for multi-class classification. While logistic regression would be much faster relative to the SVMs, neither of the models performed well for binary classification.







