## Project Description <br>
The goal of this project is to predict the quality of wine given 11 physicochemical properties. <br>
There are two datasets, one for red and one for white variants of the Portuguese "Vinho Verde" wine. <br>
The data was found on the UC Irvine Machine Learning Repository website. <br>
Requested Citation: <br>
Cortez,Paulo, Cerdeira,A., Almeida,F., Matos,T., and Reis,J.. (2009). Wine Quality. UCI Machine Learning Repository. <br>
https://doi.org/10.24432/C56S3T.
<br>
## Data <br>
Both sets were mostly clean, spare for some duplicate observations. <br>
The target classes (scale of 1-10) were heavily imbalanced, with a vast majority of both types of wine being grade 5 or 6. <br>
I chose to change the wine grade to a binary outcome of "good"(>=6) or "bad"(<=5).<br>
Classes were still unbalanced (458/528 for Red and 946/1952 for White in the training sets). With the difference in the Red Wine classes being not too significant, I chose to upsample with SMOTE. I initially did the same for the White Wine classes but with such a large imbalance it would have more than doubled the original samples, using "fake" data. This led to overfitting. I instead chose to stratify the sampling in my train, test, and split for White Wine, this fixed the overfitting and improved accuracy by about 8%. 
<br>
## Model <br>
Both datasets were small (Red: 1599, White: 4898 observations) and consisted of only numeric variables. <br>
The three base learning algorithms chosen were SVM, KNN, and Logistic Regression to be fed into a Stacking Classifier. <br>
<br>
## Results <br>
The Red Wine Stacker achieved 75% accuracy.
<br>
The White Wine Stacker achieved 77.4% accuracy.
<br>
