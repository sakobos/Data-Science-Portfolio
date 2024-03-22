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
I chose to change the wine grade to a binary outcome of "good"(>=6) or "bad"(<=5), then resample to balance the target classes further. <br>
<br>
## Model <br>
Both datasets were small (Red: 1599, White: 4898 observations) and consisted of only numeric variables. <br>
The three base learning algorithms chosen were SVM, KNN, and Logistic Regression to be fed into a Stacking classifier. <br>
<br>
## Results <br>
The Red Wine Stacker achieved 76.4% accuracy 
<br>
The White Wine Stacker achieved 91.0% accuracy and 99.4% recall
<br>
## Analysis <br>
While the red wine model is not terrible, it is greatly outshined by the performance of the red wine model. <br>
Both versions of the models are the same, only differing with the data given, which has a few key differences that may account for the variation in performance. <br>
First, the red wine data began with 1599 observations and ended with 1232 after preprocessing, while white wine began with 4898 and ended with 3620, almost 3x as many observations in white than red. <br>
Then, to balance the target classes, the red dataset downsampled (660->572) the majority class (1) and the white dataset upsampled (1170->2450) the minority class (0). The downsampling of the red wine set got rid of some "real" observations, while the upsampling of the white wine set injected some "fake" observations into the set which could explain some difference in prediction capability. 
