## Project Description <br>
The goal of this project is to predict hospital readmission for diabetic patients using data from 130 U.S., hospitals from years 1999-2008. <be>
Data was acquired from the UC Irvine Machine Learning Repository website. <br>
Requested Citation:
Clore,John, Cios,Krzysztof, DeShazo,Jon, and Strack,Beata. (2014). Diabetes 130-US Hospitals for Years 1999-2008. <br>
UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.
<br>
## Data <br>
Extremely messy. There were tons of missing values and features integer mapped to multiple variations of NA. <br>
Had to remove some columns to preserve enough observations for meaningful analysis (started w/ 101766 and ended w/ 13272 observations). <br>
Used a Random Forest Classifier for feature selection for its ability to handle both numeric and categorical data. The numeric features were
the most important and were selected with no categorical variables to be used in the models. <br>
Target classes of <30, >30, and 'NO' were very imbalanced, targets were changed to readmitted (1) vs not readmitted (0) then the majority class
of not readmitted was downsampled to balance the targets. <br>
## Model <br>
Used Random Forest, XGBoost, and GradientBoost classifiers to feed into a Hard Voting Classifier. 
<br>
## Results <br>
Voting Classifier accuracy: 62.9% 
<be>
## Analysis <br>
The model leaves much to be desired from a performance standpoint, which is unsurprising given the quality of the dataset. <br>
What could be one of the more important factors in the lack of predictive power of the model(s) would be the removal of two features, weight, and max_glu_serum (blood glucose level), due to the heavy presence of null values throughout the set. In diabetic patients, I believe their weight and the current state of their blood sugars would provide beneficial information on their likelihood of readmission, as both are key metrics in evaluating the management of their disease. 
