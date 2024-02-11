## Notes
- Since there are three different algorithms all using the same data, a majority of the data cleaning and preprocessing steps are the same.

  The only differences are as follows:

  SVM: uses standardization with StandardScaler() from Scikitlearn, PCA num_components are 2 for Red Wine and 6 for White Wine
  Logistic Regression: uses normalization with MinMaxScaler() from Scikitlearn, PCA num_components are Red-4, White-6 <br>
  Linear Regression: data is normalized with MinMaxScaler() and quality grades are left as is (scale of 1-10)
  PCA num_components are Red- and White- 
