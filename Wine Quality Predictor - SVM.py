# 1) Importing Libraries
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# 2) Loading Data
red = pd.read_csv('/Users/skobos/Downloads/wine+quality/winequality-red.csv', delimiter=";")
white = pd.read_csv('/Users/skobos/Downloads/wine+quality/winequality-white.csv', delimiter=";")

# 3) Data Cleaning
# Changing display options to be able to view prints better
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 1000
pd.set_option('display.width', 1000)


# Function to print variety of methods of data exploration
def explore(data, name):
    print(f"\n{name} Data Shape:")
    print(data.shape)
    print(f"\n{name} Data Head:")
    print(data.head())
    print(f"\n{name} Data Description:")
    print(data.describe().round(3))
    print(f"\n{name} Data Info:")
    print(data.info())
    print(f"\n{name} Data Duplicate Observations:")
    duplicates = data.duplicated()
    print(data[duplicates])
    print(f'\n{name} Data Null Values:')
    for col in data.columns:
        print(f'{col}: {data[col].isnull().sum()}')
    print(f"\n{name} Quality Grade Counts:")
    print(data['quality'].value_counts().sort_index())


explore(red, "Red Wine")
explore(white, "White Wine")

# Dropping duplicate observations
# Red Wine has 240
red_no_duplicates = red.drop_duplicates()
# White Wine has 937
white_no_duplicates = white.drop_duplicates()

print(f"Shape of Red w/o Duplicates: {red_no_duplicates.shape}")
# Shape is (1359, 12)
print(f"Shape of White w/o Duplicates: {white_no_duplicates.shape}")
# Shape is (3961, 12)

# With duplicates removed and no columns having any null values we can move on to preprocessing


# 4) Data Preprocessing
# First step in preprocessing will be to remove outliers, as SVM is sensitive since it uses distance calculations
def no_outliers(data):
    z_scores = stats.zscore(data)
    threshold = 3
    outliers = (abs(z_scores) > threshold).any(axis=1)
    return outliers


red_trim = red_no_duplicates[~no_outliers(red)]
red_trim = pd.DataFrame(red_trim)
white_trim = white_no_duplicates[~no_outliers(white)]
white_trim = pd.DataFrame(white_trim)
print(f"\nRed Trim Shape:")
print(red_trim.shape)
print(f"\nWhite Trim Shape:")
print(white_trim.shape)
# Trimmed data frames have shapes (1228, 12) and (3603, 12) for red and white


# Second step of preprocessing is to change that target variable to a binary outcome
# A quality grade of 5 or below will be bad, 0. A quality grade of 6 or higher will be good, 1.
def change_quality(quality):
    if quality >= 6:
        return 1
    else:
        return 0


# Applying Quality Change to DFs
red_trim['quality'] = red_trim['quality'].apply(change_quality)
white_trim['quality'] = white_trim['quality'].apply(change_quality)
# Re-exploring quality distribution for each wine to check for imbalance.
print(f"\nRed Trim Quality Grade Counts:")
print(red_trim['quality'].value_counts().sort_index())
print(f"\nWhite Trim Quality Grade Counts:")
print(white_trim['quality'].value_counts().sort_index())
# Red Wine has a slight imbalance 46%/54% 0/1
# White Wine has a significant imbalance 32%/68% 0/1
# Will require resampling to balance out the target classes

# Third step of preprocessing will be the resample the target classes
# Red Wine we will downsample the majority class since it is close, won't have to worry about losing too much info
# Splitting Red Wine dataset into class 0 and class 1 sets
red0 = red_trim[red_trim['quality'] == 0]
red1 = red_trim[red_trim['quality'] == 1]
red1_downsample = resample(red1, replace=True, n_samples=len(red0), random_state=42)
print(f"\nShape of Red 0: {red0.shape}")
print(f"Shape of Old Red 1: {red1.shape}")
print(f"Shape of New Red 1: {red1_downsample.shape}")
# Shapes match at (568, 12)
red_resampled = pd.concat([red0, red1_downsample])
print(f"Shape of Red Resampled: {red_resampled.shape}")
# Shape of Red Resampled (1136, 12) as expected

# White Wine we will upsample minority class, as downsampling majority will lose a significant amount of observations
# Splitting White Wine dataset into class 0 and class 1 sets
white0 = white_trim[white_trim['quality'] == 0]
white1 = white_trim[white_trim['quality'] == 1]
white0_upsample = resample(white0, replace=True, n_samples=len(white1), random_state=42)
print(f"\nShape of White 1: {white1.shape}")
print(f"Shape of Old White 0: {white0.shape}")
print(f"Shape of New White 0: {white0_upsample.shape}")
# Shapes match at (2445, 12)
white_resampled = pd.concat([white0_upsample, white1])
print(f"Shape of White Resampled: {white_resampled.shape}")
# Shape of White Resampled (4890, 12) as expected


# Fourth step of preprocessing is standardizing the features
red_features = red_resampled.drop('quality', axis=1)
white_features = white_resampled.drop('quality', axis=1)
# Scaling feature columns
scaler = StandardScaler()
red_stand = scaler.fit_transform(red_features)
white_stand = scaler.fit_transform(white_features)
red_stand = pd.DataFrame(red_stand)
white_stand = pd.DataFrame(white_stand)

# Fifth step of preprocessing is Principal Component Analysis (PCA) for dimensionality reduction
# Red Wine PCA
red_p = red_stand
r_pca = PCA(n_components=2)
red_pca = r_pca.fit(red_p).transform(red_p)
print(f"\nRed Wine PCA Explained Variance Ratio:")
print(r_pca.explained_variance_ratio_)
print("Red Wine PCA Total Variance:")
print(r_pca.explained_variance_ratio_.sum())
# 2 components retains 46.0% of the variance
# White Wine PCA
white_p = white_stand
w_pca = PCA(n_components=5)
white_pca = w_pca.fit(white_p).transform(white_p)
print(f"\nWhite Wine PCA Explained Variance Ratio:")
print(w_pca.explained_variance_ratio_)
print("White Wine PCA Total Variance:")
print(w_pca.explained_variance_ratio_.sum())
# 6 components retains 81.9% of the variance

# Final step of preprocessing is splitting the data into training and testing sets
# Red Wine TTS
RTX = red_pca
RTY = red_resampled['quality']
RTX_train, RTX_test, rty_train, rty_test = train_test_split(RTX, RTY, test_size=.2, random_state=18)
# White Wine TTS
WTX = white_pca
WTY = white_resampled['quality']
WTX_train, WTX_test, wty_train, wty_test = train_test_split(WTX, WTY, test_size=.2, random_state=18)


# 5) Modeling
# Support Vector Machine (SVM)
def svm_w_grid(parameters, x_train, y_train, x_test, name):
    grid = GridSearchCV(SVC(), parameters, refit=True, verbose=0)
    grid.fit(x_train, y_train)
    c = grid.best_estimator_.C
    g = grid.best_estimator_.gamma
    print(f"\n{name} Wine GridSearch Best Estimators: {grid.best_estimator_}")
    svm = SVC(C=c, gamma=g)
    svm.fit(x_train, y_train)
    global training_svm_preds, testing_svm_preds
    training_svm_preds = svm.predict(x_train)
    testing_svm_preds = svm.predict(x_test)


# 6) Model Evaluation
def evaluate_model(y_test, y_train, training_preds, testing_preds, name):
    testing_accuracy = accuracy_score(y_test, testing_preds)
    training_accuracy = accuracy_score(y_train, training_preds)
    precision = precision_score(y_test, testing_preds)
    recall = recall_score(y_test, testing_preds)
    f1 = f1_score(y_test, testing_preds)
    training_cf = confusion_matrix(y_train, training_preds)
    testing_cf = confusion_matrix(y_test, testing_preds)
    print(f"{name} Training Accuracy: {training_accuracy}")
    print(f"{name} Testing Accuracy: {testing_accuracy}")
    print(f"{name} Testing Precision: {precision}")
    print(f"{name} Testing Recall: {recall}")
    print(f"{name} Testing F1: {f1}")
    print('Training Confusion Matrix:')
    print(training_cf)
    print("Testing Confusion Matrix:")
    print(testing_cf)


params = {'C': [.1, 1, 10], 'gamma': [10, 1, .01, .001], 'kernel': ['rbf']}

# Red Wine SVM
svm_w_grid(params, RTX_train, rty_train, RTX_test, "Red Wine")
evaluate_model(rty_test, rty_train, training_svm_preds, testing_svm_preds,"Red Wine")
# White Wine SVM
svm_w_grid(params, WTX_train, wty_train, WTX_test, "White Wine")
evaluate_model(wty_test, wty_train, training_svm_preds, testing_svm_preds, "White Wine")

# 7) Model Assessments
"""Red Wine Model:
Best Overall: 2 principal components (46% variance), 70.2% testing accuracy, 73.5% training accuracy. 3.3% 
difference between the two, so there is some slight overfitting but not too bad. Increasing the number of principal
components at best increased the testing accuracy by 6.6%, but increased the overfitting by almost 10%, not worth
increasing the included variance. 
"""
"""White Wine Model:
Best Overall Model: 5 PC's (74.9% variance), 90.3% testing accuracy, 99.6% training accuracy. 9.3% difference between 
the two, but it was the lowest difference for all #'s of PC's and at relatively high accuracy, this moderate overfitting
was accepted. This model also has exceptional recall of 97.1%, so if it were to suggest a wine would be "bad", it would 
not often be wrong"""

"""Possible explanations for the difficulty in reducing overfitting in the Red Wine model could be due to the lower
of observations relative to the White Wine dataset that was significantly less prone to overfitting. Red Wine had 1,136
observations after downsampling the majority class (1), while White Wine had 4,890 observations after upsampling the 
minority class (0). The only concern with the upsampling of the White Wine minority class is that the upsampling more
than doubled the original sample size of the class, originally 1,158 samples, 1,287 were added in the upsampling. While
such a large injection of "fake" data isn't ideal, I believed it would have been better than the alternatives of either 
not balancing the classes, or to downsample the majority class and lose the information held by the 1,287 real 
observations."""


# 8) Conclusions
"""While the Red Wine model leaves much to be desired, the White Wine model seems to have potential to accurately
classify wines as good or bad, based on the physicochemical properties included in this study. 

To improve both models, increasing the number of samples would be the first suggestion. More high-quality data can only 
help the models learn. My second suggestion would be to find more wines of grades other than 5 or 6. On a scale of 1-10,
5 and 6 are both average when it comes to grading a beverage; the class separation boundary of 5/6 was a choice made by
lack of other grades. It would have been preferred to have enough observations in each class to use each integer grade 
as its own class, or to be able to create 3 classes of bad (1-3), average (4-6), good (7-10) to produce a more useful 
result using a more intuitive boundary. Additionally, if there were enough observations belonging to each class, instead
 of grouping the classes together, each integer could be used as its own class, leading to much more specific 
 classifications. Including additional physicochemical properties to the observations could also benefit, similarly to 
 increasing the number of observations, the more information the better."""


