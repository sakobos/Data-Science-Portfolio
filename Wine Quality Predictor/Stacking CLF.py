# 1) Importing Libraries
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np

# 2) Loading Data
red = pd.read_csv('/Users/skobos/Documents/Data Science Portfolio/Wine Quality/wine+quality/winequality-red.csv',
                  delimiter=";")
white = pd.read_csv('/Users/skobos/Documents/Data Science Portfolio/Wine Quality/wine+quality/winequality-white.csv',
                    delimiter=";")

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
red = red.drop_duplicates()
# White Wine has 937
white = white.drop_duplicates()

print(f"\nShape of Red w/o Duplicates: {red.shape}")
# Shape is (1359, 12)
print(f"Shape of White w/o Duplicates: {white.shape}")
# Shape is (3961, 12)
# With duplicates removed and no columns having any null values we can move on to preprocessing


# 4) Data Preprocessing
# First step of preprocessing is to change that target variable to a binary outcome
# A quality grade of 5 or below will be "bad", 0. A quality grade of 6 or higher will be "good", 1.
# Binary outcome is required due to low class count for quality grades other than 5 and 6.
def change_quality(quality):
    if quality >= 6:
        return 1
    else:
        return 0


# Applying Quality Change to DFs
red['quality'] = red['quality'].apply(change_quality)
white['quality'] = white['quality'].apply(change_quality)


# Second step of preprocessing will be to split the data into training and testing sets
# Red Wine TTS
RX = red.drop('quality', axis=1)
RY = red['quality']
RX_train, RX_test, ry_train, ry_test = train_test_split(RX, RY, test_size=.2, random_state=18)
# White Wine TTS
WX = white.drop('quality', axis=1)
WY = white['quality']
WX_train, WX_test, wy_train, wy_test = train_test_split(WX, WY, test_size=.2, stratify=WY, random_state=18)


# Third step in preprocessing will be to remove outliers from the training data
def no_outliers(data):
    z_scores = stats.zscore(data)
    threshold = 3
    outliers = (abs(z_scores) > threshold).any(axis=1)
    return outliers


# Dropping outlier indexes from both X and y train
ry_train = ry_train[~no_outliers(RX_train)]
RX_train = RX_train[~no_outliers(RX_train)]

wy_train = wy_train[~no_outliers(WX_train)]
WX_train = WX_train[~no_outliers(WX_train)]


print(f"\nRed Training (No Outliers) Feature Description:")
print(RX_train.describe())
print(f"\nWhite Training (No Outliers) Feature Description:")
print(WX_train.describe())
# Trimmed data frames have shapes (986, 11) and (2902, 11) for red and white
# checking that Target Training data shapes match their corresponding Feature Training data shapes
print(f"\nShape of Red Training: {RX_train.shape}, {ry_train.shape}")
print(f"Shape of Red Training: {WX_train.shape}, {wy_train.shape}")


# Fourth step of preprocessing will be to re-explore quality distribution for each wine to check for imbalance.
print(f"\nRed Training Data Quality Grade Counts:")
print(ry_train.value_counts().sort_index())
print(f"\nWhite Training Data Quality Grade Counts:")
print(wy_train.value_counts().sort_index())
# Red Wine has a slight imbalance (458/528) 46%/54% 0/1
# White Wine has a significant imbalance (952/1950) 32%/68% 0/1

# Upsample both training datasets with SMOTE to balance out the target classes in the training data
red_upsample = SMOTE()
RX_upsample, ry_upsample = red_upsample.fit_resample(RX_train, ry_train)
print("\nRed Upsampled Class Distribution:")
print(pd.Series(ry_upsample).value_counts())
# classes balanced as expected to 528 each

# Opted to not balance White Wine classes, used Stratified Sampling instead

# Fifth step of preprocessing is scaling the training data
red_scaler = StandardScaler()
red_scaler.fit(RX_upsample)
RX_upsample_scaled = red_scaler.transform(RX_upsample)
white_scaler = StandardScaler()
white_scaler.fit(WX_train)
WX_upsample_scaled = white_scaler.transform(WX_train)

# Sixth step of preprocessing is applying PCA to the features of each training dataset
# will shoot for retaining ~90% of variance
# Red Wine PCA
red_p = RX_upsample_scaled
r_pca = PCA(n_components=7)
r_pca.fit(red_p)
red_pca = r_pca.transform(red_p)
red_pca = pd.DataFrame(red_pca)
print("\nRed Wine Data Post-PCA:")
print(red_pca.describe())
print(f"\nRed Wine PCA Explained Variance Ratio:")
print(r_pca.explained_variance_ratio_)
print("Red Wine PCA Total Variance:")
print(r_pca.explained_variance_ratio_.sum())
# 7 components retains 91.0% of the variance

# White Wine PCA
white_p = WX_upsample_scaled
w_pca = PCA(n_components=7)
w_pca.fit(white_p)
white_pca = w_pca.transform(white_p)
white_pca = pd.DataFrame(white_pca)
print("\nWhite Wine Data Post-PCA:")
print(white_pca.describe())
print(f"\nWhite Wine PCA Explained Variance Ratio:")
print(w_pca.explained_variance_ratio_)
print("White Wine PCA Total Variance:")
print(w_pca.explained_variance_ratio_.sum())
# 7 components retains 88.3% of the variance

# Ranking features by their contribution to the PCA
# The higher the loading, the more information it contributed to the PCA, the more "valuable" the feature is
# Access the loading scores for Red Wine
red_loading_scores = r_pca.components_
feature_names = red.columns
# Sum up the absolute loading scores across all components for each feature and calculate percentages
red_total_loading_scores = np.sum(np.abs(red_loading_scores), axis=0)
red_total_loading_sum = np.sum(red_total_loading_scores)
red_percentage_loading_scores = (red_total_loading_scores / red_total_loading_sum) * 100
# Rank the features based on their total contribution
red_ranked_features_indices = np.argsort(red_total_loading_scores)[::-1]
# Print the ranked features
print("\nRed Wine Features Ranked:")
print("Feature".ljust(25), "Total Loading Score".ljust(20), "Percentage")
for feature_index in red_ranked_features_indices:
    feature_name = feature_names[feature_index]
    score = red_total_loading_scores[feature_index]
    percentage = red_percentage_loading_scores[feature_index]
    print(f"{feature_name.ljust(25)} {score:.4f}".ljust(10), f"{percentage:.2f}%".rjust(22))

# Access the loading scores for White Wine
white_loading_scores = w_pca.components_
# Sum up the absolute loading scores across all components for each feature
white_total_loading_scores = np.sum(np.abs(white_loading_scores), axis=0)
white_total_loading_sum = np.sum(white_total_loading_scores)
white_percentage_loading_scores = (white_total_loading_scores / white_total_loading_sum) * 100
# Rank the features based on their total contribution
white_ranked_features_indices = np.argsort(white_total_loading_scores)[::-1]
# Print the ranked features
print("\nWhite Wine Features Ranked:")
print("Feature".ljust(25), "Total Loading Score".ljust(20), "Percentage")
for feature_index in white_ranked_features_indices:
    feature_name = feature_names[feature_index]
    score = white_total_loading_scores[feature_index]
    percentage = white_percentage_loading_scores[feature_index]
    print(f"{feature_name.ljust(25)} {score:.4f}".ljust(10), f"{percentage:.2f}%".rjust(22))

# Seventh and final step of preprocessing is applying the scaling and PCA fits from training sets onto testing sets
# Red Wine Test Scaling
RX_test_scaled = red_scaler.transform(RX_test)
# Red Wine Test PCA
red_test_pca = r_pca.transform(RX_test_scaled)
# White Wine Test Scaling
WX_test_scaled = white_scaler.transform(WX_test)
# Red Wine Test PCA
white_test_pca = w_pca.transform(WX_test_scaled)

# 5) Modeling
# Creating functions to be able to call these models twice (once for each dataset)
def svm_w_grid(parameters, x_train, y_train, x_test, name):
    """grid = GridSearchCV(SVC(), parameters, refit=True, verbose=0)
    grid.fit(x_train, y_train)
    c = grid.best_estimator_.C
    g = grid.best_estimator_.gamma
    print(f"\n{name} Wine GridSearch Best Estimators: {grid.best_estimator_}")"""
    if name == "Red SVM":
        svm = SVC(C=10, gamma=.01)
    else:
        svm = SVC(C=1, gamma=10)
    svm.fit(x_train, y_train)
    global svm_training_preds, svm_testing_preds
    svm_training_preds = svm.predict(x_train)
    svm_testing_preds = svm.predict(x_test)


# K-Nearest Neigbors (KNN)
def knn(parameters, x_train, y_train, x_test, name):
    """grid = GridSearchCV(KNeighborsClassifier(), parameters, refit=True, verbose=0)
    grid.fit(x_train, y_train)
    print(f"\n{name}  GridSearch Best Parameters: {grid.best_params_}")
    n = grid.best_estimator_.n_neighbors
    w = grid.best_estimator_.weights
    a = grid.best_estimator_.algorithm"""
    if name == "Red KNN":
        knn = KNeighborsClassifier(n_neighbors=28)
    else:
        knn = KNeighborsClassifier(n_neighbors=22)
    knn.fit(x_train, y_train)
    global knn_testing_preds, knn_training_preds
    knn_testing_preds = knn.predict(x_test)
    knn_training_preds = knn.predict(x_train)


# Logistic Regression
def logistic_regression(parameters, x_train, y_train, x_test, name):
    """grid = GridSearchCV(LogisticRegression(max_iter=2000), parameters, refit=True, verbose=0)
    grid.fit(x_train, y_train)
    penalty = grid.best_estimator_.penalty
    C = grid.best_estimator_.C
    solver = grid.best_estimator_.solver
    print(f"\n{name} GridSearch Best Parameters: {grid.best_params_}")"""
    if name == "Red LogReg":
        log = LogisticRegression(max_iter=2000, penalty='l1', C=1, solver='liblinear')
    else:
        log = LogisticRegression(max_iter=2000, penalty='l1', C=2, solver='liblinear')
    log.fit(x_train, y_train)
    global log_training_preds, log_testing_preds
    log_training_preds = log.predict(x_train)
    log_testing_preds = log.predict(x_test)


# Stacking Classifier
def stacking_clf(estimators, x_train, y_train, x_test):
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000, penalty='l1',
                                                                                       C=1, solver='liblinear'))
    clf.fit(x_train, y_train)
    global stacking_training_preds, stacking_testing_preds
    stacking_training_preds = clf.predict(x_train)
    stacking_testing_preds = clf.predict(x_test)


# 6) Model Evaluation
def evaluate_model(y_test, y_train, training_preds, testing_preds, name):
    testing_accuracy = accuracy_score(y_test, testing_preds)
    training_accuracy = accuracy_score(y_train, training_preds)
    precision = precision_score(y_test, testing_preds)
    recall = recall_score(y_test, testing_preds)
    f1 = f1_score(y_test, testing_preds)
    training_cf = confusion_matrix(y_train, training_preds)
    testing_cf = confusion_matrix(y_test, testing_preds)
    print(f"\n{name} Training Accuracy: {training_accuracy}")
    print(f"{name} Testing Accuracy: {testing_accuracy}")
    print(f"{name} Testing Precision: {precision}")
    print(f"{name} Testing Recall: {recall}")
    print(f"{name} Testing F1: {f1}")
    print('Training Confusion Matrix:')
    print(training_cf)
    print("Testing Confusion Matrix:")
    print(testing_cf)

# Parameters for GridSearch
svm_params = {'C': [.01, .1, 1,]}
knn_params = {'n_neighbors': np.arange(10, 20).tolist()}
log_params = {'penalty': ['l2'], 'C': [.01, .1, 1, 10],
              'solver': ['lbfgs']}


# Estimators for StackingClassifier
red_estimators = [
    ("svm", SVC(C=.1)),
    ("knn", KNeighborsClassifier(n_neighbors=18)),
    ("log", LogisticRegression(max_iter=2000, penalty='l2', C=.01, solver='lbfgs'))]
white_estimators = [
    ("svm", SVC(C=.1)),
    ("knn", KNeighborsClassifier(n_neighbors=20)),
    ("log", LogisticRegression(max_iter=2000, penalty='l2', C=.01, solver='lbfgs'))]


# Calling Modeling and Evaluation functions
# Red Wine SVM
# Best parameters from GridSearch, C=1, gamma=1
svm_w_grid(svm_params, RTX_train, rty_train, RTX_test, "Red SVM")
evaluate_model(rty_test, rty_train, svm_training_preds, svm_testing_preds,"Red SVM")
# Red Wine KNN
knn(knn_params, RTX_train, rty_train, RTX_test, "Red KNN")
evaluate_model(rty_test, rty_train, knn_training_preds, knn_testing_preds, "Red KNN")
# Red Wine Log Reg
logistic_regression(log_params, RTX_train, rty_train, RTX_test, "Red LogReg")
evaluate_model(rty_test, rty_train, log_training_preds, log_testing_preds,"Red LogReg")
# Red Stacking Classifier
stacking_clf(red_estimators, RTX_train, rty_train, RTX_test)
evaluate_model(rty_test, rty_train, stacking_training_preds, stacking_testing_preds, "Red Stacking")


# White Wine SVM
# Best parameters from GridSearch, C=1, gamma=10
svm_w_grid(svm_params, WTX_train, wty_train, WTX_test, "White SVM")
evaluate_model(wty_test, wty_train, svm_training_preds, svm_testing_preds, "White SVM")
# White Wine KNN
knn(knn_params, WTX_train, wty_train, WTX_test, "White KNN")
evaluate_model(wty_test, wty_train, knn_training_preds, knn_testing_preds, "White KNN")
# White Wine Log Reg
logistic_regression(log_params, WTX_train, wty_train, WTX_test, "White LogReg")
evaluate_model(wty_test, wty_train, log_training_preds, log_testing_preds, "White LogReg")
# White Stacking Classifier
stacking_clf(white_estimators, WTX_train, wty_train, WTX_test)
evaluate_model(wty_test, wty_train, stacking_training_preds, stacking_testing_preds, "White Stacking")
