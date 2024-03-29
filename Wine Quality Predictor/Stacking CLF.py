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
# First step in preprocessing will be to remove outliers
def no_outliers(data):
    z_scores = stats.zscore(data)
    threshold = 3
    outliers = (abs(z_scores) > threshold).any(axis=1)
    return outliers


red = red[~no_outliers(red)]

white = white[~no_outliers(white)]

print(f"\nRed (No Outliers) Data Description:")
print(red.describe())
print(f"\nWhite (No Outliers) Data Description:")
print(white.describe())
# Trimmed data frames have shapes (1232, 12) and (3620, 12) for red and white


# Second step of preprocessing is to change that target variable to a binary outcome
# A quality grade of 5 or below will be bad, 0. A quality grade of 6 or higher will be good, 1.
# Binary outcome is required due to low class count for quality grades other than 5 and 6.
def change_quality(quality):
    if quality >= 6:
        return 1
    else:
        return 0


# Applying Quality Change to DFs
red['quality'] = red['quality'].apply(change_quality)
white['quality'] = white['quality'].apply(change_quality)
# Re-exploring quality distribution for each wine to check for imbalance.
print(f"\nRed Quality Grade Counts:")
print(red['quality'].value_counts().sort_index())
print(f"\nWhite Quality Grade Counts:")
print(white['quality'].value_counts().sort_index())
# Red Wine has a slight imbalance 46%/54% 0/1
# White Wine has a significant imbalance 32%/68% 0/1
# Both datasets will require resampling to balance out the target classes

# Third step of preprocessing will be the resample the target classes
# Red Wine we will downsample the majority class since it is close, won't have to worry about losing too much info
# Splitting Red Wine dataset into class 0 and class 1 sets
red0 = red[red['quality'] == 0]
red1 = red[red['quality'] == 1]
red1_downsample = resample(red1, replace=True, n_samples=len(red0), random_state=42)
print(f"\nShape of Red 0: {red0.shape}")
print(f"Shape of Old Red 1: {red1.shape}")
print(f"Shape of New Red 1: {red1_downsample.shape}")
# Shapes match at (572, 12)
red_resampled = pd.concat([red0, red1_downsample])
print(f"Shape of Red Resampled: {red_resampled.shape}")
# Shape of Red Resampled (1144, 12) as expected


# White Wine we will upsample minority class, as downsampling majority will lose a significant amount of observations
# Splitting White Wine dataset into class 0 and class 1 sets
white0 = white[white['quality'] == 0]
white1 = white[white['quality'] == 1]
white0_upsample = resample(white0, replace=True, n_samples=len(white1), random_state=42)
print(f"\nShape of White 1: {white1.shape}")
print(f"Shape of Old White 0: {white0.shape}")
print(f"Shape of New White 0: {white0_upsample.shape}")
# Shapes match at (2450, 12)
white_resampled = pd.concat([white0_upsample, white1])
print(f"Shape of White Resampled: {white_resampled.shape}")
# Shape of White Resampled (4900, 12) as expected


# Fourth step of preprocessing is standardizing the features
red_features = red_resampled.drop('quality', axis=1)
white_features = white_resampled.drop('quality', axis=1)
# Scaling feature columns
scaler = StandardScaler()
red_stand = scaler.fit_transform(red_features)
white_stand = scaler.fit_transform(white_features)
red_stand = pd.DataFrame(red_stand)
white_stand = pd.DataFrame(white_stand)
print("\nRed Wine Standardized Data Description:")
print(red_stand.describe())
print("\nWhite Standardized Data Description:")
print(white_stand.describe())


# Fifth step of preprocessing is Principal Component Analysis (PCA) for dimensionality reduction
# Red Wine PCA
red_p = red_stand
r_pca = PCA(n_components=5)
red_pca = r_pca.fit(red_p).transform(red_p)
red_pca = pd.DataFrame(red_pca)
print("\nRed Wine Data Post-PCA:")
print(red_pca.describe())
print(f"\nRed Wine PCA Explained Variance Ratio:")
print(r_pca.explained_variance_ratio_)
print("Red Wine PCA Total Variance:")
print(r_pca.explained_variance_ratio_.sum())
# 2 components retains 46.0% of the variance
# White Wine PCA
white_p = white_stand
w_pca = PCA(n_components=8)
white_pca = w_pca.fit(white_p).transform(white_p)
white_pca = pd.DataFrame(white_pca)
print("\nWhite Wine Data Post-PCA:")
print(white_pca.describe())
print(f"\nWhite Wine PCA Explained Variance Ratio:")
print(w_pca.explained_variance_ratio_)
print("White Wine PCA Total Variance:")
print(w_pca.explained_variance_ratio_.sum())
# 6 components retains 82.2% of the variance

# Ranking features by their contribution to the PCA
# The higher the loading, the more information it contributed to the PCA, the more valuable the feature is
# Access the loading scores for Red Wine
red_loading_scores = r_pca.components_
feature_names = red.columns
# Sum up the absolute loading scores across all components for each feature
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
svm_params = {'C': [.1, 1, 10], 'gamma': [10, 1, .01, .001], 'kernel': ['rbf']}
knn_params = {'n_neighbors': np.arange(1, 40).tolist(), 'weights': ['uniform', 'distance'],
              'algorithm': ['ball_tree', 'kd_tree', 'brute']}
log_params = {'penalty': ['l1', 'l2', 'elasticnet'], 'C': [1, 2, 3, 4],
              'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']}


# Estimators for StackingClassifier
red_estimators = [
    ("svm", SVC(C=10, gamma=.01)),
    ("knn", KNeighborsClassifier(n_neighbors=28)),
    ("log", LogisticRegression(max_iter=2000, penalty='l1', C=1, solver='liblinear'))]
white_estimators = [
    ("svm", SVC(C=1, gamma=10)),
    ("knn", KNeighborsClassifier(n_neighbors=22)),
    ("log", LogisticRegression(max_iter=2000, penalty='l1', C=2, solver='liblinear'))]


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
