# 1) Importing Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from scipy import stats


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
# First step in preprocessing will be to remove outliers
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


# Fourth step of preprocessing is normalizing the features
red_features = red_resampled.drop('quality', axis=1)
white_features = white_resampled.drop('quality', axis=1)
# Normalizing feature columns
scaler = MinMaxScaler()
red_norm = scaler.fit_transform(red_features)
white_norm = scaler.fit_transform(white_features)
red_norm = pd.DataFrame(red_norm)
white_norm = pd.DataFrame(white_norm)

# Fifth step of preprocessing is Principal Component Analysis (PCA) for dimensionality reduction
# Red Wine PCA
red_p = red_norm
r_pca = PCA(n_components=4)
red_pca = r_pca.fit(red_p).transform(red_p)
print(f"\nRed Wine PCA Explained Variance Ratio:")
print(r_pca.explained_variance_ratio_)
print("Red Wine PCA Total Variance:")
print(r_pca.explained_variance_ratio_.sum())
# 4 components retains 68.5% of the variance
# White Wine PCA
white_p = white_norm
w_pca = PCA(n_components=6)
white_pca = w_pca.fit(white_p).transform(white_p)
print(f"\nWhite Wine PCA Explained Variance Ratio:")
print(w_pca.explained_variance_ratio_)
print("White Wine PCA Total Variance:")
print(w_pca.explained_variance_ratio_.sum())
# 6 components retains 84.5% of the variance

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
# Logistic Regression
def logistic_regression(x_train, y_train, x_test):
    log = LogisticRegression(max_iter=2000)
    log.fit(x_train, y_train)
    global training_log_preds, testing_log_preds
    training_log_preds = log.predict(x_train)
    testing_log_preds = log.predict(x_test)


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


# Red Wine Log Reg
logistic_regression(RTX_train, rty_train, RTX_test)
evaluate_model(rty_test, rty_train, training_log_preds, testing_log_preds,"Red Wine")
# White Wine Log Reg
logistic_regression(WTX_train, wty_train, WTX_test)
evaluate_model(wty_test, wty_train, training_log_preds, testing_log_preds, "White Wine")

# 7) Model Assessments
"""Red Wine Model:
Best Overall: 4 principal components (68.5% variance), 75% testing accuracy, 70.6% training accuracy."""
"""White Wine Model:
Best Overall: 6 principal components (84.5% variance), 72.7% testing accuracy, 71.8% training accuracy."""
"""GridSearchCV was run to tune hyperparameters of the model, but the model defaults proved to be the best, so the
grid search was removed."""

# 8) Conclusions
"""Both models fail to impress in terms of accuracy, however, compared to the Support Vector Machine model, the
Red Wine Logistic Regression model was able to improve almost 5%. Some credit of that may be due to the decision
to normalize the data instead of standardizing, along with the different algorithm."""
