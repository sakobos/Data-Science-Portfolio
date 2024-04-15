# 1) Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# DATA USED FROM UC IRVINE MACHINE LEARNING REPOSITORY
# link to page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
# Requested citation: Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura,
# Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on
# Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,”
# BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

# 2) Loading Data
diabetes = pd.read_csv("/Users/skobos/Documents/Data Science Portfolio/Diabetes Readmission/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv")

# 3) Data Cleaning & Preprocessing
# Changing display options to be able to view prints better
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 1000
pd.set_option('display.width', 1000)

# Exploring data
print("\nData Shape:")
print(diabetes.shape)
print("\nData Head:")
print(diabetes.head())
print("\nData Description:")
print(diabetes.describe().round(3))
print("\nData Info:")
print(diabetes.info())
print("\nData Duplicate Observations:")
duplicates = diabetes.duplicated()
print(diabetes[duplicates])
print("\nData Null Values:")
for col in diabetes.columns:
    print(f'{col}: {diabetes[col].isnull().sum()}')
print(f"\nTarget Variable Counts:")
print(diabetes['readmitted'].value_counts().sort_index())
# There are a ton of missing values, categorical variables numerically encoded with various versions of "NULL"

# Checking each column for the number of unique values and each unique value
print("\nData Unique Values:")
for col in diabetes.columns:
    print(f'{col}: # of Unique Values: {len(diabetes[col].unique())}, Unique Values: {diabetes[col].unique()}')
# Appears there are no special characters being used for NULLs other than ?

print(f"\nCount of ?'s in Each Column: ")
# Checking how many times we find ? in columns
for col in diabetes.columns:
    print(f"{col}, # of ?'s: {(diabetes[col] == '?').sum()}")
# need to remove weight (98569), payer_code (40256), medical_specialty (49949) since they have so many ?'s

# Checking how many "NULLs" are in the numerically encoded categorical columns
print(f"\nNumerically Encoded NULLs:")
print(f"discharge_disposition_id: {(diabetes['discharge_disposition_id'].isin([18, 25, 26])).sum()}")
print(f"admission_type_id: {(diabetes['admission_type_id'].isin([5, 6, 8])).sum()}")
print(f"admission_source_id: {(diabetes['admission_source_id'].isin([9, 15, 17, 20, 21])).sum()}")
# not numerically encoded but still need to count these "unknown/invalid" entries
print(f"gender: {(diabetes['gender'] == 'Unknown/Invalid').sum()}")

# Dropping a few feature columns of weight, payer_code, and medical_specialty because they have too many NaN's
# Dropping encounter_id, not particularly information rich, just a unique identifier of an encounter
diabetes = diabetes.drop(["weight", "payer_code", "medical_specialty", "max_glu_serum",  "encounter_id",
                          "patient_nbr"], axis=1)
# Takes dataframe down to (101766, 44)
print(f"\nDiabetes Dataframe Post-Drop Shape: {diabetes.shape}")

# Replacing rows with any variety of unknown values (some numeric codes are for unknown/unavailable/etc.) to NA
diabetes.loc[diabetes['race'] == '?', 'race'] = pd.NA
diabetes.loc[diabetes['diag_1'] == '?', 'diag_1'] = 0
diabetes.loc[diabetes['diag_2'] == '?', 'diag_2'] = 0
diabetes.loc[diabetes['diag_3'] == '?', 'diag_3'] = 0
diabetes.loc[diabetes["discharge_disposition_id"].isin([18, 25, 26]), 'discharge_disposition_id'] = pd.NA
diabetes.loc[diabetes["gender"] == "Unknown/Invalid", 'gender'] = pd.NA
diabetes.loc[diabetes["admission_type_id"].isin([5, 6, 8]), 'admission_type_id'] = pd.NA
diabetes.loc[diabetes["admission_source_id"].isin([9, 15, 17, 20, 21]), 'admission_source_id'] = pd.NA
# Dropping all observations with missing values
diabetes.dropna(inplace=True)
print(f"\nDiabetes Dataframe Post-Null Removal Shape: {diabetes.shape}")
# We are now left with a dataframe of shape (14387, 44)

# With the presence of majority categorical features, will use a Random Forest classifier for feature selection
# Splitting data for RF. Not using train/test sets since this is strictly for feature selection
X = diabetes.drop('readmitted', axis=1)
Y = diabetes['readmitted']
# Dummying Features to be used in the Random Forest
x_encoded = pd.get_dummies(X, columns=['race', 'gender', 'age',
                                       'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                                       'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
                                       'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                                       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                                       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                                       'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
                                       'glipizide-metformin', 'glimepiride-pioglitazone',
                                       'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
                                       'diabetesMed'])
# Replace True/False from Dummy with 1/0
x_encoded = x_encoded.astype(int)
print(f"\nShape of Encoded Dataframe: {x_encoded.shape}")
# shape increased to (14387, 1615) with the dummy variables
print(f"\nHead of Encoded Dataframe: ")
print(x_encoded.head())

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(x_encoded, Y)
# Get feature importance
importance = rf.feature_importances_
# Sorting features from most to least important
sorted_indices = np.argsort(importance)[::-1]
# Printing top 10 features
print("Top 10 Feature ranking:")
for i, idx in enumerate(sorted_indices[:10]):
    column_name = x_encoded.columns[idx]
    print(f"{i+1}. {column_name}: {importance[idx]}")
# From this RF it the features with the most importance are the numeric features (#'s 0-6)

# New DF with only the numeric features, as they were deemed most important by the RF
diabetes = diabetes[["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
                    "number_emergency", "number_inpatient", "readmitted"]]

print("\nShape of Numeric Only Dataframe:")
print(diabetes.shape)
# (14387,8) as expected
print("\nDescription of Numeric Data")
print(diabetes.describe())

# Train, Test, Split
DX = diabetes.drop('readmitted', axis=1)
DY = diabetes['readmitted']
DX_train, DX_test, dy_train, dy_test = train_test_split(DX, DY, test_size=.2, random_state=18)


# Removing outliers from training data
def no_outliers(data):
    z_scores = stats.zscore(data)
    threshold = 3
    outliers = (abs(z_scores) > threshold).any(axis=1)
    return outliers


# Dropping outlier indexes from both X and y train
dy_train = dy_train[~no_outliers(DX_train)]
DX_train = DX_train[~no_outliers(DX_train)]

# Checking target distribution for readmitted variable
print("\nTarget Readmitted Distribution Outlier-Free:")
print(dy_train.value_counts().sort_index())
# still very imbalanced (<30-947, >30-3442, NO-6230)


# Will change target to binary outcome, combining <30 and >30 into a "YES" variable for readmission
def binary_target(target):
    if target == ">30" or target == "<30":
        return 0
    else:
        return 1


dy_train = dy_train.apply(binary_target)
# Checking target distribution for readmitted variable
print("\nTraining Binary Target Readmitted Distribution:")
print(dy_train.value_counts().sort_index())
# Now 4389-0 and 6230-1
dy_test = dy_test.apply(binary_target)
print("\nTesting Binary Target Readmitted Distribution:")
print(dy_test.value_counts().sort_index())
# 1283-0 to 1595-1

# Random Forest
RF = RandomForestClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=3, random_state=18)
RF.fit(DX_train, dy_train)
rf_train_acc = RF.score(DX_train, dy_train)
rf_test_acc = RF.score(DX_test, dy_test)
print(f"\nRandom Forest Training Accuracy: {rf_train_acc}")
print(f"Random Forest Testing Accuracy: {rf_test_acc}")
