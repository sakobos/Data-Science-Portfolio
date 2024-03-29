# 1) Importing Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier


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
    print(f"Column: {col}, # of ?'s: {(diabetes[col] == '?').sum()}")
# need to remove weight (98569), payer_code (40256), medical_specialty (49949) since they have so many ?'s

# Checking how many "NULLs" are in the numerically encoded categorical columns
print(f"\nNumerically Encoded NULLs:")
print(f"discharge_disposition_id: {(diabetes['discharge_disposition_id'].isin([18, 25, 26])).sum()}")
print(f"admission_type_id: {(diabetes['admission_type_id'].isin([5, 6, 8])).sum()}")
print(f"admission_source_id: {(diabetes['admission_source_id'].isin([9, 15, 17, 20, 21])).sum()}")
# not numerically encoded but still need to count these "unknown/invalid" entries
print(f"gender: {(diabetes['gender'] == 'Unknown/Invalid').sum()}")

# Dropping a few feature columns of weight, payer_code, and medical_specialty because they have too many NaN's
# Dropping encounter_id and patient_nbr since they are just ID #'s and not particularly information rich.
diabetes = diabetes.drop(["weight", "payer_code", "medical_specialty", "max_glu_serum",  "encounter_id",
                          "patient_nbr"], axis=1)
# Takes dataframe down to (101766, 44)
print(f"\nDiabetes Dataframe Post-Drop Shape: {diabetes.shape}")

# Replacing rows with any variety of unknown values (some numeric codes are for unknown/unavailable/etc.) to NA
diabetes['race'].replace('?', pd.NA, inplace=True)
diabetes['diag_1'].replace("?", 0, inplace=True)
diabetes['diag_2'].replace("?", 0, inplace=True)
diabetes['diag_3'].replace("?", 0, inplace=True)
diabetes["discharge_disposition_id"].replace([18, 25, 26], pd.NA, inplace=True)
diabetes["gender"].replace("Unknown/Invalid", pd.NA, inplace=True)
diabetes["admission_type_id"].replace([5, 6, 8], pd.NA, inplace=True)
diabetes["admission_source_id"].replace([9, 15, 17, 20, 21], pd.NA, inplace=True)
# Dropping all observations with missing values
diabetes.dropna(inplace=True)
print(f"\nDiabetes Dataframe Post-Null Removal Shape: {diabetes.shape}")
# We are now left with a dataframe of shape (14387, 44)

# Rechecking distribution of target "readmitted"
print(f"\nTarget Distribution of Cleaned DF: ")
print(diabetes['readmitted'].value_counts().sort_index())
# very imbalanced (<30-1410, >30-4827, NO-8150), but will reassess after further data cleaning

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
    print(f"{i+1}. Feature {idx}: {importance[idx]}")
# From this RF it the features with the most importance are the numeric features (#'s 0-6)

# New DF with only the numeric features, as they were deemed most important by the RF
diabetes = diabetes[["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
                    "number_emergency", "number_inpatient", "readmitted"]]

print("\nShape of Numeric Only Dataframe:")
print(diabetes.shape)
# (14387,8) as expected
print("\nDescription of Numeric Data")
print(diabetes.describe())


# Separating feature/target columns to remove outliers
features = diabetes.drop('readmitted', axis=1)
target = diabetes['readmitted']
# Getting mean and standard deviation from feature columns
means = features.mean()
std_devs = features.std()
# Getting outliers
outliers = (features > (means + 3 * std_devs)) | (features < (means - 3 * std_devs))
# Dropping outliers from features df
features = features[~outliers.any(axis=1)]
print("\nShape of Outlier-Free Features: ")
print(features.shape)
# Dropping outlier observations from target column
target = target[~outliers.any(axis=1)]
print("\nShape of Outlier-Free Target: ")
print(target.shape)
# shape of target and features match at 13272, removed 1115 outliers


# Checking target distribution for readmitted variable
print("\nTarget Readmitted Distribution Outlier-Free:")
print(target.value_counts().sort_index())
# still very imbalanced (<30-1208, >30-4324, NO-7740)


# Scaling Features
# Storing column names in a list to maintain them through the standardization process
column_names = features.columns.tolist()
print(f"\nFeature Column Names: {column_names}")
scaled = StandardScaler()
scaled_features = scaled.fit_transform(features)
scaled_features = pd.DataFrame(scaled_features, columns=column_names)
print(f"\nStandardized Shape: {scaled_features.shape}")
print("\nStandardized Features Description: ")
print(scaled_features.describe())


# Concatenating Feature & Target DFs in order to downsample majority "NO" class
# Have to reset indices of target, since scaling process does so for the features
target_reset = target.reset_index(drop=True)
diabetes_final = pd.concat([scaled_features, target_reset], axis=1)
print(f"\nShape of Standardized Diabetes DF: {diabetes_final.shape}")
print("\nHead of Standardized Diabetes DF:")
print(diabetes_final.head(10))
# Changing target variable to binary outcome
# 0 will be NO, 1 will combine >30 and <30 days from readmission
diabetes_final['readmitted'] = diabetes_final['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)
print("\nStandardized Data w/ Binary Outcome Head:")
print(diabetes_final.head(10))

# Will use RandomUnderSampler on the model file; won't make sense to do it here, then recombine the dataset after TTS

# Saving Data to be used in separate model files
diabetes_final.to_csv('Diabetes Data.csv', index=False)
