import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Reading in clean data 
data = pd.read_csv("/Users/skobos/PycharmProjects/Data Science Projects/Diabetes Classifier/Diabetes Data.csv")


# Cleaning column names to be used in XGBoost (it doesn't cooperate w/ special characters in column names)
def clean_column_names(dataframe):
    dataframe.columns = dataframe.columns.str.replace('[\[\],<]', '', regex=True)


clean_column_names(data)

# Changing display options to be able to view prints better
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 1000
pd.set_option('display.width', 1000)

# Train Test Split to use RandomUnderSampler
DX = data.drop('readmitted', axis=1)
DY = data['readmitted']
DX_train, DX_test, dy_train, dy_test = train_test_split(DX, DY, test_size=.2, random_state=18)
print("\nTarget Training Distribution Before Downsampling:")
print(pd.Series(dy_train).value_counts())
print("\nTarget Testing Distribution Before Downsampling:")
print(pd.Series(dy_test).value_counts())
# Downsampling Majority Class, 0
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
DX_resampled, dy_resampled = undersampler.fit_resample(DX_train, dy_train)
print("\nTarget Distribution After Downsampling:")
print(pd.Series(dy_resampled).value_counts())
# Classes are now balanced at 4432 apiece in the training set
print("\nTarget Testing Distribution After Downsampling:")
print(pd.Series(dy_test).value_counts())
# Classes are slightly imbalanced in testing set but will not be altered 

# Parameters for GridSearch
rf_params = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'criterion': ['gini', 'entropy', 'log_loss'],
             'max_depth': [2, 4, 6, 8, 10], 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5],
             'min_weight_fraction_leaf': [0.1, 0.2, 0.3, 0.4, 0.5]}
xgb_params = {"learning_rate": [.1, .2, .3, .4, .5], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16],
              'reg_lambda': [0, 1], 'reg_alpha': [0, 1]}
gboost_params = {'loss': ['log_loss', 'exponential'], 'learning_rate': [.01, .1, 1, 10, 100],
                 'n_estimators': [50, 100, 150, 200], 'criterion': ['friedman_mse', 'squared_error']}
dt_params = {'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': [2, 4, 6, 8, 10], 'min_samples_split':
    [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5], 'min_weight_fraction_leaf': [0.1, 0.2, 0.3, 0.4, 0.5]}


# Running models to find the best parameters w/ GridSearch before building Voter
def random_forest(parameters, x_train, y_train, x_test, name):
    """grid = GridSearchCV(RandomForestClassifier(), parameters, refit=True, verbose=0)
    grid.fit(x_train, y_train)
    n_estimators = grid.best_estimator_.n_estimators
    criterion = grid.best_estimator_.criterion
    max_depth = grid.best_estimator_.max_depth
    min_samples_split = grid.best_estimator_.min_samples_split
    min_samples_leaf = grid.best_estimator_.min_samples_leaf
    min_weight_fraction_leaf = grid.best_estimator_.min_weight_fraction_leaf
    print(f"\n{name} GridSearch Best Parameters: {grid.best_params_}")"""
    random = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=2, min_samples_split=6,
                                    min_samples_leaf=3, min_weight_fraction_leaf=0.1)
    random.fit(x_train, y_train)
    global forest_testing_preds, forest_training_preds
    forest_testing_preds = random.predict(x_test)
    forest_training_preds = random.predict(x_train)


def xg_boosting(parameters, x_train, y_train, x_test, name):
    """grid = GridSearchCV(XGBClassifier(), parameters)
    grid.fit(x_train, y_train)
    print(f"\n{name} GridSearch Best Parameters: {grid.best_params_}")
    learning_rate = grid.best_estimator_.learning_rate
    max_depth = grid.best_estimator_.max_depth
    reg_lambda = grid.best_estimator_.reg_lambda
    reg_alpha = grid.best_estimator_.reg_alpha"""
    xg_boost = XGBClassifier(booster='gbtree', learning_rate=.1, max_depth=1, reg_lambda=1, reg_alpha=1)
    xg_boost.fit(x_train, y_train)
    global xgboost_testing_preds, xgboost_training_preds
    xgboost_testing_preds = xg_boost.predict(x_test)
    xgboost_training_preds = xg_boost.predict(x_train)


def gradient_boosting(parameters, x_train, y_train, x_test, name):
    """grid = GridSearchCV(GradientBoostingClassifier(), parameters)
    grid.fit(x_train, y_train)
    print(f"\n{name}  GridSearch Best Parameters: {grid.best_params_}")
    loss = grid.best_estimator_.loss
    learning_rate = grid.best_estimator_.learning_rate
    n_estimators = grid.best_estimator_.n_estimators
    criterion = grid.best_estimator_.criterion"""
    gradient_boost = GradientBoostingClassifier(loss='log_loss', learning_rate=.1, n_estimators=150,
                                                criterion='friedman_mse')
    gradient_boost.fit(x_train, y_train)
    global gboost_testing_preds, gboost_training_preds
    gboost_testing_preds = gradient_boost.predict(x_test)
    gboost_training_preds = gradient_boost.predict(x_train)
    gboost_testing_preds = gradient_boost.predict(x_test)


# Base Learners for Voting Classifier
rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=2, min_samples_split=6, min_samples_leaf=3,
                            min_weight_fraction_leaf=0.1)
xg = XGBClassifier(booster='gbtree', learning_rate=.1, max_depth=1, reg_lambda=1, reg_alpha=1)
gboost = GradientBoostingClassifier(loss='log_loss', learning_rate=.1, n_estimators=150, criterion='friedman_mse')

voting_clf = VotingClassifier(
    estimators=[('Random_Forest', rf), ('XG_Boost', xg), ('Gradient_Boost', gboost)], voting='hard')


def voting_ensemble(x_train, y_train, x_test, y_test, name):
    voting_clf.fit(x_train, y_train)
    print(f"\n{name} Accuracy Scores:")
    for clf in (rf, xg, gboost, voting_clf):
        clf.fit(x_train, y_train)
        y_testing_pred = clf.predict(x_test)
        y_training_pred = clf.predict(x_train)
        test_conf_mat = confusion_matrix(y_test, y_testing_pred)
        train_conf_mat = confusion_matrix(y_train, y_training_pred)
        print(f"{clf.__class__.__name__} Training: {accuracy_score(y_train, y_training_pred)}")
        print(f"{clf.__class__.__name__} Testing: {accuracy_score(y_test, y_testing_pred)}")
        print(f"{clf.__class__.__name__} Training Confusion Matrix: \n{train_conf_mat}")
        print(f"{clf.__class__.__name__} Testing Confusion Matrix: \n{test_conf_mat}")
    global voting_training_preds, voting_testing_preds
    voting_training_preds = voting_clf.predict(x_train)
    voting_testing_preds = voting_clf.predict(x_test)


# Model Evaluation
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


# Used for finding the best parameters w/ GridSearch before building Voter
"""# Random Forest
random_forest(rf_params, DX_resampled, dy_resampled, DX_test, "Random Forest")
evaluate_model(dy_test, dy_resampled, forest_training_preds, forest_testing_preds, "Random Forest")
# XGBoost
xg_boosting(xgb_params, DX_resampled, dy_resampled, DX_test, "XGBoost")
evaluate_model(dy_test, dy_resampled, xgboost_training_preds, xgboost_testing_preds, "XGBoost")
# GradientBoost
gradient_boosting(gboost_params, DX_resampled, dy_resampled, DX_test, "Gradient Boost")
evaluate_model(dy_test, dy_resampled, gboost_training_preds, gboost_testing_preds, "Gradient Boost")"""

# Voting Classifier
voting_ensemble(DX_resampled, dy_resampled, DX_test, dy_test, "Voting Classifier")
