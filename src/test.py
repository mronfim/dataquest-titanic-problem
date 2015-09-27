import pandas, support
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Get datasets
data_train 	= pandas.read_csv("../dataset/train.csv")
data_test 	= pandas.read_csv("../dataset/test.csv")

# constants
AGE_MEDIAN = data_train["Age"].median()
FARE_MEDIAN = data_train["Fare"].median()

# ------------------------------------------------------
# ---------------- Preprocess a dataset ---------------- 
# ------------------------------------------------------
def preprocess(dataset):
	# Clean null entries.
	dataset["Age"] 		= dataset["Age"].fillna(AGE_MEDIAN)
	dataset["Fare"] 	= dataset["Fare"].fillna(FARE_MEDIAN)
	dataset["Embarked"] = dataset["Embarked"].fillna("S")
	# Convert non-numeric entries.
	dataset.loc[dataset["Sex"] 		== "male", 	 "Sex"] = 0
	dataset.loc[dataset["Sex"] 		== "female", "Sex"] = 1
	dataset.loc[dataset["Embarked"] == "S", "Embarked"] = 0
	dataset.loc[dataset["Embarked"] == "C", "Embarked"] = 1
	dataset.loc[dataset["Embarked"] == "Q", "Embarked"] = 2
	# Generate a family size column (Siblings + Parent/Children).
	dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"]
	# Get all the titles and see how often each one occurs.
	titles = dataset["Name"].apply(support.get_title)
	# Map each title to an integer. Some titles are very rare, and are compressed into the same codes as other titles.
	title_mapping = {
		"Mr": 	1, 	"Miss": 2,	"Mrs": 3, 		"Master": 4, 	"Dr": 5, 	"Rev": 6, 	"Major": 7, "Col": 7, "Mlle": 8, "Mme": 8,
		"Don":	9, 	"Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, 	"Capt": 7, 	"Ms": 2, 	"Dona": 10
	}
	# Map the titles to their key number
	for k, v in title_mapping.items():
		titles[titles == k] = v
	# Add in the title column
	dataset["Title"] = titles
	family_ids = dataset.apply(support.get_family_id, axis=1)
	# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
	family_ids[dataset["FamilySize"] < 3] = -1
	dataset["FamilyId"] = family_ids


def find_best_features():
	preprocess(data_train)
	predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
	# Perform feature selection
	selector = SelectKBest(f_classif, k=5)
	selector.fit(data_train[predictors], data_train["Survived"])
	# Get the row p-value from each feature, and transform from p-values into scores.
	scores = -np.log10(selector.pvalues_)
	# Plot the scores. See how "Pclass", "Sex", "Title", "Fare" are the best?
	plt.bar(range(len(predictors)), scores)
	plt.xticks(range(len(predictors)), predictors, rotation='vertical')
	plt.show()


# -------------------------------------------------------------
# ---------------- Linear Regression Algorithm ---------------- 
# -------------------------------------------------------------
def LinearRegressionTest(train, test):
	predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
	alg = LinearRegression()
	train_predictors 	= train[predictors]
	train_target		= train["Survived"]
	alg.fit(train_predictors, train_target)
	predictions = alg.predict(test[predictors])
	predictions[predictions > .5] = 1
	predictions[predictions <= .5] = 0
	return predictions


# ---------------------------------------------------------------
# ---------------- Logistic Regression Algorithm ---------------- 
# ---------------------------------------------------------------
def LogisticRegressionTest(train, test):
	predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
	alg = LogisticRegression(random_state=1)
	train_predictors	= train[predictors]
	train_target		= train["Survived"]
	alg.fit(train_predictors, train_target)
	predictions = alg.predict(test[predictors])
	return predictions


# ---------------------------------------------------------------
# ------------------- Random Forest Algorithm ------------------- 
# ---------------------------------------------------------------
def RandomForestTest(train, test):
	predictors = ["Pclass", "Sex", "Fare", "Title"]
	# Initialize our algorithm with the default paramters
	# n_estimators is the number of trees we want to make
	# min_samples_split is the minimum number of rows we need to make a split
	# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
	alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
	train_predictors	= train[predictors]
	train_target		= train["Survived"]
	alg.fit(train_predictors, train_target)
	predictions = alg.predict(test[predictors])
	return predictions


# ---------------------------------------------------------------
# ------------------- Random Forest Algorithm ------------------- 
# ---------------------------------------------------------------
def EnsembleAndGradientBoostingTest(train, test):
	algorithms = 	[[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
					["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
					[LogisticRegression(random_state=1),
					["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]]
	train_target = train["Survived"]
	full_test_predictions = []
	for alg, predictors in algorithms:
		alg.fit(train[predictors], train_target)
		test_predictions = alg.predict_proba(test[predictors].astype(float))[:, 1]
		full_test_predictions.append(test_predictions)
	test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
	test_predictions[test_predictions <=.5] = 0
	test_predictions[test_predictions > .5] = 1
	return test_predictions

preprocess(data_train)
preprocess(data_test)
result = EnsembleAndGradientBoostingTest(data_train, data_test)
print result