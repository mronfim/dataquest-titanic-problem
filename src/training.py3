import pandas
import numpy as np

data = pandas.read_csv("../dataset/train.csv")

# Clean null values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())
data["Embarked"] = data["Embarked"].fillna("S")

# Convert Sex to numeric
data.loc[data["Sex"] == "male", "Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1

# Convert Embarked values to numeric values
data.loc[data["Embarked"] == "S", "Embarked"] = 0
data.loc[data["Embarked"] == "C", "Embarked"] = 1
data.loc[data["Embarked"] == "Q", "Embarked"] = 2

# ----------------------------------------
# ---------- Linear Regression -----------
# ----------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = LinearRegression()
kf = KFold(data.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
	train_predictors = data[predictors].iloc[train, :]
	train_target = data["Survived"].iloc[train]
	alg.fit(train_predictors, train_target)
	test_predictions = alg.predict(data[predictors].iloc[test, :])
	predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

accuracy = sum(predictions[predictions == data["Survived"]]) /  len(predictions)
print("Linear Regression:", accuracy)


# ------------------------------------------
# ---------- Logistic Regression -----------
# ------------------------------------------
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# initialize our algorithm
alg = LogisticRegression(random_state=1)

# compute the accuracy of all cross validation folds.
scores = cross_validation.cross_val_score(alg, data[predictors], data["Survived"], cv=3)

# take the mean of the scores, because we have one for each fold
print("Logistic Regression:", scores.mean())


# -------------------------------------
# ---------- Random Forrest -----------
# -------------------------------------
from sklearn.ensemble import RandomForestClassifier

# Generating a FamilySize column
data["FamilySize"] = data["SibSp"] + data["Parch"]

# The .apply method generates a new series
data["NameLength"] = data["Name"].apply(lambda x: len(x))

import re
# A function to get the title from a Name
def get_title(name):
	# Use a regular expression to search for a title.
	# Titles always consist of capitlal and lower case letters and end with a period.
	title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""

# Get all the titles and print how often each one occurs.
titles = data["Name"].apply(get_title)
# print(pandas.value_counts(titles))

# Map each title to an integer. Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {
	"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8,
	"Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2
}
for k, v in title_mapping.items():
	titles[titles == k] = v

# Verify that we converted everything.
# print(pandas.value_counts(titles))

# Add in the title column
data["Title"] = titles

import operator

# A dictionary mapping family name to id
family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
	# Find the last name by splitting on a comma
	last_name = row["Name"].split(",")[0]
	# Create the family id
	family_id = "{0}{1}".format(last_name, row["FamilySize"])
	# Look up the id in the mapping
	if family_id not in family_id_mapping:
		if len(family_id_mapping) == 0:
			current_id = 1
		else:
			# Get the maximum id from the mapping and add one to it if we don't have an id
			current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
		family_id_mapping[family_id] = current_id
	return family_id_mapping[family_id]

# Get the family ids with the apply method
family_ids = data.apply(get_family_id, axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[data["FamilySize"] < 3] = -1

data["FamilyId"] = family_ids

### Finding the best features
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(data[predictors], data["Survived"])

# Get the row p-value from each feature, and transform from p-values into scores.
scores = -np.log10(selector.pvalues_)

# Plot the scores. See how "Pclass", "Sex", "Title", "Fare" are the best?
# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

# Pick only the fou best features.
predictors = ["Pclass", "Sex", "Fare", "Title"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)

scores = cross_validation.cross_val_score(alg, data[predictors], data["Survived"], cv=3)
print("Random Forest:", scores.mean())



# -------------------------------------------------------
# ---------- Ensembling and Gradient Boosting -----------
# -------------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

# The algorithms to ensemble.
# Using the more linear predictors for the logistic regression, and everything witht the gradient boosting classifier.
algorithms = [
	[	
		GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
		["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
	],
	[
		LogisticRegression(random_state=1),
		["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]
	]
]

# Initialize the cross validation folds
kf = KFold(data.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
	train_target = data["Survived"].iloc[train]
	full_test_predictions = []
	# Make predictions for each algorithm on each fold
	for alg, predictors in algorithms:
		# Fit the algorithm on the training data.
		alg.fit(data[predictors].iloc[train, :], train_target)
		# Select and predict on the train fold.
		# The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
		test_predictions = alg.predict_proba(data[predictors].iloc[test, :].astype(float))[:, 1]
		full_test_predictions.append(test_predictions)
	# Use  a simple ensembling scheme -- just average the predictions to get the final classification.
	test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
	# Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
	test_predictions[test_predictions <=.5] = 0
	test_predictions[test_predictions > .5] = 1
	predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the trainig data.
accuracy = sum(predictions[predictions == data["Survived"]]) / len(predictions)
print("Ensembling and Gradient Boosting:", accuracy)


# submission = pandas.DataFrame ({
#     "PassengerId": titanic_test["PassengerID"],
#     "Survived": predictions
# })