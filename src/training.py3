import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

data = pandas.read_csv("../dataset/train.cvs")

# Clean null Ages values
data["Age"] = data["Age"].fillna(data["Age"].median())

# Convert Sex to numeric
data.loc[data["Sex"] == "male", "Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1

# Clean null Embarked values
data["Embarked"] = data["Embarked"].fillna("S")
# Convert Embarked values to numeric values
data.loc[data["Embarked"] == "S", "Embarked"] = 0
data.loc[data["Embarked"] == "C", "Embarked"] = 1
data.loc[data["Embarked"] == "Q", "Embarked"] = 2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = LinearRegression()
kf = KFold(data.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
	train_predictors = (data[predictors].iloc[train, :])
	train_target = data["Survived"].iloc[train]
	alg.fit(train_predictors, train_target)
	test_predictions = alg.predict(data[predictors].iloc[test, :])
	predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

accuracy = sum(predictions[predictions == data["Survived"]]) /  len(predictions)
print(accuracy)


