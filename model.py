import sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def info():
	print("Hello,\nThis model predcits posibility of survival in the event of heart attack.")
	print("Info to input\n")
	print('age => age')
	print('anaemia => Decrease of red blood cells or hemoglobin (boolean)')
	print('creatinine_phosphokinase => Level of the CPK enzyme in the blood (mcg/L)')
	print('diabetes => If the patient has diabetes (boolean)')
	print('ejection_fraction => Percentage of blood leaving the heart at each contraction (percentage)')
	print('high_blood_pressure => If the patient has hypertension (boolean)')
	print('platelets => Platelets in the blood (kiloplatelets/mL)')
	print('serum_creatinine => Level of serum creatinine in the blood (mg/dL))')
	print('serum_sodium => Level of serum sodium in the blood (mEq/L)')
	print('sex => Woman or man (binary)')
	print('smoking => If the patient smokes or not (boolean)')
	print('time => Follow-up period (days)\n')


class data:

	def __init__(self):
		self.df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
		self.Y = self.df["DEATH_EVENT"]
		self.X = self.df.iloc[:, 0:12]
		#self.X = self.X.drop(columns=['time'])
		self.X = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(self.X)
		self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.15, random_state=34)

	def pairplot(self):
		sns.pairplot(self.df, hue="DEATH_EVENT", height=3)
		plt.show()

	def __str__(self):
		return self.df


class model(data):

	def __init__(self):
		super(model, self).__init__()
		temp = input("Retrain models?\t")
		if ((temp == "T") or (temp == "t") or (temp == "true") or (temp == "True")):
			self.network()
			self.sgd()
			self.knn()
			self.logistic_regression()
			self.svc()
			self.kmeans()
			self.tree()
			self.random_forest()
			print("models saved")
		else:
			print(self.X_test.shape)
			self.load_models()
			self.predict()

	def load_models(self):
		self.network = joblib.load("MLPClassifier.joblib")
		self.sgd = joblib.load("SGD.joblib")
		self.tree = joblib.load("Tree.joblib")
		self.svc = joblib.load("SVC.joblib")
		self.kmeans = joblib.load("KMeans.joblib")
		self.random_forest = joblib.load("RandomForest.joblib")
		self.knn = joblib.load("KNN.joblib")
		self.logistic_regression = joblib.load("LogisticRegression.joblib")

		self.info()

	def info(self):
		print(self.network)
		print(self.sgd)
		print(self.tree)
		print(self.svc)
		print(self.kmeans)
		print(self.random_forest)
		print(self.knn)
		print(self.logistic_regression)
		print()

	def predict(self):
		self.dane = []
		print("data: ")
		self.dane.append(input('age\t'))
		self.dane.append(input('anaemia\t'))
		self.dane.append(input('creatinine_phosphokinase\t'))
		self.dane.append(input('diabetes\t'))
		self.dane.append(input('ejection_fraction \t'))
		self.dane.append(input('high_blood_pressure \t'))
		self.dane.append(input('platelets\t'))
		self.dane.append(input('serum_creatinine\t'))
		self.dane.append(input('serum_sodium\t'))
		self.dane.append(input('sex\t'))
		self.dane.append(input('smoking\t'))
		self.dane.append(input('time\t'))
		self.dane = np.matrix(self.dane)
		self.dane = self.dane.astype(np.float64)

		self.predictions = {}
		self.predictions["MLP"] = (np.array((self.network.predict(self.dane))))
		self.predictions["SGD"] = (np.array((self.sgd.predict(self.dane))))
		self.predictions["KNN"] = (np.array((self.knn.predict(self.dane))))
		self.predictions["LogisticRegression"] = (np.array((self.logistic_regression.predict(self.dane))))
		self.predictions["SVC"] = (np.array((self.svc.predict(self.dane))))
		self.predictions["KMeans"] = (np.array((self.kmeans.predict(self.dane))))
		self.predictions["Tree"] = (np.array((self.tree.predict(self.dane))))
		self.predictions["RandomForest"] = (np.array((self.random_forest.predict(self.dane))))

		print("predictions = ", self.predictions)

	def network(self):
		self.network = MLPClassifier(alpha=0.0001, max_iter=1000, learning_rate_init=0.001, tol=0.5, learning_rate='adaptive', hidden_layer_sizes=(4, 470, 410, 300, 311), solver='adam', activation='tanh')
		self.network.fit(self.X_train, self.Y_train)
		y_pred = self.network.predict(self.X_test)
		print("accuracy of mlp = ", metrics.accuracy_score(self.Y_test, y_pred))
		joblib.dump(self.network, "MLPClassifier.joblib")

	def sgd(self):
		self.sgd = SGDClassifier()
		self.sgd.fit(self.X_train, self.Y_train)
		y_pred = self.sgd.predict(self.X_test)
		print("accuracy of sgd = ", metrics.accuracy_score(self.Y_test, y_pred))
		joblib.dump(self.sgd, "SGD.joblib")

	def knn(self):
		self.knn = KNeighborsClassifier()
		self.knn.fit(self.X_train, self.Y_train)
		y_pred = self.knn.predict(self.X_test)
		print("accuracy of knn = ", metrics.accuracy_score(self.Y_test, y_pred))
		joblib.dump(self.knn, "KNN.joblib")

	def logistic_regression(self):
		self.logistic_regression = LogisticRegression()
		self.logistic_regression.fit(self.X_train, self.Y_train)
		y_pred = self.logistic_regression.predict(self.X_test)
		print("accuracy of logistic_regression = ", metrics.accuracy_score(self.Y_test, y_pred))
		joblib.dump(self.logistic_regression, "LogisticRegression.joblib")

	def svc(self):
		self.svc = SVC()
		self.svc.fit(self.X_train, self.Y_train)
		y_pred = self.svc.predict(self.X_test)
		print("accuracy of svc = ", metrics.accuracy_score(self.Y_test, y_pred))
		joblib.dump(self.svc, "SVC.joblib")

	def kmeans(self):
		self.kmeans = KMeans()
		self.kmeans.fit(self.X_train, self.Y_train)
		y_pred = self.kmeans.predict(self.X_test)
		print("accuracy of kmeans = ", metrics.accuracy_score(self.Y_test, y_pred))
		joblib.dump(self.kmeans, "KMeans.joblib")

	def tree(self):
		self.tree = DecisionTreeClassifier()
		self.tree.fit(self.X_train, self.Y_train)
		y_pred = self.tree.predict(self.X_test)
		print("accuracy of tree = ", metrics.accuracy_score(self.Y_test, y_pred))
		joblib.dump(self.tree, "Tree.joblib")

	def random_forest(self):
		self.random_forest = RandomForestClassifier()
		self.random_forest.fit(self.X_train, self.Y_train)
		y_pred = self.random_forest.predict(self.X_test)
		print("accuracy of random_forest = ", metrics.accuracy_score(self.Y_test, y_pred))
		joblib.dump(self.random_forest, "RandomForest.joblib")


if __name__ == '__main__':
	info()
	model = model()