from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

Grid_ = GridSearchCV()

def Linear_SVC(X_train, Y_train, X_test, Y_test):
	#~ parameters = [
	svc = LinearSVC()
	svc.fit(X_train, Y_train)
	pred = svc.predict(X_test)
	#~ print(pred)
	print("Classification Accuracy (Linear SVC) : ", classification_report(Y_test, pred))
	
def Random_Forest(X_train, Y_train, X_test, Y_test):
	rf = RandomForestClassifier()
	rf.fit(X_train, Y_train)
	pred = rf.predict(X_test)
	#~ print(pred)
	print("Classification Accuracy (Random Forest) : ", classification_report(Y_test, pred))
	
	
def Stochastic_Gradient_Descent(X_train, Y_train, X_test, Y_test):
	sgd = SGDClassifier()
	sgd.fit(TRAIN_X, TRAIN_Y)
	y_pred_sgd = sgd.predict(TEST_X)
	print("Classification Accuracy (Stochastic_Gradient_Descent) : ", classification_report(Y_test, y_pred_sgd))

def GaussianNB (X_train, Y_train, X_test, Y_test):
	GNB = GaussianNB()
	GNB.fit(TRAIN_X, TRAIN_Y)
	y_pred_GNB = GNB.predict(TEST_X)

	print("Classification Report (GaussianNB) : ",classification_report(TEST_Y, y_pred_GNB))

def SVC (X_train, Y_train, X_test, Y_test):
	svc = SVC()
	svc.fit(TRAIN_X, TRAIN_Y)
	y_pred_svc = svc.predict(TEST_X)

	print("Classification Report (svc) ",classification_report(TEST_Y, y_pred_svc))
	
		

	

