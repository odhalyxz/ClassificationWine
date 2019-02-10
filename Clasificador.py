# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,matthews_corrcoef


def k_nn(X_scaled_Train,y_train,X_scaled_Test,y_test,num_neighbors,metrica):
	#Create KNN Classifier
	knn = KNeighborsClassifier(n_neighbors=num_neighbors,metric=metrica)
	print('Configuracion clasificador:',knn)
	#Train the model using the training sets
	knn.fit(X_scaled_Train, y_train)
	#Predict the response for test dataset
	y_pred = knn.predict(X_scaled_Test)
	# Calculas las metricas MCC y otros ..........

	(MCC,CR,MC) = metricas(y_test, y_pred)

	return MCC,CR,MC


def metricas(y_test, y_pred):
	MCC = matthews_corrcoef(y_test, y_pred)
	CR = classification_report(y_test,y_pred)
	MC = confusion_matrix(y_test, y_pred)

	return MCC,CR,MC

def SplitDataSet(data, target_attribute):
	# Split dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(data, target_attribute, test_size=0.3,stratify=target_attribute,random_state=13) # 70% training and 30% test
	# Normalizacion
	X_scaled_Train = preprocessing.scale(X_train)
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_scaled_Test = scaler.transform(X_test)

	return X_scaled_Train,X_scaled_Test,y_train, y_test