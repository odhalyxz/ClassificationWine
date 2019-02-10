from readFile import *
from Clasificador import *


wine = readDataSet()
target_attribute = wine.target

# Split dataset into training set and test set (Normalizacion)
X_scaled_Train,X_scaled_Test,y_train, y_test = SplitDataSet(wine.data, target_attribute)
MCC,CR,MC = k_nn(X_scaled_Train,y_train,X_scaled_Test,y_test,7,'cityblock')

print("MCC=",MCC)
print("classification_report")
print(CR)
print("confusion_matrix")
print(MC)



from sklearn.model_selection import cross_val_score
knn_cv = KNeighborsClassifier(n_neighbors=7,metric='cityblock')
scores = cross_val_score(knn_cv, X_scaled_Train, y_train, cv=10)
print(scores)