
# Esto solo es para obetener los mejores parametros de cada clasificador

from readFile import *
from Clasificador import *
from tune import *


wine = readDataSet()
target_attribute = wine.target

# Split dataset into training set and test set (Normalizacion)
X_scaled_Train,X_scaled_Test,y_train, y_test = SplitDataSet(wine.data, target_attribute)
best_params,best_score = setParams_Knn(X_scaled_Train,y_train,11)
print(best_params,best_score)










print("---------------Fase Test con datos tunnig -----------------")


#% ----------------------------------



'''from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=1)
scores = cross_val_score(knn, X_scaled_Train, y_train, cv=10)
print(scores)
'''








