import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def tunnigModel(X_scaled_Train,y_train,params,clasificador):

	grid = GridSearchCV(estimator=clasificador,param_grid=params)
	grid.fit(X_scaled_Train, y_train)
	print('best_score',grid.best_score_)
	print('best_params',grid.best_params_ )

	return grid.best_params_ , grid.best_score_


def setParams_Knn(X_scaled_Train,y_train,num_neighbors):

	params = {"n_neighbors": np.arange(1,num_neighbors), "metric": ["euclidean", "cityblock","minkowski"],"weights": ["uniform", "distance"]}
	knn = KNeighborsClassifier(n_neighbors=8,metric="cityblock")
	(best_params,best_score) = tunnigModel(X_scaled_Train,y_train,params,knn)

	return best_params,best_score
