# ClassificationWine
The dataset used is the Wine Dataset available at UCI. This dataset has continuous features that are heterogeneous in scale due to differing properties that they measure

Configuracion clasificador: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='cityblock',
           metric_params=None, n_jobs=1, n_neighbors=7, p=2,
           weights='uniform')



# Classification report in test
             precision    recall  f1-score   support

          0       0.95      1.00      0.97        18
          1       1.00      0.95      0.98        21
          2       1.00      1.00      1.00        15

avg / total       0.98      0.98      0.98        54
MCC= 0.9725108966955406
# Confusion_matrix
[[18  0  0]
 [ 1 20  0]
 [ 0  0 15]]
