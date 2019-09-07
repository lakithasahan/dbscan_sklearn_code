# dbscan_sklearn_code
Its a simple implementation of DBSCAN algorithm suing sklearn python library.

Density-based spatial clustering of applications with noise (DBSCAN) is a data clustering algorithm proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu in 1996.It is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away). DBSCAN is one of the most common clustering algorithms and also most cited in scientific literature. 



This algorithm consist of gridsearch function to find you the best parameters for your DBSCAN algorithm by calculating F1 score for each instances and finally selecting the best score generated parameters fully autonomously.



Dataset used- IRIS dataset(150 samples)
Sklearn_documentation DBSCAN-https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
Sklearn_documentation for GRIDSEARCHCv-https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

