# dbscan_sklearn_code
Its a simple implementation of DBSCAN algorithm suing sklearn python library.

Density-based spatial clustering of applications with noise (DBSCAN) is a data clustering algorithm proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu in 1996.It is a density-based clustering non-parametric algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away). DBSCAN is one of the most common clustering algorithms and also most cited in scientific literature. 

In this algorithm i have included GRIDSEARCH to find the optimal parameters for the the DBSCAN to give better F1 score value.

Sklearn_documentation DBSCAN-https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
Sklearn_documentation for GRIDSEARCHCv-https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
