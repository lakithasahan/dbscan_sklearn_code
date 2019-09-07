from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import silhouette_score as sc
from sklearn.metrics import pairwise_distances,f1_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import average_precision_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.metrics import classification_report
def accuracy1(Y,labels):
	
	correct = 0
	a=[-1,0 ,1, 2 ]
	b=[0, 0, 0, 0]
	
	for x in range(len(Y)):
		if Y[x] == labels[x]:
			for i in range(len(a)):
				if Y[x]==a[i]:
					b[i]=b[i]+1
	total=0.0
	for x in range(1,len(b),1):
		total=total+b[x]	
				
	print(b)		
	return (total/len(Y))
 
 
 
def accuracy(estimator, X):
    estimator.fit(X)
    cluster_labels=estimator.labels_

   
    num_labels = len(set(cluster_labels))
    num_samples = len(target_data)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        #print(cluster_labels)
        dt=f1_score(target_data, cluster_labels,average='weighted')
        print('Accuracy -'+str(dt))
        return (dt)
 
#####################################################################################################################################
 
iris =load_iris()
input_data=iris.data
target_data=iris.target

#X_train, X_test, y_train, y_test = train_test_split(input_data1,target_data1, test_size=0.1, random_state=42)
#print(len(X_train))
#print(len(y_train))

#input_data=X_train
#target_data=y_train

poly = PolynomialFeatures(2)
input_data=poly.fit_transform(input_data)  
#print(input_data)

input_data=QuantileTransformer(n_quantiles=50, random_state=0).fit_transform(input_data)

scaler = MinMaxScaler()
 
scaler.fit(input_data)
normalised_input_data=scaler.transform(input_data)

distan=pairwise_distances(normalised_input_data,metric='euclidean')


scaler.fit(distan)
normalised_distance=scaler.transform(distan)


sscaler = StandardScaler()
sscaler.fit(normalised_distance)
normalised_distance=sscaler.transform(normalised_distance)


pca = PCA(n_components=4)
normalised_distance = pca.fit_transform(normalised_distance)

scaler.fit(normalised_distance)
normalised_distance=scaler.transform(normalised_distance)


print(normalised_distance)
print('normalised_distance')

#####################################################################################################################################


eps_values= np.arange(0.1,5 ,0.1)
min_sample_values = np.arange(2,20,1)

parameters = {'eps':eps_values, 'min_samples':min_sample_values,'metric':['manhattan','euclidean','cityblock']}
clustering = DBSCAN(algorithm='auto',n_jobs=-1)


cv = [(slice(None), slice(None))]
gs = GridSearchCV(estimator=clustering,param_grid=parameters, 
                  scoring=accuracy, cv=cv)
results=gs.fit(normalised_distance)
para=results.best_params_
print('Best parameters selected by gridsearch-'+str(para))


#################################################################################################################################
##Test plot
clustering = DBSCAN(eps=para['eps'], metric=para['metric'], min_samples=para['min_samples'],n_jobs=-1).fit(normalised_distance)

core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
labels = clustering.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
y_pred = clustering.fit_predict(normalised_distance)



plt.subplot(2, 1, 1)
plt.scatter(normalised_distance[:,0], normalised_distance[:,1],c=y_pred, cmap='Paired')
plt.title("DBSCAN predicted cluster outputs")

plt.subplot(2, 1, 2)
plt.scatter(normalised_distance[:,0], normalised_distance[:,1],c=target_data, cmap='Paired')
plt.title("Actual target outputs")

plt.tight_layout()
plt.show()



   
print(y_pred)
print(target_data) 

print(accuracy1(target_data,y_pred))
print('precision_score- '+str(precision_score(target_data,y_pred,average='weighted',labels=np.unique(y_pred))))
print('recall_score- '+str(recall_score(target_data,y_pred,average='weighted',labels=np.unique(y_pred))))
