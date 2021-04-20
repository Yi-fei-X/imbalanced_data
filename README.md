# imbalanced_data

#### Dataset description:  
The dataset contains credit card transactions from 2013. Due to privacy concerns, the actual features have been transformed using Principal Components Analysis (PCA). As such, nearly all of the features do not have intrinsic meaning. The only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction. The feature 'Class' is the target variable. “Class” takes a value of 1 in case of fraud and 0 otherwise. 

#### Evaluation metrics:  
For all experiments, use the following evaluation metric:  
Area under the receiver operating characteristic curve (AUC score) (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)  
The final test set performance will also be evaluated using the AUC score. 
