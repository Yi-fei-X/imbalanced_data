# imbalanced_data

#### Dataset description:  
The dataset contains credit card transactions from 2013. Due to privacy concerns, the actual features have been transformed using Principal Components Analysis (PCA). As such, nearly all of the features do not have intrinsic meaning. The only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction. The feature 'Class' is the target variable. “Class” takes a value of 1 in case of fraud and 0 otherwise. 

Training set: https://wikispaces.psu.edu/download/attachments/395383213/credit_card_train.csv?api=v2  
Test set: https://wikispaces.psu.edu/download/attachments/395383213/credit_card_test.csv?api=v2

#### Evaluation metrics:  
For all experiments, use the following evaluation metric:  
Area under the receiver operating characteristic curve (AUC score) (https://scikitlearn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)  
The final test set performance will also be evaluated using the AUC score. 

#### Instructions:  
Q1) Using 5-fold cross-validation, implement a very naive baseline classifier where the majority class (no fraud) is predicted for each sample. Report the mean and standard deviation of the AUC score in a table.  

Q2) Using 5-fold cross-validation, perform hyper parameter and model selection. Evaluate each of the following model:  
Random forest  
XGBOOST  
SVM  
KNN  
Naive Bayes  
Report the following:  
Provide the hyperparameter values tried, as well as the mean and standard deviation for the AUC score. Tune the following hyperparameters:  
Random forest: n_estimators  
XGBOOST: learning rate  
SVM: c (regularization penalty)  
KNN: number of neighbors  

Q3) Retrain the models from Q2 using cross-validation. This time train each model using the SMOTE algorithm.   
Tune the same parameters as the previous section.  
Report the mean and standard deviation for the AUC score for each model with SMOTE in a table.  

Q4) Identify the best performing model from all previous questions. Using 5-fold cross validation, plot the full ROC curve (https://scikitlearn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) against each validation fold. There should be five figures in total.  

Q5) Retrain on the best performing model from all previous questions on all the training data.  Predict on the test data.  
Describe the model selection process you used.  
Which model and why?  
Did you use oversampling?  
What hyperparameter values did you select?  
Describe the performance of your chosen model and parameter on the training data.
