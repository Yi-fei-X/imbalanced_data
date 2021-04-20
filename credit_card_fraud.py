import numpy as np
import csv
import random
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from prettytable import PrettyTable
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)

#read input files
def loadCSV(filename):
    dataSet=[]
    with open(filename,'r') as file:
        csvReader=csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet

A = loadCSV('credit_card_train.csv')
A.pop(0)
#random.shuffle(A)
A_array = np.array(A,dtype='float64')
A_array_df = pd.DataFrame(A_array)
A_array_df_sample = A_array_df.sample(frac=0.2, random_state=0) #random sample 20% of data
A_array_sample = np.array(A_array_df_sample)

train_val_X = A_array_sample[:,0:-1]
train_val_y = A_array_sample[:,-1]

B = loadCSV('credit_card_test.csv')
B.pop(0)
len_test = len(B)
B_array = np.array(B,dtype='float64')

kf = KFold(n_splits=5, shuffle=True, random_state=0)  #5fold cross validation

#Q1 naive baseline classifier
train_acc = []
val_acc = []
train_auc = []
val_auc = []
train_auc_std = []
val_auc_std = []

for train_index, val_index in kf.split(train_val_X):
    ##########################
    train_X = train_val_X[train_index, :]
    val_X = train_val_X[val_index, :]

    train_y = train_val_y[train_index]
    val_y = train_val_y[val_index]

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train_X, train_y)

    train_acc.append(dummy_clf.score(train_X, train_y))
    val_acc.append(dummy_clf.score(val_X, val_y))

    train_pred_prob = dummy_clf.predict_proba(train_X)
    train_pred_prob = train_pred_prob[:,0]
    val_pred_prob = dummy_clf.predict_proba(val_X)
    val_pred_prob = val_pred_prob[:,0]

    train_auc.append(roc_auc_score(train_y,train_pred_prob))
    val_auc.append(roc_auc_score(val_y,val_pred_prob))
    ##########################
avg_train_acc = sum(train_acc) / len(train_acc)
avg_val_acc = sum(val_acc) / len(val_acc)
avg_train_auc = sum(train_auc) / len(train_auc)
avg_val_auc = sum(val_auc) / len(val_auc)
train_auc_std = np.std(train_auc)
val_auc_std = np.std(val_auc)

print("Training accuracy: ", avg_train_acc * 100, "%")
print("Validation accuracy: ", avg_val_acc * 100, "%")
print("Training AUC: ", avg_train_auc)
print("Validation AUC: ", avg_val_auc)
print("Training AUC std: ", train_auc_std)
print("Validation AUC std: ", val_auc_std)

#From a table
Table_Q1 = PrettyTable()
Table_Q1_title = ['Data', 'Accuracy', 'AUC mean', 'AUC std']
Table_Q1.field_names = Table_Q1_title
Table_Q1.add_row(['Training Data',avg_train_acc, avg_train_auc, train_auc_std])
Table_Q1.add_row(['Validation Data', avg_val_acc, avg_val_auc, val_auc_std])
print(Table_Q1)

#Q2
#Random forest
def Random_forest (train_val_X, train_val_y, smote):
    num_trees = np.arange(10,200,20) #np.arange(10,300,10)
    train_acc_forest = []
    val_acc_forest = []
    train_auc_forest = []
    val_auc_forest = []
    train_std_forest = []
    val_std_forest = []

    for num_tree in num_trees:
        train_acc = []
        val_acc = []
        train_auc = []
        val_auc = []
        train_auc_std = []
        val_auc_std = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################

            train_X = train_val_X[train_index, :]
            train_y = train_val_y[train_index]

            if smote == 1:
                smo = SMOTE(random_state=0, sampling_strategy=0.2)
                train_X, train_y = smo.fit_sample(train_X, train_y)

            val_X = train_val_X[val_index, :]
            val_y = train_val_y[val_index]

            dtc = RandomForestClassifier(n_estimators=num_tree)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))

            train_pred_prob = dtc.predict_proba(train_X)
            train_pred_prob = train_pred_prob[:, 1]
            val_pred_prob = dtc.predict_proba(val_X)
            val_pred_prob = val_pred_prob[:, 1]

            train_auc.append(roc_auc_score(train_y, train_pred_prob))
            val_auc.append(roc_auc_score(val_y, val_pred_prob))
            ##########################

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        avg_train_auc = sum(train_auc) / len(train_auc)
        avg_val_auc = sum(val_auc) / len(val_auc)

        train_auc_std = np.std(train_auc)
        val_auc_std = np.std(val_auc)

        print("Number of trees", num_tree)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")
        print("Training AUC: ", avg_train_auc)
        print("Validation AUC: ", avg_val_auc)
        print("Training AUC std: ", train_auc_std)
        print("Validation AUC std: ", val_auc_std)

        train_acc_forest.append(avg_train_acc)
        val_acc_forest.append(avg_val_acc)
        train_auc_forest.append(avg_train_auc)
        val_auc_forest.append(avg_val_auc)
        train_std_forest.append(train_auc_std)
        val_std_forest.append(val_auc_std)

    return train_acc_forest, val_acc_forest, train_auc_forest, val_auc_forest, train_std_forest, val_std_forest, num_trees

train_acc_forest, val_acc_forest, train_auc_forest, val_auc_forest, train_std_forest, val_std_forest, num_trees = Random_forest(train_val_X, train_val_y, 0)

#Form a table
num_trees = num_trees.tolist()
Table_Q2_forest = PrettyTable()
Table_Q2_forest_title = num_trees.copy()
Table_Q2_forest_title.insert(0,'type/number of trees')
Table_Q2_forest.field_names = Table_Q2_forest_title
Table_Q2_train_acc_forest = train_acc_forest.copy()
Table_Q2_train_acc_forest.insert(0,"train_acc_forest")
Table_Q2_forest.add_row(Table_Q2_train_acc_forest)
Table_Q2_val_acc_forest = val_acc_forest.copy()
Table_Q2_val_acc_forest.insert(0,"val_acc_forest")
Table_Q2_forest.add_row(Table_Q2_val_acc_forest)
Table_Q2_train_auc_forest = train_auc_forest.copy()
Table_Q2_train_auc_forest.insert(0,"train auc forest")
Table_Q2_forest.add_row(Table_Q2_train_auc_forest)
Table_Q2_val_auc_forest = val_auc_forest.copy()
Table_Q2_val_auc_forest.insert(0,"val auc forest")
Table_Q2_forest.add_row(Table_Q2_val_auc_forest)
Table_Q2_train_std_forest = train_std_forest.copy()
Table_Q2_train_std_forest.insert(0,"train std forest")
Table_Q2_forest.add_row(Table_Q2_train_std_forest)
Table_Q2_val_std_forest = val_std_forest.copy()
Table_Q2_val_std_forest.insert(0,"val std forest")
Table_Q2_forest.add_row(Table_Q2_val_std_forest)
print(Table_Q2_forest)

#Q2 XGBOOST
def XGBOOST(train_val_X, train_val_y, smote):
    etas = np.arange(0.1,1,0.2) #np.arange(0.1,1,0.2)
    train_acc_xgboost = []
    val_acc_xgboost = []
    train_auc_xgboost = []
    val_auc_xgboost = []
    train_std_xgboost = []
    val_std_xgboost = []

    for eta in etas:
        train_acc = []
        val_acc = []
        train_auc = []
        val_auc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            train_y = train_val_y[train_index]

            if smote == 1:
                smo = SMOTE(random_state=0, sampling_strategy=0.2)
                train_X, train_y = smo.fit_sample(train_X, train_y)

            val_X = train_val_X[val_index, :]
            val_y = train_val_y[val_index]

            dtc = xgb.XGBClassifier(eta=eta)
            dtc.fit(train_X, train_y)
            train_acc.append(dtc.score(train_X, train_y))
            val_acc.append(dtc.score(val_X, val_y))

            train_pred_prob = dtc.predict_proba(train_X)
            train_pred_prob = train_pred_prob[:, 1]
            val_pred_prob = dtc.predict_proba(val_X)
            val_pred_prob = val_pred_prob[:, 1]

            train_auc.append(roc_auc_score(train_y, train_pred_prob))
            val_auc.append(roc_auc_score(val_y, val_pred_prob))
            ##########################

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        avg_train_auc = sum(train_auc) / len(train_auc)
        avg_val_auc = sum(val_auc) / len(val_auc)

        train_auc_std = np.std(train_auc)
        val_auc_std = np.std(val_auc)

        print("Number of eta", eta)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")
        print("Training AUC: ", avg_train_auc)
        print("Validation AUC: ", avg_val_auc)
        print("Training AUC std: ", train_auc_std)
        print("Validation AUC std: ", val_auc_std)

        train_acc_xgboost.append(avg_train_acc)
        val_acc_xgboost.append(avg_val_acc)
        train_auc_xgboost.append(avg_train_auc)
        val_auc_xgboost.append(avg_val_auc)
        train_std_xgboost.append(train_auc_std)
        val_std_xgboost.append(val_auc_std)

    return train_acc_xgboost, val_acc_xgboost, train_auc_xgboost, val_auc_xgboost, train_std_xgboost, val_std_xgboost, etas
train_acc_xgboost, val_acc_xgboost, train_auc_xgboost, val_auc_xgboost, train_std_xgboost, val_std_xgboost, etas = XGBOOST(train_val_X, train_val_y, 0)

#Form a table
etas = etas.tolist()
Table_Q2_xgboost = PrettyTable()
Table_Q2_xgboost_title = etas.copy()
Table_Q2_xgboost_title.insert(0,'type/etas')
Table_Q2_xgboost.field_names = Table_Q2_xgboost_title
Table_Q2_train_acc_xgboost = train_acc_xgboost.copy()
Table_Q2_train_acc_xgboost.insert(0,"train_acc_xgboost")
Table_Q2_xgboost.add_row(Table_Q2_train_acc_xgboost)
Table_Q2_val_acc_xgboost = val_acc_xgboost.copy()
Table_Q2_val_acc_xgboost.insert(0,"val_acc_xgboost")
Table_Q2_xgboost.add_row(Table_Q2_val_acc_xgboost)
Table_Q2_train_auc_xgboost = train_auc_xgboost.copy()
Table_Q2_train_auc_xgboost.insert(0,"train auc xgboost")
Table_Q2_xgboost.add_row(Table_Q2_train_auc_xgboost)
Table_Q2_val_auc_xgboost = val_auc_xgboost.copy()
Table_Q2_val_auc_xgboost.insert(0,"val auc xgboost")
Table_Q2_xgboost.add_row(Table_Q2_val_auc_xgboost)
Table_Q2_train_std_xgboost = train_std_xgboost.copy()
Table_Q2_train_std_xgboost.insert(0,"train std xgboost")
Table_Q2_xgboost.add_row(Table_Q2_train_std_xgboost)
Table_Q2_val_std_xgboost = val_std_xgboost.copy()
Table_Q2_val_std_xgboost.insert(0,"val std xgboost")
Table_Q2_xgboost.add_row(Table_Q2_val_std_xgboost)
print(Table_Q2_xgboost)

#Q2 SVM
def SVM(train_val_X, train_val_y, smote):
    C_svms = np.arange(0.1,1,0.2) #np.arange(0.1,1,0.2)
    train_acc_SVM = []
    val_acc_SVM = []
    train_auc_SVM = []
    val_auc_SVM = []
    train_std_SVM = []
    val_std_SVM = []

    for C_svm in C_svms:
        train_acc = []
        val_acc = []
        train_auc = []
        val_auc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            train_y = train_val_y[train_index]

            if smote == 1:
                smo = SMOTE(random_state=0, sampling_strategy=0.2)
                train_X, train_y = smo.fit_sample(train_X, train_y)

            val_X = train_val_X[val_index, :]
            val_y = train_val_y[val_index]

            SVM = svm.SVC(C=C_svm, probability=True)
            SVM.fit(train_X, train_y)
            train_acc.append(SVM.score(train_X, train_y))
            val_acc.append(SVM.score(val_X, val_y))

            train_pred_prob = SVM.predict_proba(train_X)
            train_pred_prob = train_pred_prob[:, 1]
            val_pred_prob = SVM.predict_proba(val_X)
            val_pred_prob = val_pred_prob[:, 1]

            train_auc.append(roc_auc_score(train_y, train_pred_prob))
            val_auc.append(roc_auc_score(val_y, val_pred_prob))
            ##########################

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        avg_train_auc = sum(train_auc) / len(train_auc)
        avg_val_auc = sum(val_auc) / len(val_auc)

        train_auc_std = np.std(train_auc)
        val_auc_std = np.std(val_auc)

        print("Number of C", C_svm)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")
        print("Training AUC: ", avg_train_auc)
        print("Validation AUC: ", avg_val_auc)
        print("Training AUC std: ", train_auc_std)
        print("Validation AUC std: ", val_auc_std)

        train_acc_SVM.append(avg_train_acc)
        val_acc_SVM.append(avg_val_acc)
        train_auc_SVM.append(avg_train_auc)
        val_auc_SVM.append(avg_val_auc)
        train_std_SVM.append(train_auc_std)
        val_std_SVM.append(val_auc_std)

    return train_acc_SVM, val_acc_SVM, train_auc_SVM, val_auc_SVM, train_std_SVM, val_std_SVM, C_svms
train_acc_SVM, val_acc_SVM, train_auc_SVM, val_auc_SVM, train_std_SVM, val_std_SVM, C_svms = SVM(train_val_X, train_val_y, 0)

#Form a table
C_svms = C_svms.tolist()
Table_Q2_SVM = PrettyTable()
Table_Q2_SVM_title = C_svms.copy()
Table_Q2_SVM_title.insert(0,'type/C')
Table_Q2_SVM.field_names = Table_Q2_SVM_title
Table_Q2_train_acc_SVM = train_acc_SVM.copy()
Table_Q2_train_acc_SVM.insert(0,"train_acc_SVM")
Table_Q2_SVM.add_row(Table_Q2_train_acc_SVM)
Table_Q2_val_acc_SVM = val_acc_SVM.copy()
Table_Q2_val_acc_SVM.insert(0,"val_acc_SVM")
Table_Q2_SVM.add_row(Table_Q2_val_acc_SVM)
Table_Q2_train_auc_SVM = train_auc_SVM.copy()
Table_Q2_train_auc_SVM.insert(0,"train auc SVM")
Table_Q2_SVM.add_row(Table_Q2_train_auc_SVM)
Table_Q2_val_auc_SVM = val_auc_SVM.copy()
Table_Q2_val_auc_SVM.insert(0,"val auc SVM")
Table_Q2_SVM.add_row(Table_Q2_val_auc_SVM)
Table_Q2_train_std_SVM = train_std_SVM.copy()
Table_Q2_train_std_SVM.insert(0,"train std SVM")
Table_Q2_SVM.add_row(Table_Q2_train_std_SVM)
Table_Q2_val_std_SVM = val_std_SVM.copy()
Table_Q2_val_std_SVM.insert(0,"val std SVM")
Table_Q2_SVM.add_row(Table_Q2_val_std_SVM)
print(Table_Q2_SVM)

#Q2 KNN
def KNN(train_val_X, train_val_y, smote):
    num_neighbors = np.arange(1,20,2) #np.arange(1,20,2)
    train_acc_KNN = []
    val_acc_KNN = []
    train_auc_KNN = []
    val_auc_KNN = []
    train_std_KNN = []
    val_std_KNN = []

    for num_neighbor in num_neighbors:
        train_acc = []
        val_acc = []
        train_auc = []
        val_auc = []
        for train_index, val_index in kf.split(train_val_X):
            ##########################
            train_X = train_val_X[train_index, :]
            train_y = train_val_y[train_index]

            if smote == 1:
                smo = SMOTE(random_state=0, sampling_strategy=0.2)
                train_X, train_y = smo.fit_sample(train_X, train_y)

            val_X = train_val_X[val_index, :]
            val_y = train_val_y[val_index]

            knn = KNeighborsClassifier(num_neighbor)
            knn.fit(train_X, train_y)
            train_acc.append(knn.score(train_X, train_y))
            val_acc.append(knn.score(val_X, val_y))

            train_pred_prob = knn.predict_proba(train_X)
            train_pred_prob = train_pred_prob[:, 1]
            val_pred_prob = knn.predict_proba(val_X)
            val_pred_prob = val_pred_prob[:, 1]

            train_auc.append(roc_auc_score(train_y, train_pred_prob))
            val_auc.append(roc_auc_score(val_y, val_pred_prob))
            ##########################

        avg_train_acc = sum(train_acc) / len(train_acc)
        avg_val_acc = sum(val_acc) / len(val_acc)

        avg_train_auc = sum(train_auc) / len(train_auc)
        avg_val_auc = sum(val_auc) / len(val_auc)

        train_auc_std = np.std(train_auc)
        val_auc_std = np.std(val_auc)

        print("Number of neighbors", num_neighbor)
        print("Training accuracy: ", avg_train_acc * 100, "%")
        print("Validation accuracy: ", avg_val_acc * 100, "%")
        print("Training AUC: ", avg_train_auc)
        print("Validation AUC: ", avg_val_auc)
        print("Training AUC std: ", train_auc_std)
        print("Validation AUC std: ", val_auc_std)

        train_acc_KNN.append(avg_train_acc)
        val_acc_KNN.append(avg_val_acc)
        train_auc_KNN.append(avg_train_auc)
        val_auc_KNN.append(avg_val_auc)
        train_std_KNN.append(train_auc_std)
        val_std_KNN.append(val_auc_std)

    return train_acc_KNN, val_acc_KNN, train_auc_KNN, val_auc_KNN, train_std_KNN, val_std_KNN, num_neighbors
train_acc_KNN, val_acc_KNN, train_auc_KNN, val_auc_KNN, train_std_KNN, val_std_KNN, num_neighbors = KNN(train_val_X, train_val_y, 0)

#Form a table
num_neighbors = num_neighbors.tolist()
Table_Q2_KNN = PrettyTable()
Table_Q2_KNN_title = num_neighbors.copy()
Table_Q2_KNN_title.insert(0,'type/number of neighbors')
Table_Q2_KNN.field_names = Table_Q2_KNN_title
Table_Q2_train_acc_KNN = train_acc_KNN.copy()
Table_Q2_train_acc_KNN.insert(0,"train_acc_KNN")
Table_Q2_KNN.add_row(Table_Q2_train_acc_KNN)
Table_Q2_val_acc_KNN = val_acc_KNN.copy()
Table_Q2_val_acc_KNN.insert(0,"val_acc_KNN")
Table_Q2_KNN.add_row(Table_Q2_val_acc_KNN)
Table_Q2_train_auc_KNN = train_auc_KNN.copy()
Table_Q2_train_auc_KNN.insert(0,"train auc KNN")
Table_Q2_KNN.add_row(Table_Q2_train_auc_KNN)
Table_Q2_val_auc_KNN = val_auc_KNN.copy()
Table_Q2_val_auc_KNN.insert(0,"val auc KNN")
Table_Q2_KNN.add_row(Table_Q2_val_auc_KNN)
Table_Q2_train_std_KNN = train_std_KNN.copy()
Table_Q2_train_std_KNN.insert(0,"train std KNN")
Table_Q2_KNN.add_row(Table_Q2_train_std_KNN)
Table_Q2_val_std_KNN = val_std_KNN.copy()
Table_Q2_val_std_KNN.insert(0,"val std KNN")
Table_Q2_KNN.add_row(Table_Q2_val_std_KNN)
print(Table_Q2_KNN)

#Q2 Naive Bayes
def NB(train_val_X, train_val_y, smote):
    train_acc = []
    val_acc = []
    train_auc = []
    val_auc = []
    for train_index, val_index in kf.split(train_val_X):
        ##########################
        train_X = train_val_X[train_index, :]
        train_y = train_val_y[train_index]

        if smote == 1:
            smo = SMOTE(random_state=0, sampling_strategy=0.2)
            train_X, train_y = smo.fit_sample(train_X, train_y)

        val_X = train_val_X[val_index, :]
        val_y = train_val_y[val_index]

        gnb = GaussianNB()
        gnb.fit(train_X, train_y)
        train_acc.append(gnb.score(train_X, train_y))
        val_acc.append(gnb.score(val_X, val_y))

        train_pred_prob = gnb.predict_proba(train_X)
        train_pred_prob = train_pred_prob[:, 1]
        val_pred_prob = gnb.predict_proba(val_X)
        val_pred_prob = val_pred_prob[:, 1]

        train_auc.append(roc_auc_score(train_y, train_pred_prob))
        val_auc.append(roc_auc_score(val_y, val_pred_prob))
        ##########################

    avg_train_acc = sum(train_acc) / len(train_acc)
    avg_val_acc = sum(val_acc) / len(val_acc)

    avg_train_auc = sum(train_auc) / len(train_auc)
    avg_val_auc = sum(val_auc) / len(val_auc)

    train_auc_std = np.std(train_auc)
    val_auc_std = np.std(val_auc)

    print("Training accuracy: ", avg_train_acc * 100, "%")
    print("Validation accuracy: ", avg_val_acc * 100, "%")
    print("Training AUC: ", avg_train_auc)
    print("Validation AUC: ", avg_val_auc)
    print("Training AUC std: ", train_auc_std)
    print("Validation AUC std: ", val_auc_std)

    return avg_train_acc, avg_val_acc, avg_train_auc, avg_val_auc, train_auc_std, val_auc_std
train_acc_NB, val_acc_NB, train_auc_NB, val_auc_NB, train_std_NB, val_std_NB, = NB(train_val_X, train_val_y, 0)

#Form a table
Table_Q2_NB = PrettyTable()
Table_Q2_NB_title = ['Data', 'Accuracy', 'AUC mean', 'AUC std']
Table_Q2_NB.field_names = Table_Q2_NB_title
Table_Q2_NB.add_row(['Training Data',train_acc_NB, train_auc_NB, train_std_NB])
Table_Q2_NB.add_row(['Validation Data', val_acc_NB, val_auc_NB, val_std_NB])
print(Table_Q2_NB)


# #SMOTE
# smo = SMOTE(random_state=0, sampling_strategy=0.2)
# train_val_X_smo, train_val_y_smo = smo.fit_sample(train_val_X,train_val_y)


#Q3 Random forest
train_acc_forest_smo, val_acc_forest_smo, train_auc_forest_smo, val_auc_forest_smo, train_std_forest_smo, val_std_forest_smo, num_trees = Random_forest(train_val_X, train_val_y, 1)

#Form a table
num_trees = num_trees.tolist()
Table_Q3_forest = PrettyTable()
Table_Q3_forest_title = num_trees.copy()
Table_Q3_forest_title.insert(0,'type/number of trees')
Table_Q3_forest.field_names = Table_Q3_forest_title
Table_Q3_train_acc_forest = train_acc_forest_smo.copy()
Table_Q3_train_acc_forest.insert(0,"train_acc_forest_smo")
Table_Q3_forest.add_row(Table_Q3_train_acc_forest)
Table_Q3_val_acc_forest = val_acc_forest_smo.copy()
Table_Q3_val_acc_forest.insert(0,"val_acc_forest_smo")
Table_Q3_forest.add_row(Table_Q3_val_acc_forest)
Table_Q3_train_auc_forest = train_auc_forest_smo.copy()
Table_Q3_train_auc_forest.insert(0,"train auc forest_smo")
Table_Q3_forest.add_row(Table_Q3_train_auc_forest)
Table_Q3_val_auc_forest = val_auc_forest_smo.copy()
Table_Q3_val_auc_forest.insert(0,"val auc forest_smo")
Table_Q3_forest.add_row(Table_Q3_val_auc_forest)
Table_Q3_train_std_forest = train_std_forest_smo.copy()
Table_Q3_train_std_forest.insert(0,"train std forest_smo")
Table_Q3_forest.add_row(Table_Q3_train_std_forest)
Table_Q3_val_std_forest = val_std_forest_smo.copy()
Table_Q3_val_std_forest.insert(0,"val std forest_smo")
Table_Q3_forest.add_row(Table_Q3_val_std_forest)
print(Table_Q3_forest)

#Q3 xgboost
train_acc_xgboost_smo, val_acc_xgboost_smo, train_auc_xgboost_smo, val_auc_xgboost_smo, train_std_xgboost_smo, val_std_xgboost_smo, etas = XGBOOST(train_val_X, train_val_y, 1)

#Form a table
etas = etas.tolist()
Table_Q3_xgboost = PrettyTable()
Table_Q3_xgboost_title = etas.copy()
Table_Q3_xgboost_title.insert(0,'type/etas')
Table_Q3_xgboost.field_names = Table_Q3_xgboost_title
Table_Q3_train_acc_xgboost = train_acc_xgboost_smo.copy()
Table_Q3_train_acc_xgboost.insert(0,"train_acc_xgboost_smo")
Table_Q3_xgboost.add_row(Table_Q3_train_acc_xgboost)
Table_Q3_val_acc_xgboost = val_acc_xgboost_smo.copy()
Table_Q3_val_acc_xgboost.insert(0,"val_acc_xgboost_smo")
Table_Q3_xgboost.add_row(Table_Q3_val_acc_xgboost)
Table_Q3_train_auc_xgboost = train_auc_xgboost_smo.copy()
Table_Q3_train_auc_xgboost.insert(0,"train auc xgboost_smo")
Table_Q3_xgboost.add_row(Table_Q3_train_auc_xgboost)
Table_Q3_val_auc_xgboost = val_auc_xgboost_smo.copy()
Table_Q3_val_auc_xgboost.insert(0,"val auc xgboost_smo")
Table_Q3_xgboost.add_row(Table_Q3_val_auc_xgboost)
Table_Q3_train_std_xgboost = train_std_xgboost_smo.copy()
Table_Q3_train_std_xgboost.insert(0,"train std xgboost_smo")
Table_Q3_xgboost.add_row(Table_Q3_train_std_xgboost)
Table_Q3_val_std_xgboost = val_std_xgboost_smo.copy()
Table_Q3_val_std_xgboost.insert(0,"val std xgboost_smo")
Table_Q3_xgboost.add_row(Table_Q3_val_std_xgboost)
print(Table_Q3_xgboost)

#Q3 SVM
train_acc_SVM_smo, val_acc_SVM_smo, train_auc_SVM_smo, val_auc_SVM_smo, train_std_SVM_smo, val_std_SVM_smo, C_svms = SVM(train_val_X, train_val_y, 1)

#Form a table
C_svms = C_svms.tolist()
Table_Q3_SVM = PrettyTable()
Table_Q3_SVM_title = C_svms.copy()
Table_Q3_SVM_title.insert(0,'type/C')
Table_Q3_SVM.field_names = Table_Q3_SVM_title
Table_Q3_train_acc_SVM = train_acc_SVM_smo.copy()
Table_Q3_train_acc_SVM.insert(0,"train_acc_SVM_smo")
Table_Q3_SVM.add_row(Table_Q3_train_acc_SVM)
Table_Q3_val_acc_SVM = val_acc_SVM_smo.copy()
Table_Q3_val_acc_SVM.insert(0,"val_acc_SVM_smo")
Table_Q3_SVM.add_row(Table_Q3_val_acc_SVM)
Table_Q3_train_auc_SVM = train_auc_SVM_smo.copy()
Table_Q3_train_auc_SVM.insert(0,"train auc SVM_smo")
Table_Q3_SVM.add_row(Table_Q3_train_auc_SVM)
Table_Q3_val_auc_SVM = val_auc_SVM_smo.copy()
Table_Q3_val_auc_SVM.insert(0,"val auc SVM_smo")
Table_Q3_SVM.add_row(Table_Q3_val_auc_SVM)
Table_Q3_train_std_SVM = train_std_SVM_smo.copy()
Table_Q3_train_std_SVM.insert(0,"train std SVM_smo")
Table_Q3_SVM.add_row(Table_Q3_train_std_SVM)
Table_Q3_val_std_SVM = val_std_SVM_smo.copy()
Table_Q3_val_std_SVM.insert(0,"val std SVM_smo")
Table_Q3_SVM.add_row(Table_Q3_val_std_SVM)
print(Table_Q3_SVM)

#Q3 KNN
train_acc_KNN_smo, val_acc_KNN_smo, train_auc_KNN_smo, val_auc_KNN_smo, train_std_KNN_smo, val_std_KNN_smo, num_neighbors = KNN(train_val_X, train_val_y, 1)

#Form a table
num_neighbors = num_neighbors.tolist()
Table_Q3_KNN = PrettyTable()
Table_Q3_KNN_title = num_neighbors.copy()
Table_Q3_KNN_title.insert(0,'type/number of neighbors')
Table_Q3_KNN.field_names = Table_Q3_KNN_title
Table_Q3_train_acc_KNN = train_acc_KNN_smo.copy()
Table_Q3_train_acc_KNN.insert(0,"train_acc_KNN_smo")
Table_Q3_KNN.add_row(Table_Q3_train_acc_KNN)
Table_Q3_val_acc_KNN = val_acc_KNN_smo.copy()
Table_Q3_val_acc_KNN.insert(0,"val_acc_KNN_smo")
Table_Q3_KNN.add_row(Table_Q3_val_acc_KNN)
Table_Q3_train_auc_KNN = train_auc_KNN_smo.copy()
Table_Q3_train_auc_KNN.insert(0,"train auc KNN_smo")
Table_Q3_KNN.add_row(Table_Q3_train_auc_KNN)
Table_Q3_val_auc_KNN = val_auc_KNN_smo.copy()
Table_Q3_val_auc_KNN.insert(0,"val auc KNN_smo")
Table_Q3_KNN.add_row(Table_Q3_val_auc_KNN)
Table_Q3_train_std_KNN = train_std_KNN_smo.copy()
Table_Q3_train_std_KNN.insert(0,"train std KNN_smo")
Table_Q3_KNN.add_row(Table_Q3_train_std_KNN)
Table_Q3_val_std_KNN = val_std_KNN_smo.copy()
Table_Q3_val_std_KNN.insert(0,"val std KNN_smo")
Table_Q3_KNN.add_row(Table_Q3_val_std_KNN)
print(Table_Q3_KNN)

#Q3 Naive Bayes
train_acc_NB_smo, val_acc_NB_smo, train_auc_NB_smo, val_auc_NB_smo, train_std_NB_smo, val_std_NB_smo, = NB(train_val_X, train_val_y, 1)

#Form a table
Table_Q3_NB = PrettyTable()
Table_Q3_NB_title = ['Data', 'Accuracy', 'AUC mean', 'AUC std']
Table_Q3_NB.field_names = Table_Q3_NB_title
Table_Q3_NB.add_row(['Training Data Smote',train_acc_NB_smo, train_auc_NB_smo, train_std_NB_smo])
Table_Q3_NB.add_row(['Validation Data Smote', val_acc_NB_smo, val_auc_NB_smo, val_std_NB_smo])
print(Table_Q3_NB)

#Q4
#Compare
best_number_trees = num_trees[np.argmax(val_auc_forest_smo)]
best_learning_rate = etas[np.argmax(val_auc_xgboost_smo)]
best_C = C_svms[np.argmax(val_auc_SVM_smo)]
best_k = num_neighbors[np.argmax(val_auc_KNN_smo)]

best_random_forest = np.max(val_auc_forest_smo)
best_xgboost = np.max(val_auc_xgboost_smo)
best_SVM = np.max(val_auc_SVM_smo)
best_KNN = np.max(val_auc_KNN_smo)
best_NB = val_auc_NB_smo
best_each = [best_random_forest, best_xgboost, best_SVM, best_KNN, best_NB]
best_model = np.argmax(best_each)

if best_model == 0:
    print("Best model is Random forest, number of trees = ", best_number_trees)
elif best_model == 1:
    print("Best model is xgboost, eta = ", best_learning_rate)
elif best_model == 2:
    print("Best model is SVM, number of C = ", best_C)
elif best_model == 3:
    print("Best model is KNN, number of neighbors = ", best_k)
else:
    print("Best model is Naive Bayes")



def Best_model (train_val_X, train_val_y, best_number_trees):
    train_acc = []
    val_acc = []
    train_auc = []
    val_auc = []
    train_auc_std = []
    val_auc_std = []
    fold = 1
    for train_index, val_index in kf.split(train_val_X):
        ##########################
        train_X = train_val_X[train_index, :]
        train_y = train_val_y[train_index]

        smo = SMOTE(random_state=0, sampling_strategy=0.2)
        train_X, train_y = smo.fit_sample(train_X, train_y)

        val_X = train_val_X[val_index, :]
        val_y = train_val_y[val_index]

        dtc = RandomForestClassifier(n_estimators=best_number_trees)
        dtc.fit(train_X, train_y)
        train_acc.append(dtc.score(train_X, train_y))
        val_acc.append(dtc.score(val_X, val_y))

        train_pred_prob = dtc.predict_proba(train_X)
        train_pred_prob = train_pred_prob[:, 1]
        val_pred_prob = dtc.predict_proba(val_X)
        val_pred_prob = val_pred_prob[:, 1]

        train_auc.append(roc_auc_score(train_y, train_pred_prob))
        val_auc.append(roc_auc_score(val_y, val_pred_prob))

        #Plot the figure
        fpr, tpr, _ = metrics.roc_curve(val_y, val_pred_prob)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score(val_y, val_pred_prob))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic %d fold' % fold)
        plt.legend(loc="lower right")
        plt.show()

        fold = fold + 1
        ##########################
    avg_train_acc = sum(train_acc) / len(train_acc)
    avg_val_acc = sum(val_acc) / len(val_acc)

    avg_train_auc = sum(train_auc) / len(train_auc)
    avg_val_auc = sum(val_auc) / len(val_auc)

    train_auc_std = np.std(train_auc)
    val_auc_std = np.std(val_auc)

    print("Best model in rondam forest")
    print("Number of trees", best_number_trees)
    print("Training accuracy: ", avg_train_acc * 100, "%")
    print("Validation accuracy: ", avg_val_acc * 100, "%")
    print("Training AUC: ", avg_train_auc)
    print("Validation AUC: ", avg_val_auc)
    print("Training AUC std: ", train_auc_std)
    print("Validation AUC std: ", val_auc_std)

    return avg_train_acc, avg_val_acc, avg_train_auc, avg_val_auc, train_auc_std, val_auc_std

train_acc_best, val_acc_best, train_auc_best, val_auc_best, train_std__best, val_std_best = Best_model(train_val_X, train_val_y, best_number_trees)


#Question 5
train_acc = []
train_auc = []

best_number_trees = 130
#SMOTE all training data
smo = SMOTE(random_state=0)
train_val_X_smo, train_val_y_smo = smo.fit_sample(train_val_X,train_val_y)

dtc = RandomForestClassifier(n_estimators=best_number_trees) #Best model
dtc.fit(train_val_X_smo, train_val_y_smo) #train using all data with SMOTE

train_acc = dtc.score(train_val_X_smo, train_val_y_smo)
train_pred_prob = dtc.predict_proba(train_val_X_smo)
train_pred_prob = train_pred_prob[:, 1]
train_auc = roc_auc_score(train_val_y_smo, train_pred_prob)

print("Question 5 Training")
print("Number of trees", best_number_trees)
print("Training accuracy: ", train_acc * 100, "%")
print("Training AUC: ", train_auc)

test_predict = dtc.predict(B_array) #predict test set

#Write my prediction into CSV file
with open("labels.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len_test):
        writer.writerow([test_predict[i]])

print()
print()
















