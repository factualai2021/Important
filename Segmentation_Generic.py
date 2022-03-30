import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import itertools
import json
import pandas as pd
import time
import datetime
import re
import os
import sys
import io

from sklearn import cluster, tree, decomposition
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import sklearn
#import Decision_Tree_Generic

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from feature_engine.selection import (
RecursiveFeatureElimination,
DropConstantFeatures,
DropDuplicateFeatures,
)

import s3_connection_Updated as s3
from flask import Flask
from flask_socketio import SocketIO, join_room
from flask import request, jsonify


########################## Completed FLASH related initial setup

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

########################################################################


def get_data(path1):
    return pd.read_csv(path1)

def get_features(df):
    return df.columns

def check_objects(df, object_df, int_df):
    non_imp = []
    imp_cat = []
    imp_int = []
    for item in object_df:
        st = len(set(df[item].unique()))
        if st < df.shape[0]*.1:
            imp_cat.append(item)
        else:
            non_imp.append(item)
    for item in int_df:
        st = len(set(df[item].unique()))
        if st < df.shape[0]*.1:
            imp_cat.append(item)
        else:
            imp_int.append(item)
    return non_imp, imp_cat, imp_int

def check_multicollinearity(df, know_Your_Data):
    X = df[list(df.columns)].assign(const=1)
    vif_info = pd.DataFrame(columns=['Column', 'VIF'])
    vif_info['Column'] = X.columns
    vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_info = vif_info.sort_values('VIF', ascending=False)
    print(vif_info)
    print(vif_info.shape)
    print(type(vif_info))

    know_Your_Data['multicolinearity'] = [{'data':vif_info.to_dict(orient="records")}]
    #                                        {'graph':[{'chart':'bar'}, {'x-axis': 'Variable'}, {'y-axis': 'VIF Score'}]}}
    return know_Your_Data


def write_features(int_df, object_df, non_important_features, missing_columns, know_Your_Data):
    temp = []
    result = {}
    for element in int_df:
        temp.append(element)
    result['Numerical_Features'] = temp
    temp = []
    for element in object_df:
        temp.append(element)
    result['Categorical_Features'] = temp
    temp = []
    for element in non_important_features:
        temp.append(element)
    result['Non_Important_Features'] = temp
    temp = []
    for element in missing_columns:
        if element != '':
            temp.append(element)
    result['Missing_Value_Features'] = temp

    know_Your_Data['Features_Summary'] = [result]
    return know_Your_Data


def check_pca(df, features):
    # Standardize data to 0 mean and 1 variance
    PC_columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15',
                 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23', 'PC24', 'PC25', 'PC26', 'PC27', 'PC28', 'PC29',
                  'PC30', 'PC31', 'PC32', 'PC33', 'PC34', 'PC35', 'PC36', 'PC37', 'PC38', 'PC39', 'PC40', 'PC41', 'PC42', 'PC43',
                  'PC44', 'PC45', 'PC46', 'PC47', 'PC48', 'PC49', 'PC50', 'PC51', 'PC52', 'PC53', 'PC54', 'PC55', 'PC56', 'PC57',
                  'PC58', 'PC59', 'PC60', 'PC61', 'PC62', 'PC63', 'PC64', 'PC65', 'PC66', 'PC67', 'PC68', 'PC69', 'PC70']
    #PC_no = round(df2.shape[1]/2)
    #PC_no = df.shape[1]
    #x = df.loc[:, features].values
    x = df[features]
    PC_no = x.shape[1]
    x = StandardScaler().fit_transform(x)

    # Perform PCA using desired componenents (k=PC_no)
    pca = PCA(n_components=PC_no)
    principalComponents = pca.fit_transform(x)

    # Return the explained variance by each componenent using np.around to round to two decimal places
    variation = pd.DataFrame(np.around(pca.explained_variance_ratio_*100, 2), columns = ["Variation_Explained"])
    variation['cum_sum'] = variation['Variation_Explained'].cumsum()
    return(variation)

def do_pca(df, features, PC_no):
    # Standardize data to 0 mean and 1 variance
    PC_columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15',
                 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22', 'PC23', 'PC24', 'PC25', 'PC26', 'PC27', 'PC28', 'PC29',
                  'PC30', 'PC31', 'PC32', 'PC33', 'PC34', 'PC35', 'PC36', 'PC37', 'PC38', 'PC39', 'PC40', 'PC41', 'PC42', 'PC43',
                  'PC44', 'PC45', 'PC46', 'PC47', 'PC48', 'PC49', 'PC50', 'PC51', 'PC52', 'PC53', 'PC54', 'PC55', 'PC56', 'PC57',
                  'PC58', 'PC59', 'PC60', 'PC61', 'PC62', 'PC63', 'PC64', 'PC65', 'PC66', 'PC67', 'PC68', 'PC69', 'PC70']
    #x = df.loc[:, features].values
    x = df[features]
    x = StandardScaler().fit_transform(x)

    # Perform PCA using desired componenents (k=PC_no)
    pca = PCA(n_components=PC_no)
    principalComponents = pca.fit_transform(x)

    #principalDf = pd.DataFrame(data = principalComponents, columns = PC_columns[0:PC_no])
    principalDf = pd.DataFrame(data = principalComponents)
    return principalDf

def pca_process(df2, n_features):
    mat = check_pca(df2, n_features)
    n_90percent = mat[mat['cum_sum'] > 90].iloc[1,].name
    n_95percent = mat[mat['cum_sum'] > 95].iloc[1,].name

    # If number of PCA components needed to reach 90% of variation_explained is leass than half of original components than considering PCA
    if n_90percent < df2.shape[1]/2:
        consider_pca = True
    else:
        consider_pca = False

    if consider_pca == True:
        df_f = do_pca(df2, n_features, n_95percent)
    else:
        df_f = df2[n_features]
        df_f[n_features] = preprocessing.scale(df_f[n_features])

    return df_f

def choose_k(df_f):
    cluster_group = []
    sil_score = []
    for n_cluster in range(2, 11):
        kmeans = KMeans(n_clusters=n_cluster).fit(df_f)
        label = kmeans.labels_
        sil_coeff = silhouette_score(df_f, label, metric='euclidean')
        cluster_group.append(n_cluster)
        sil_score.append(sil_coeff)
    max_value = max(sil_score)
    max_sil_score_index = sil_score.index(max_value)
    req_cluster_group = cluster_group[max_sil_score_index]
    print("1. req_cluster_group ........", req_cluster_group)
    return req_cluster_group

def process_k_means(comp_k_mean, req_cluster_group, n_features, group_explainability, df2, df_f,
                    s3_object, bucket_name, upload_output_path):
    # K-Means with required number of groups

    km = cluster.KMeans(n_clusters=req_cluster_group, max_iter=300, random_state=None)
    df_f[comp_k_mean] = km.fit_predict(df_f)

    if comp_k_mean != 'Final_K_mean':
        df2 = pd.concat([df2, df_f[comp_k_mean]],axis=1)

    cluster_mean = df2.groupby(comp_k_mean)[n_features].mean()
    pop_mean = pd.DataFrame(df2[n_features].mean()).T
    pop_mean[comp_k_mean]="pop_mean"
    pop_mean.set_index(comp_k_mean, inplace=True)
    summary = pd.concat([cluster_mean, pop_mean],axis=0)

    pop_std = pd.DataFrame(df2[n_features].std()).T
    pop_std[comp_k_mean]="pop_std"
    pop_std.set_index(comp_k_mean, inplace=True)
    summary = pd.concat([summary, pop_std],axis=0)

    cluster_zscore = pd.DataFrame(columns=cluster_mean.columns)

    for row in range(cluster_mean.shape[0]):
        for col in cluster_mean.columns:
            cluster_zscore.loc[row,col] = (cluster_mean.loc[row,col] - pop_mean.loc['pop_mean',col])/pop_std.loc['pop_std',col]

    print("2. df2 ........", df2.shape, df2.columns)
    l1 = list(df2.groupby(comp_k_mean)[summary.columns[1]].agg('count').values)
    l1.append(df2.shape[0])
    l1 = pd.DataFrame(l1, columns = ['count'])
    k = []
    for i in l1.index:
        if i < l1.shape[0]-1:
            k.append(i)
        else:
            k.append('pop_mean')
    k = pd.DataFrame(k, columns=[comp_k_mean])
    l1 = pd.concat([l1, k],axis=1)
    l1.set_index(comp_k_mean, inplace=True)

    summary = pd.merge(l1, summary, left_index=True, right_index=True, how='inner')
    if group_explainability:
        group_statement = pd.DataFrame(columns=['Group', 'Details'])
        k = 0
        for i in range(cluster_zscore.shape[0]):
            for j in cluster_zscore.columns:
                if cluster_zscore.loc[i,j] > 1 or cluster_zscore.loc[i,j] < -1:
                    value = round(summary.loc[i,j]-summary.loc['pop_mean',j]/summary.loc['pop_mean',j]*100, 2)
                    statement = " "
                    if value > 1:
                        trend = "% higher than the mean value"
                    else:
                        trend = "% lower than the mean value"
                    statement = str(j) + " is " + str(abs(value)) + trend
                    group_statement.loc[k,'Group'] = "Group " + str(i)
                    group_statement.loc[k,'Details'] = statement
                    k = k+1

        group_statement_json = {}
        group_statement_json['group_statement'] = {'data':group_statement.to_dict(orient="records")}
        summary_str = comp_k_mean + '_GroupStatement.json'
        with open(summary_str, 'w') as f:
            json.dump(group_statement_json, f)
        upload_prediction_object = upload_output_path + summary_str
        s3_object.upload_to_bucket(bucket_name, upload_prediction_object, summary_str)
    return df2, df_f

def rfe_select(dfn):
    # Seperate train and test set
    X_train, X_test, y_train, y_test = train_test_split(
    dfn.iloc[:,:dfn.shape[1]-1],
    dfn.iloc[:,dfn.shape[1]-1],
    test_size=0.3,
    random_state=9)

    # Remove Constant, Quasi-constant, duplicated features
    pipe = Pipeline([
        ('constant', DropConstantFeatures(tol=0.998)),
        ('duplicated', DropDuplicateFeatures()),
    ])
    pipe.fit(X_train)

    # Remove Features
    X_train = pipe.transform(X_train)
    X_test = pipe.transform(X_test)
    X_train.shape, X_test.shape

    model = GradientBoostingClassifier(
    n_estimators = X_train.shape[1],
    max_depth=2,
    random_state=10,)

    sel = RecursiveFeatureElimination(
    variables = None,
    estimator = model,
    scoring = 'roc_auc',
    threshold = 0.0005,
    cv = 2,)

    print(X_train.head(), X_train.shape, type(X_train))
    print(y_train.head(), y_train.shape, type(y_train))
    sel.fit(X_train, y_train)
    X_train = sel.transform(X_train)
    print("Checkpoint 1....")
    return(X_train.columns)

def process_features(comp_k_mean, df2, df_f):

    # Rename few columns ending with '.0' in df2
    new_col = {}
    for p in df2.columns:
        #new_col.append(p.replace('.0', ''))
        new_col[p] = str(p).replace('.0', '')
    new_col
    df2 = df2.rename(columns = new_col)

    # Rename few columns ending with '.0' in df_f
    new_col = {}
    for p in df_f.columns:
        new_col[p] = str(p).replace('.0', '')
    new_col
    df_f = df_f.rename(columns = new_col)

    # Form all combinations of 2 classes at a time
    s1 = set(df2[comp_k_mean])
    s1 = list(s1)
    combinations_object = itertools.combinations(s1, 2)
    combinations_list = list(combinations_object)
    combinations_list

    s2 = set()
    for k in combinations_list:
        df3 = df_f[df2[comp_k_mean].isin(k)]
        l1 = rfe_select(df3)
        for l in l1:
            s2.add(l)
    s2 = list(s2)

    return s2, df2, df_f

def rfe_dataset(s2, comp_k_mean, df2):
    # Make formula for glm
    col = s2
    df_str = []
    col_str = comp_k_mean + ' ~ '
    for c in col:
        col_str = col_str + ' + ' + str(c)
        df_str.append(str(c))
    col_str = col_str.replace('+', '', 1)

    df3 = pd.DataFrame(preprocessing.scale(df2[df_str]))
    df3.columns = df_str

    df3 = pd.concat([df3, df2.iloc[:,df2.shape[1]-1:]],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
    df3.iloc[:,:df3.shape[1]-1],
    df3.iloc[:,df3.shape[1]-1],
    test_size=0.3,
    random_state=9)

    return df_str, X_train, X_test, y_train, y_test


def rf_process(df_str, req_cluster_group, X_train, X_test, y_train, y_test):

    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators = len(df_str), criterion = 'entropy', random_state = 42)
    classifier.fit(X_train[df_str], y_train)

    arr = y_test.unique()
    print(arr)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test[df_str])
    if req_cluster_group == 2:
        roc_final = roc_auc_score(y_test, y_pred)
    else:
        roc_final = roc_auc_score(y_test, classifier.predict_proba(X_test[df_str]), multi_class='ovr')

    return roc_final, classifier

def lr_process(df_str, req_cluster_group, X_train, X_test, y_train, y_test):

    model = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', max_iter=500)
    fitted_model = model.fit(X_train[df_str], y_train)

    arr = y_test.unique()

    # Predicting the Test set results
    y_pred = fitted_model.predict(X_test[df_str])
    if req_cluster_group == 2:
        roc_final = roc_auc_score(y_test, y_pred)
    else:
        roc_final = roc_auc_score(y_test, fitted_model.predict_proba(X_test[df_str]), multi_class='ovr')

    return roc_final, fitted_model



def feature_sensitivity(fitted_model, X_test, df_str):
    y = np.exp(fitted_model.intercept_)
    proba_inter = pd.DataFrame(np.round(1/((1/y)+1) * 100, 2))
    proba_inter.rename(columns = {0:'Intercept'}, inplace=True)
    proba_inter

    x = np.exp(fitted_model.coef_)
    proba = pd.DataFrame(x-1)
    #proba = pd.DataFrame(np.round(1/((1/x)+1) * 100, 2))
    #odds_ratio = pd.DataFrame(x-1)
    #proba = np.round(odds_ratio/(1+odds_ratio)*100, 2)
    proba.columns = X_test[df_str].columns

    proba = pd.concat([proba_inter, proba],axis=1)
    print(proba)
    return proba



def tree_process(df2, df_str):
    df4 = df2[df_str]
    df4 = pd.concat([df4, df2.iloc[:,df2.shape[1]-1]],axis=1)

    data4_X = df4.iloc[:, 0:(df4.shape[1]-1)]
    data4_Y = df4.iloc[:, df4.shape[1]-1]

    feature_cols = list(data4_X.columns)

    decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=3)
    decision_tree = decision_tree.fit(data4_X, data4_Y)

    y_pred = decision_tree.predict(data4_X)

    #r = export_text(decision_tree, feature_names=feature_cols)
    #list_r = [x for x in r.split('\n')]

    class_names = list(data4_Y.unique())
    list_r = get_rules(decision_tree, feature_cols, class_names)

    return list_r, metrics.accuracy_score(data4_Y, y_pred)
    #confusion_matrix(data4_Y, y_pred)


def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


def process_final(df_z, s3_object, bucket_name, upload_output_path):
    final_df = df_z.loc[:,df_z.columns.str.contains("K_mean")]
    num_features = []
    cat_features = final_df.columns
    group_features = []

    # Create dummy variables from categorical features
    final_df = pd.get_dummies(final_df, columns=cat_features, drop_first=True)
    n_features = final_df.columns

    df_f = final_df

    # Choose appropriate K
    req_cluster_group = choose_k(df_f)

    # K-Means Clustering
    comp_k_mean = 'Final_K_mean'
    group_explainability = True
    final_df, df_f = process_k_means(comp_k_mean, req_cluster_group, n_features, group_explainability, final_df, df_f,
                                    s3_object, bucket_name, upload_output_path)

    return df_f[comp_k_mean]



def process_data(s3_object, bucket_name, upload_output_path, df1, comp, num_features, cat_features, group_features,
                 group_explainability, feature_importance, tree_explainability, sensitivity_analysis):

    # Combine categorical and numerical features
    df2 = df1[num_features+cat_features]

    num_count = len(num_features)

    # Create dummy variables from categorical features
    df2 = pd.get_dummies(df2, columns=cat_features, drop_first=True)
    n_features = df2.columns

    df2 = df2.astype("float")

    df2 = pd.concat([df2, df1[group_features]],axis=1)

    # If group_features is not null, include group features and group by these features
    if len(group_features) == 0:
        pass
    else:
        df2 = df2.groupby(group_features).mean().reset_index().dropna(axis=0)

    # Do PCA if required
    df_f = pca_process(df2, n_features)

    # Choose appropriate K
    req_cluster_group = choose_k(df_f)

    # K-Means Clustering
    comp_k_mean = comp + '_K_mean'
    df2, df_f = process_k_means(comp_k_mean, req_cluster_group, n_features, group_explainability, df2, df_f,
                                s3_object, bucket_name, upload_output_path)

    # Feature Selection
    s2, df2, df_f = process_features(comp_k_mean, df2, df_f)

    # Train Test split
    df_str, X_train, X_test, y_train, y_test = rfe_dataset(s2, comp_k_mean, df2)

    # Random Forest
    rf_roc_final, rf_classifier = rf_process(df_str, req_cluster_group, X_train, X_test, y_train, y_test)

    # Logistic Regression
    lr_roc_final, lr__classifier = lr_process(df_str, req_cluster_group, X_train, X_test, y_train, y_test)

    # Feature Importance from Random Forest
    if feature_importance:
        importance = rf_classifier.feature_importances_
        list_features = []
        for i,j in zip(df_str,importance):
            list_features.append({i:j})
        list_features.append({'graph':{'chart':'bar', 'x-axis': 'Variable', 'y-axis': 'Score'}})
        summary_str = comp + '_feature_importance.json'
        with open(summary_str, 'w') as f:
            json.dump(list_features, f)
        upload_prediction_object = upload_output_path + summary_str
        s3_object.upload_to_bucket(bucket_name, upload_prediction_object, summary_str)

    # Feature Sensitivity
    proba = feature_sensitivity(lr__classifier, X_test, df_str)
    if sensitivity_analysis:
        summary_str = comp + '_sensitivity.json'
        data = proba.to_json(summary_str, orient='index')
        upload_prediction_object = upload_output_path + summary_str
        s3_object.upload_to_bucket(bucket_name, upload_prediction_object, summary_str)

    # Tree Explainability
    list_r, dt_auc = tree_process(df2, df_str)
    if tree_explainability:
        # summary_str = comp + '_tree_explainability.txt'
        # textfile = open(summary_str, "w")
        # for element in list_r:
        #     textfile.write(element + "\n")
        # textfile.close()
        # upload_prediction_object = upload_output_path + summary_str
        # s3_object.upload_to_bucket(bucket_name, upload_prediction_object, summary_str)
        summary_str = comp + '_tree_explainability.json'
        tree_features = []
        tree_features.append({'rule':list_r})
        with open(summary_str, 'w') as f:
            json.dump(tree_features, f)
        upload_prediction_object = upload_output_path + summary_str
        s3_object.upload_to_bucket(bucket_name, upload_prediction_object, summary_str)

    return df2[comp_k_mean]


#def upload_routine()


def process_s3(process, bucket_name, download_input_path, file_name, upload_output_path, s3_object):

    # Downloading and load train and test csv file objects from s3 storage
    remote_object = download_input_path + file_name
    df = s3_object.get_csv(bucket_name, remote_object)
    #df = pd.read_csv("/Users/mohammadjawedasad/Desktop/Projects/IOT/Enterprise-AI/Unsupervised/brazilian-ecommerce/Combined_brazilian_commerce.csv")
    features = get_features(df)

    # Identify NON IMP Features, Numerical and Categorical Features
    object_df = features[df.dtypes=='object']
    int_df = features[df.dtypes=='int64']
    non_important_features, categorical_features, int_features = check_objects(df, object_df, int_df)

    # Drop non important features
    df1 = df.drop(non_important_features, axis=1)
    features = get_features(df1)
    float_df = features[df1.dtypes=='float64']
    int_df = int_features
    int_df.extend(float_df)
    object_df = categorical_features

    features = int_df + object_df

    df1 = df1[features]

    # Drop columns with high NULL values (>10%)
    non_missing_columns = df1.isnull().sum() < df.shape[0]*0.1
    missing_columns1 = df1.isnull().sum() > df.shape[0]*0.1
    missing_columns = get_features(df1.loc[:, missing_columns1])
    df1 = df1.loc[:,non_missing_columns]

    # Drop rows with any missing values
    df1 = df1.dropna()

    # Remove Constant, Quasi-constant, duplicated features
    pipe = Pipeline([
            ('constant', DropConstantFeatures(tol=0.998)),
            ('duplicated', DropDuplicateFeatures()),
           ])
    pipe.fit(df1)

    ####  Preprocessing ends

    if process == 'knowYourData':

        know_Your_Data ={}

        # 1. Check multicollinearity
        know_Your_Data = check_multicollinearity(df1[int_df], know_Your_Data)
        print("jawed0")
        # 2. Write different features details
        know_Your_Data = write_features(int_df, object_df, non_important_features, missing_columns, know_Your_Data)
        print("Jawed1")
        # 3. Correlation Heat Map
        know_Your_Data['Corr_HeatMap'] = [{'data':df1.corr().to_json(orient='records')}]
        print("jawed2")
        # 4. Describe Your Data
        know_Your_Data['Describe_Data'] = [{'data':df1.describe(include='all').T.to_json(orient='columns')}]
        print("jawed3")
        #json_result = json.dumps(know_Your_Data, indent=4)

        know_Data = [{'know_Your_Data':know_Your_Data}]

        summary_str = 'KnowYourData.json'
        with open(summary_str, 'w') as f:
            json.dump(know_Data, f)
        upload_prediction_object = upload_output_path + summary_str
        s3_object.upload_to_bucket(bucket_name, upload_prediction_object, summary_str)

    return df1


@app.route('/hello')
def hello():
    return 'Hello, World'



@app.route('/buildSegmentation', methods=['GET', 'POST'])
def buildSegmentation():
    #print(request, file=sys.stdout)
    print("Jawed")
    print(request)
    json_data = request.json
    conf = json_data
    #print(json_data, file=sys.stdout)
    print(json_data)
    counter = 0
    bucket_name = conf['bucket_name']
    download_input_path = conf['download_input_path']
    upload_output_path = conf['upload_output_path']
    input_file_name = conf['input_file_name']
    #model_name = conf['model_name']
    process = 'buildSegmentation'
    s3_object = s3.Connection()
    df1 = process_s3(process, bucket_name, download_input_path, input_file_name, upload_output_path, s3_object)

    for i1 in conf.keys():
        if i1 in ['bucket_name', 'download_input_path', 'upload_output_path', 'input_file_name', 'model_name']:
            continue
        counter = counter + 1
        comp=i1
        num_features = conf[i1]['num_features']
        cat_features = conf[i1]['cat_features']
        group_features = conf[i1]['group_features']
        group_explainability = conf[i1]['group_explainability']
        feature_importance = conf[i1]['feature_importance']
        tree_explainability = conf[i1]['tree_explainability']
        sensitivity_analysis = conf[i1]['sensitivity_analysis']
        if comp == 'component1':
            if len(group_features) == 0:
                df_z = df1
            else:
                df_z = df1.groupby(group_features).mean().reset_index().dropna(axis=0)

        col1 = process_data(s3_object, bucket_name, upload_output_path, df1, comp, num_features, cat_features, group_features,
                            group_explainability, feature_importance, tree_explainability, sensitivity_analysis)

        df_z = pd.concat([df_z, col1],axis=1)

    if counter > 1:
        col1 = process_final(df_z, s3_object, bucket_name, upload_output_path)
        df_z = pd.concat([df_z, col1],axis=1)
    return "OK"


@app.route('/knowYourData', methods=['GET', 'POST'])
def knowYourData():
    #print(request, file=sys.stdout)
    print(request)
    json_data = request.json
    conf = json_data
    #print(json_data, file=sys.stdout)
    print(json_data)

    bucket_name = conf['bucket_name']
    download_input_path = conf['download_input_path']
    upload_output_path = conf['upload_output_path']
    input_file_name = conf['input_file_name']
    #model_name = conf['model_name']
    process = 'knowYourData'
    s3_object = s3.Connection()
    df1 = process_s3(process, bucket_name, download_input_path, input_file_name, upload_output_path, s3_object)
    return "OK"


if __name__ == "__main__":
    #context = ('Certificates.pem', 'Certificates.pem')
    #app.run(host='localhost', port=8000, ssl_context=context, threaded=True, debug=True)
    app.run(host='localhost', port=8001)
