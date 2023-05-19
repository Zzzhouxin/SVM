import pandas as pd
import numpy as np
from pprint import pprint
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

'''
    load data
'''
df = pd.read_csv('__result.csv')
print(df.head())

'''
    text cleaning and preparation
'''
product_code = {
    "nginx": 1,
    "iis": 2,
    "lighttpd": 3,
    "boa": 4,
    "tomcat": 5,
    "rompager": 6,
    "http server": 7,
    "micro httpd": 8,
    "jetty": 9,
    "tengine": 0
}

df["product_code"] = df["production"]
df = df.replace({"product_code": product_code})
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df['feature'], df['product_code'], test_size=0.0526, random_state=8)

'''
    text representation and Parameter election
'''
ngram_range = (1, 1)
min_df = 0.04
max_df = 0.3
max_features = 210

'''
    TF_IDF
'''
tfidf = TfidfVectorizer(encoding='utf-8', ngram_range=ngram_range, stop_words=None, lowercase=False, max_df=max_df,
                        min_df=min_df, max_features=max_features, sublinear_tf=True)

tf_fit = tfidf.fit_transform(X_train.values.astype('U'))

features_train = tf_fit.toarray()
labels_train = y_train
# print(tfidf.get_feature_names())
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)

'''
    Cross-Validation for Hyperparameter tuning
'''
# C
C = [0.01, 0.1, 1, 10, 100]

# gamma
gamma = [0.001, 0.01, 0.1, 1, 10, 100]

# degree
degree = [1, 2, 3, 4, 5]

# kernel
kernel = ['linear', 'rbf', 'poly']

# probability
probability = [True]

# Create the random grid
random_grid = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree, 'probability': probability}
pprint(random_grid)

'''
# Randomized Search Cross Validation
# First create the base model to tune
'''
svc = svm.SVC(random_state=8)

# Definition of the random search
random_search = RandomizedSearchCV(estimator=svc, param_distributions=random_grid, n_iter=50, scoring='accuracy', cv=3,
                                   verbose=1, random_state=8)

# Fit the random search model
random_search.fit(features_train, labels_train)

'''
    Create the parameter grid based on the results of random search 
'''
# Create the parameter grid based on the results of random search
C = [0.1, 1, 10, 100]
gamma = [1, 10, 100]

param_grid = [
    {'C': C, 'kernel': ['rbf'], 'gamma': gamma}
]

# Create a base model
svc = svm.SVC(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=cv_sets, verbose=1)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)

print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)

'''
Fitting 3 folds for each of 12 candidates, totalling 36 fits
The best hyperparameters from Grid Search are:
{'C': 10, 'gamma': 1, 'kernel': 'rbf'}

The mean accuracy of a model with these hyperparameters is:
0.9941959103491383
'''

# 需要注释 重新运行
# best_svc = svm.SVC(C=10.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#                    decision_function_shape='ovr',
#                    degree=3, gamma=1, kernel='rbf', max_iter=-1, probability=False, random_state=8, shrinking=True,
#                    tol=0.001, verbose=False)
#
# best_svc.fit(features_train, labels_train)
#
# svc_pred = best_svc.predict(features_test)
#
# # # pprint(best_svc.get_params())
#
# # Training accuracy
# print("The training accuracy is: ")
# print(accuracy_score(labels_train, best_svc.predict(features_train)))
#
# # Test accuracy
# print("The test accuracy is: ")
# print(accuracy_score(labels_test, svc_pred))
#
# base_model = svm.SVC(random_state=8)
# base_model.fit(features_train, labels_train)
#
# print('base model test accuracy score is: ', accuracy_score(labels_test, base_model.predict(features_test)))
# print('base model train accuracy score is: ', accuracy_score(labels_train, base_model.predict(features_train)))
# best_svc.fit(features_train, labels_train)
# print('best svm model accuracy score is: ', accuracy_score(labels_test, best_svc.predict(features_test)))
