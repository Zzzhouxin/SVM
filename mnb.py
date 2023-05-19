from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn import decomposition, ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, \
    CountVectorizer  # nbc means naive bayes classifier
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

import pandas as pd

# extract data
df_train = pd.read_csv('去重.csv')
train_set = pd.DataFrame(columns=['feature', 'production'])
train_set['feature'] = df_train['feature']
train_set['production'] = df_train['production']
print(train_set)

# split the data to train set and test set
X_train, X_test, y_train, y_test = train_test_split(train_set['feature'], train_set['production'], test_size=0.0526,
                                                    random_state=1)

# encoder the y
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# tf-idf

# WordLevel tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train_set['feature'])
xtrain_tfidf = tfidf_vect.transform(X_train)
xtest_tfidf = tfidf_vect.transform(X_test)

# ngram tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
tfidf_vect_ngram.fit(train_set['feature'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(X_train)
xtest_tfidf_ngram = tfidf_vect_ngram.transform(X_test)

# charlevel tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
tfidf_vect_ngram_chars.fit(train_set['feature'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_train)
xtest_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_test)

cv = KFold(10, shuffle=True, random_state=0)
cv.get_n_splits(xtrain_tfidf)
# cv.get_n_splits(X_train)
scores = cross_val_score(MultinomialNB(), xtrain_tfidf, y_train, cv=cv)
print(scores)
parameters = {
    'alpha': (1, 0.1, 0.01, 0.001, 0.0001),
}
grid_search = GridSearchCV(MultinomialNB(), parameters, n_jobs=1, cv=cv)
grid_search.fit(xtrain_tfidf, y_train)
print(grid_search.best_params_)
print("Best score: %0.3f" % grid_search.best_score_)

m1 = MultinomialNB(alpha=0.01).fit(xtrain_tfidf, y_train)
predictions = m1.predict(xtest_tfidf)
print("MNB, WordLevel TF-IDF: ", metrics.accuracy_score(predictions, y_test))

cv = KFold(10, shuffle=True, random_state=0)
cv.get_n_splits(xtrain_tfidf_ngram)
# cv.get_n_splits(X_train)
scores = cross_val_score(MultinomialNB(), xtrain_tfidf_ngram, y_train, cv=cv)
print(scores)
parameters = {
    'alpha': (1, 0.1, 0.01, 0.001, 0.0001),
}
grid_search = GridSearchCV(MultinomialNB(), parameters, n_jobs=1, cv=cv)
grid_search.fit(xtrain_tfidf_ngram, y_train)
print(grid_search.best_params_)
print("Best score: %0.3f" % grid_search.best_score_)

m2 = MultinomialNB(alpha=0.01).fit(xtrain_tfidf_ngram, y_train)
predictions = m2.predict(xtest_tfidf_ngram)
print("MNB, N-Gram Vectors: ", metrics.accuracy_score(predictions, y_test))

cv = KFold(10, shuffle=True, random_state=0)
cv.get_n_splits(xtrain_tfidf_ngram_chars)
# cv.get_n_splits(X_train)
scores = cross_val_score(MultinomialNB(), xtrain_tfidf_ngram_chars, y_train, cv=cv)
print(scores)
parameters = {
    'alpha': (1, 0.1, 0.01, 0.001, 0.0001),
}
grid_search = GridSearchCV(MultinomialNB(), parameters, n_jobs=1, cv=cv)
grid_search.fit(xtrain_tfidf_ngram_chars, y_train)
print(grid_search.best_params_)
print("Best score: %0.3f" % grid_search.best_score_)

m3 = MultinomialNB(alpha=0.01).fit(xtrain_tfidf_ngram_chars, y_train)
predictions = m3.predict(xtest_tfidf_ngram_chars)
print("MNB, CharLevel Vectors: ", metrics.accuracy_score(predictions, y_test))

# Create a vector counter object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train_set['feature'])

# Use vector counter object to convert training set and validation set
xtrain_count = count_vect.transform(X_train)
xvalid_count = count_vect.transform(X_test)
print(len(count_vect.get_feature_names()))
print(len(count_vect.vocabulary_))

cv = KFold(10, shuffle=True, random_state=0)
cv.get_n_splits(xtrain_count)
scores = cross_val_score(MultinomialNB(), xtrain_count, y_train, cv=cv)
print(scores)
parameters = {
    'alpha': (1, 0.1, 0.01, 0.001, 0.0001),
}
grid_search = GridSearchCV(MultinomialNB(), parameters, n_jobs=1, cv=cv)
grid_search.fit(xtrain_count, y_train)
print(grid_search.best_params_)
print("Best score: %0.3f" % grid_search.best_score_)

m_count = MultinomialNB(alpha=1).fit(xtrain_count, y_train)
predictions = m_count.predict(xvalid_count)
print("NB, Count Vectors: ", metrics.accuracy_score(predictions, y_test))

predicted_y = m_count.predict(xvalid_count)
print(accuracy_score(y_test, predicted_y))
print(precision_score(y_test, predicted_y, average='micro'))
print(recall_score(y_test, predicted_y, average='micro'))
print(f1_score(y_test, predicted_y, average='micro'))
print(f1_score(y_test, predicted_y, average='macro'))
target = ['nginx',
          'iis',
          'apache',
          ]

print(classification_report(y_test, predicted_y, target_names=target))
