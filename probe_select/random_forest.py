from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
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
df_train = pd.read_csv('../去重.csv')
train_set = pd.DataFrame(columns=['feature', 'production'])
train_set['feature'] = df_train['feature']
train_set['production'] = df_train['production']
print(train_set)

# split the data to train set and test set
X_train, X_test, y_train, y_test = train_test_split(train_set['feature'], train_set['production'], test_size=0.0526,
                                                    random_state=1)

# encoder the y
encoder = preprocessing.LabelEncoder()

x_train = encoder.fit_transform(X_train)
x_test = encoder.fit_transform(X_test)
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

# 创建随机森林分类器
rfc = RandomForestClassifier(n_estimators=100, random_state=0)

# 使用随机森林分类器拟合数据
rfc.fit(x_train, y_train)

# 创建特征选择器
selector = SelectFromModel(rfc, prefit=True, max_features=5)

# 使用特征选择器选择特征
X_train_new = selector.transform(X_train)