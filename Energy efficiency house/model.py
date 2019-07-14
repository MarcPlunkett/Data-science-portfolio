## Import libraries ##

# Linear algebra
import numpy as np 

# Data processing
import pandas as pd

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib import style

# Import data sets

data = pd.read_csv('/Users/marc/Data-science-portfolio/Energy efficiency house/Dataset/ENB2012_data.csv')

data.columns
data.dtypes
data.isnull().sum()
data = data.drop('Unnamed: 10', 1)
data = data.drop('Unnamed: 11', 1)
data.corr(method='pearson')

data.isnull().sum()

## Univariate analysis ##
# There are two target variables, heating load and cooling load #
# Get the count and proportion of the dependent variable and plot the values

heatmap = data.corr() 
f, ax= plt.subplots(figsize=(9, 6))
sns.heatmap(heatmap, cmap='BuPu', annot=True)

X = data.drop('Heating Load', 1)
X = X.drop('Cooling Load', 1)
X=X.values
Y1 = data['Heating Load']
Y1=Y1.values
Y2 = data['Cooling Load']
Y2 = Y2.values


from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor

ETR = ExtraTreesRegressor()
rfe = RFE(ETR, 3)
fit = rfe.fit(X, Y1)

fit.n_features_
fit.support_
fit.ranking_

fit = rfe.fit(X, Y2)
fit.n_features_
fit.support_
fit.ranking_

#Features selected for heating load model are wall area, roof area, and Overall height
#Features selected for cooling load model are relative compactness, wall area and overall height


## Import models to be tested
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import model_selection


num_instances = len(X)

models = []
models.append(('LiR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('ElasticNet', ElasticNet()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVM', SVR()))

## Model results and comparison for heating load

results = []
names = []
scoring = []

for name, model in models:
    model.fit(X, Y1)
    y_pred = model.predict(X)

    kfold =  model_selection.KFold(n_splits=num_instances)
    cv_results = model_selection.cross_val_score(model, X, Y1, cv=10)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f(%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure(figsize=(20,10))
ax= fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_facecolor('xkcd:salmon')
plt.show()

# Bag_Re is the best estimators for heating load


for name, model in models:
    

    kfold =  model_selection.KFold(n_splits=num_instances)
    cv_results = model_selection.cross_val_score(model, X, Y2, cv=10)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f(%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure(figsize=(20,10))
ax= fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_facecolor('xkcd:salmon')
plt.show()

model = SVR()
model.fit(X, Y2)
y_pred = model.predict(X)

plt.plot(y_pred)
plt.plot(Y1)
plt.show

#Bag_Re is also the best estimator for cooling load

# from sklearn.model_selection import GridSearchCV

# paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}

# grid_search=GridSearchCV(BaggingRegressor(random_state=1),paramgrid)

# estimator = BaggingRegressor()

# estimator.get_params().keys()

# grid_search.fit(X,Y1)

# GridSearchCV(cv=5, error_score='raise', estimator=BaggingRegressor(),       
# fit_params=None, iid=True, n_jobs=1,       
# param_grid={'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 161, 181]},       
# pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',       
# scoring=None, verbose=0)

# grid_search.best_estimator_

#Accuracy can be increased by tuning hyperparameters of the models

# from sklearn.model_selection import StratifiedKFold
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.utils import np_utils
# from sklearn.model_selection import StratifiedKFold
# from keras.constraints import max_norm
# from sklearn.metrics import mean_absolute_error

# kf = StratifiedKFold(n_splits=5, shuffle =True, random_state=1)
# cvscores=[]

# from sklearn.utils.multiclass import type_of_target
# type_of_target(Y1)

# from sklearn.preprocessing import LabelEncoder

# from sklearn.model_selection import train_test_split
# X_train, X_test, y1_train, y1_test = train_test_split(X, Y1, test_size = 0.2, random_state = 0)

# label_encoder = LabelEncoder()
# y1_train = label_encoder.fit_transform(y1_train)

# model = Sequential()
# model.add(Dense(15, input_dim=8, init='uniform', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(8, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(5, init='uniform', activation='relu'))
# model.add(Dense(1, init='uniform', activation='relu'))

# print("%s:%.2f%%" %("Score", 100-scores))

# cvscores.append(100-scores)
# print("%.2f%% (+/- %.2f%%)" %(np.mean(cvscores), np.std(cvscores)))



