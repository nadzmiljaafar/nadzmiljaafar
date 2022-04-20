##==================================================================##
##  MODULE USED
##==================================================================##
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import joblib
import otomkdir

# machine learning
from sklearn.linear_model import LogisticRegression,Perceptron,SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.model_selection import KFold,RepeatedKFold,RepeatedStratifiedKFold,GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from scipy.stats import sem
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_recall_curve,plot_precision_recall_curve,plot_roc_curve


# create dynamic folder and data loading
mypath = otomkdir.otomkdir.auto_create_folder(folder_extend='Nadzmil')
df = pd.read_csv(str(mypath) + '\surveyA.csv')
print('hi')
##==================================================================##
##  DATA CLEANING
##==================================================================##

# income group segregation
income_group_a = ('1K to 2K','2K to 3K','3K to 4K','4K to 5K','Less than 1K')
income_group_b = ('5K to 6K','6K to 7K','7K to 8K','8K to 9K','9K to 10K','10K or more')
df['income_group'] = df['salary'].apply(lambda x: 'B' if x in income_group_a  else ('M' if x in income_group_b else None))

# observe dataset behaviour
print(df.describe())  
print(df.info())  
print("Null values:",df.isna().any(axis=1).sum())
print("Total rows:",df.shape[0])

# drop columns with a lot of nulls
cols_drop = ['house_value','age','person_living_in_house','transport_use','salary']
for cols in cols_drop:
    df.drop([cols],axis=1, inplace = True)

# drop rows with nulls, since less than 15% rows are affected
df.dropna(subset=['income_group'],inplace=True)
df.dropna(thresh=1,inplace=True)
df = df.dropna(how='any',axis=0)

# numeric columns cleaning
numeric_cols = ['house_rental_fee','house_loan_pmt','transport_spending','food_spending','kids_spending','personal_loan','education_loan','other_loan','public_transport_spending','house_utility','investment']
for cols in numeric_cols:
    df[cols] = pd.to_numeric(df[cols], errors='coerce')
df = df.dropna(how='any',axis=0)

# perform one-hot-encoding using pd.get_dummies, since datasets have multiple categorical features
cat_columns = ['race','gender','employment', 'education', 'married', 'house_type','vehicle']
df = pd.get_dummies(df, columns = cat_columns)

##==================================================================##
##  MODEL BUILDING - CLASSIFICATION COMPARISON 
##==================================================================##

X = df.drop("income_group",axis=1) 
y = df["income_group"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model testing
trees = 100
results = []
names_of_models = []

model_list = [('LR', LogisticRegression()),
             ('KNN', KNeighborsClassifier()),
             ('DTC', DecisionTreeClassifier()),
             ('RFC', RandomForestClassifier(n_estimators=trees))
             ]

for name, model in model_list:
    kfold = KFold(n_splits=20)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names_of_models.append(name)
    res = "{}: {} ({})".format(name, cv_results.mean(), cv_results.std())
    print(res)

##==================================================================##
##  HYPERTUNING - RANDOM FOREST CLASSIFIER (HIGHEST ACCURACY)
##==================================================================##

# parameters_for_testing = {
# "n_estimators"    : [50,100,150,200,250],
#  "max_features"   : [1,2,3,4,5],
# }

# model = RandomForestClassifier()

# kfold = KFold(n_splits=10, random_state=None)
# grid_cv = GridSearchCV(estimator=model, param_grid=parameters_for_testing, scoring='accuracy', cv=kfold)
# result = grid_cv.fit(X_train, y_train)

# print("Best: {} using {}".format(result.best_score_, result.best_params_))
# means = result.cv_results_['mean_test_score']
# stds = result.cv_results_['std_test_score']
# params = result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("{}  {} with: {}" .format(mean, stdev, param))

##==================================================================##
##  MODEL DEFINITION
#==================================================================##

# Best: 0.9463806970509383 using {'max_features': 1, 'n_estimators': 100}
model=RandomForestClassifier(n_estimators=100,max_features=5)
model = model.fit(X_train,y_train)

# export model
filename = (str(mypath) + '\predictor_model.sav')
joblib.dump(model, filename)

# accuracy report
predictions = model.predict(X_test)
print(accuracy_score(y_test,predictions))
print(classification_report(y_test,predictions))