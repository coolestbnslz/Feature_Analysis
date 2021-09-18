import pandas as pd
from main import *
train = pd.read_csv('credit_example.csv')
train_labels = train['TARGET']
train = train.drop(columns = ['TARGET'])
fs = FeatureAnalysis(data = train, labels = train_labels)
fs.find_missing(missing_threshold=0.6)
missing_features = fs.ops['missing']
print(missing_features)

fs.find_unique()
single_unique = fs.ops['single_unique']
print(single_unique)

fs.find_collinear(correlation_threshold=0.975)
correlated_features = fs.ops['collinear']
print(correlated_features)

fs.find_low_impt(cumulative_impt = 0.99,label='classification')
low_importance_features = fs.ops['low_importance']
print(low_importance_features)

fs.find_all( 0.6,0.98,0.99,'classification')


removed = fs.remove()
print(removed)
