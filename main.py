import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import chain
from plotting import *

class FeatureAnalysis():
    """
    Class for performing feature selection for data preprocessing or usuage of data in Machine Learning.

    Implements five different methods to identify features for removal

        1. Find columns with a missing percentage greater than a specified threshold
        2. Find columns with a single unique value
        3. Find collinear variables with a correlation greater than a specified correlation coefficient
        4. Find low importance features that do not contribute to a specified cumulative feature importance from the gbm

    Parameters
    --------
        data : dataframe
            A dataset with observations in the rows and features in the columns
        labels : array or series, default = None
            Array of labels for training the machine learning model to find feature importances. These can be either binary labels
            (if task is 'classification') or continuous targets (if task is 'regression').
            If no labels are provided, then the feature importance based methods are not available.

    Attributes
    --------

    ops : dict
        Dictionary of operations run and features identified for removal

    missing_stats : dataframe
        The fraction of missing values for all features

    record_missing : dataframe
        The fraction of missing values for features with missing fraction above threshold

    unique_stats : dataframe
        Number of unique values for all features

    record_single_unique : dataframe
        Records the features that have a single unique value

    corr_matrix : dataframe
        All correlations between all features in the data

    record_collinear : dataframe
        Records the pairs of collinear variables with a correlation coefficient above the threshold

    feature_importances : dataframe
        All feature importances from the gradient boosting machine

    record_zero_importance : dataframe
        Records the zero importance features in the data according to the gbm

    record_low_importance : dataframe
        Records the lowest importance features not needed to reach the threshold of cumulative importance according to the gbm


    Notes
    --------

        - All 4 operations can be run with the `identify_all` method.
        - If using feature importance, one-hot encoding is used for categorical variables which creates new columns

    """

    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

        if labels is None:
            print('No labels provided.')

        self.base_features = list(data.columns)
        self.one_hot_features = None
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_low_importance = None
        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None
        self.ops = {}
        self.one_hot_correlated = False
        self.all_identified=None

    def find_missing(self, missing_threshold):
        """Find the features with a fraction of missing values above 'missing_threshold'"""
        self.threshold = missing_threshold

        # Calculate the fraction of missing in each column
        missing_series = self.data.isnull().sum() / self.data.shape[0]
        self.missing_stats = pd.DataFrame(missing_series).rename(columns={'index': 'feature', 0: 'fraction'})
        self.missing_stats = self.missing_stats.sort_values('fraction', ascending=False)

        # Find the columns with a missing percentage above the threshold
        record_missing = pd.DataFrame(missing_series[missing_series > missing_threshold]).reset_index().rename(columns={'index': 'feature',0: 'fraction'})
        drop = list(record_missing['feature'])
        self.record_missing = record_missing
        self.ops['missing'] = drop
        print('%d features with greater than %0.2f missing values.\n' % (len(self.ops['missing']), self.threshold))
        plot_missing(self.missing_stats['fraction'])
    def find_unique(self):
        """Find the features having a single unique value. NaNs do not count as a unique value. """
        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'unique'})
        self.unique_stats = self.unique_stats.sort_values('unique', ascending=True)

        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns={'index': 'feature', 0: 'unique'})
        to_drop = list(record_single_unique['feature'])
        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop

        print('%d features with a single unique value.\n' % len(self.ops['single_unique']))
        plot_unique(self.unique_stats)

    def find_collinear(self, correlation_threshold,one_hot=False):
        """
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greather than `correlation_threshold`
        """

        self.corr_threshold = correlation_threshold
        self.one_hot_correlated = one_hot

        # Calculate the correlations between every column
        if one_hot:
            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)
            corr_matrix = pd.get_dummies(features).corr()

        else:
            corr_matrix = self.data.corr()

        self.corr_matrix = corr_matrix
        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]
            temp = pd.DataFrame.from_dict({'drop_feature': drop_features,'corr_feature': corr_features,'corr_value': corr_values})
            record_collinear = record_collinear.append(temp, ignore_index=True)

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (len(self.ops['collinear']), self.corr_threshold))
        plot_collinear(self.corr_matrix,self.record_collinear)

    def find_low_impt(self, cumulative_impt,label,n_iterations=10):
        """
        Finds the lowest importance features not needed to account for `cumulative_importance` fraction
        of the total feature importance from the gradient boosting machine.
        label corresponding to regession or classification
        """
        if self.labels is None:
            raise ValueError("No training labels provided.")

        features = pd.get_dummies(self.data)
        self.one_hot_features = [column for column in features.columns if column not in self.base_features]
        self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)
        feature_names = list(features.columns)
        features = np.array(features)
        labels = np.array(self.labels).reshape((-1,))
        feature_importance_values = np.zeros(len(feature_names))
        for _ in range(n_iterations):
            if label == 'classification':
                model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)

            elif label == 'regression':
                model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=-1)

            else:
                raise ValueError('Task must be either "classification" or "regression"')

            model.fit(features, labels)
            feature_importance_values += model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
        self.cumulative = cumulative_impt
        self.feature_importances = feature_importances
        # Make sure most important features are on top
        self.feature_importances = self.feature_importances.sort_values('cumulative_importance')

        # Identify the features not needed to reach the cumulative_importance
        record_low_importance = self.feature_importances[self.feature_importances['cumulative_importance'] > cumulative_impt]

        to_drop = list(record_low_importance['feature'])

        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = to_drop

        print('%d features required for cumulative importance of %0.2f.' % (len(self.feature_importances) -len(self.record_low_importance), self.cumulative))
        plot_feature_importances(self.feature_importances)

    def find_all(self,missing_threshold,correlation_threshold,cumulative_impt,label):
        """
        Use all five of the methods to identify features to remove.
        """

        # Implement each of the five methods
        self.find_missing(missing_threshold)
        self.find_unique()
        self.find_collinear(correlation_threshold)
        self.find_low_impt(cumulative_impt,label)

        # Find the number of features identified to drop
        self.all_identified = set(list(chain(*list(self.ops.values()))))
        self.n_identified = len(self.all_identified)

        print('%d total features out of %d identified for removal after one-hot encoding.\n' % (self.n_identified,self.data_all.shape[1]))

    def check_removal(self, keep_one_hot=True):

        """Check the identified features before removal. Returns a list of the unique features identified."""

        self.all_identified = set(list(chain(*list(self.ops.values()))))
        print('Total of %d features identified for removal' % len(self.all_identified))

        if not keep_one_hot:
            if self.one_hot_features is None:
                print('Data has not been one-hot encoded')
            else:
                one_hot_to_remove = [x for x in self.one_hot_features if x not in self.all_identified]
                print('%d additional one-hot features can be removed' % len(one_hot_to_remove))

        return list(self.all_identified)

    def remove(self):
        """
        Remove the features from the data according to the specified methods.
        """
        data = self.data_all
        print('{} methods have been run\n'.format(list(self.ops.keys())))
        features_to_drop = set(list(chain(*list(self.ops.values()))))
        features_to_drop = list(features_to_drop)

        # Remove the features and return the data
        data = data.drop(columns=features_to_drop)
        self.removed_features = features_to_drop
        print('Removed %d features.' % len(features_to_drop))
        return data.columns
