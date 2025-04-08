import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class PrepareData:
    def __init__(self, data_dir="", project_name=None, data_file=None, test_size=0.2):
        self.pipe_line = None
        self.project_name = project_name
        self.data_save_dir = data_dir
        self.data_df = self.__read_data_to_df(data_file=data_file)
        self.train_set, self.test_set = self.split_data_into_train_and_test(test_size=test_size)
        self.save_train_test()


    @property
    def df_columns(self):
        return self.data_df.columns.tolist()

    def __read_data_to_df(self, data_file=''):
        try:
            return pd.read_pickle(os.path.join(self.data_save_dir, data_file))
        except FileNotFoundError:
            raise FileNotFoundError(f"File {data_file} not found in the directory {self.data_save_dir}")

    def split_data_into_train_and_test(self, test_size=None, stratified=None, strat_column=None):
        # Split the data into train and test sets
        # Stratifizieren?
        if stratified:
            splitter = StratifiedShuffleSplit(
                n_splits=10,
                test_size=test_size
            )
            start_train_index, strat_test_index = splitter.split(self.data_df, self.data_df[strat_column])
            train_set = self.data_df.loc[start_train_index]
            test_set = self.data_df.loc[strat_test_index]
        else:
            train_set, test_set = train_test_split(self.data_df, test_size=test_size)

        return train_set, test_set

    def split_features_and_labels(self, label=None, data=None):
        # Split the features and labels
        features = data.drop(label, axis=1)
        labels = data[label].copy()
        return features, labels


    def save_data(self, data_name=None, data=None):
        with open(f'{self.project_name}_{data_name}.pkl', 'wb') as data_file:
            pickle.dump(data, data_file, pickle.HIGHEST_PROTOCOL)

    def save_train_test(self):
        self.save_data(data_name='train_set', data=self.train_set)
        self.save_data(data_name='test_set', data=self.test_set)

class MlPipeLine:
    
    def __init__(self, cv_folds=5, scoring_function=None):
        self.grid_search = None
        self.pipe_line = None
        self.cv_folds = cv_folds
        self.scoring_function = scoring_function
        
    @staticmethod
    def build_preprocessor(list_of_transformers=None):
        if list_of_transformers:
            return ColumnTransformer(
                transformers=list_of_transformers,
            )
        else:
            return None

    def fit_pipeline(self, features=None, labels=None):
        self.pipe_line.fit(features,labels)

    def predict_pipeline(self, features=None):
        return self.pipe_line.predict(features)

    def score_pipeline(self, features=None, labels=None):
        return self.pipe_line.score(features, labels)

    def get_pipeline(self):
        return self.pipe_line

    def get_pipeline_steps(self):
        return self.pipe_line.steps

    def get_pipeline_params(self):
        return self.pipe_line.get_params()

    def cross_val_score_pipeline(self, features=None, labels=None):
        return cross_val_score(
            self.pipe_line,
            features,
            labels,
            cv=self.cv_folds,
            scoring=self.scoring_function
        )

    def build_pipeline(self,**kwargs):
        # Define the pipeline
        list_of_steps = []
        for name, step in kwargs.items():
            if step:
                list_of_steps.append((str(name), step))
        try:
            self.pipe_line = Pipeline(list_of_steps)
        except ValueError:
            raise ValueError(f"Pipeline could not be built with the given steps: {list_of_steps}")

    def do_grid_search(self, features=None, labels=None, param_grid=None):
        grid_search = GridSearchCV(
            self.pipe_line,
            param_grid,
            cv=self.cv_folds,
            scoring=self.scoring_function,
            n_jobs=-1
        )

        self.grid_search = grid_search.fit(features, labels)
        return


class DoMl:
    def __init__(self):
        self.labels_test = None
        self.labels_train = None
        self.features_test = None
        self.features_train = None
        self.data = None
        self.ml = None

    def do_ml(self):
        # Load the data
        self.data_preparation_ml()
        # Train the model
        categorical_cols = self.features_train.select_dtypes(exclude=np.number).columns.tolist()
        numerical_cols = self.features_train.select_dtypes(include=np.number).columns.tolist()
        self.fit_ml(
            list_of_preprocessors=[
            ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('passthrough', 'passthrough', numerical_cols),
            # ('transformer', FunctionTransformer(func=np.log, inverse_func=np.exp), ['ABV', 'Body', 'Alcohol'])
            ],
            scaler=MinMaxScaler(),
            pca=PCA(),
            linreg=Ridge()
        )
        # Evaluate the model
        self.evaluate_ml()
        # Predict the model
        self.predict_ml(self.features_train)
        # Do GridSearch
        self.grid_search_ml(
            param_grid = {
                'linreg__alpha': np.arange(0.0, 5.0, 0.1),
                'linreg__solver': ["svd", "cholesky", "lsqr"]  # Adjust Ridge solver
                }
        )
        # Evaluate the model
        self.evaluate_gridsearch(full=True)

    def test_ml(self):
        self.ml.predict_pipeline(self.features_test)
        self.ml.score_pipeline(self.features_test, self.labels_test)

    def data_preparation_ml(self):
        self.data = PrepareData(data_file='Beer_truncated_data.pkl', data_dir='data')
        self.features_train, self.labels_train = self.data.split_features_and_labels(label='review_overall', data=self.data.train_set)
        self.features_test, self.labels_test = self.data.split_features_and_labels(label='review_overall', data=self.data.test_set)


    def fit_ml(self, list_of_preprocessors=None, **kwargs):
        self.ml = MlPipeLine(cv_folds=5, scoring_function='r2')
        # Define the preprocessor
        preprocessor = self.ml.build_preprocessor(list_of_preprocessors)
        self.ml.build_pipeline(
            preprocessor=preprocessor,
            **kwargs
        )
        self.ml.fit_pipeline(self.features_train, self.labels_train)

    def predict_ml(self, features=None):
        predictions = self.ml.predict_pipeline(features)
        print(predictions)
        return

    def evaluate_ml(self):
        score = self.ml.score_pipeline(self.features_train, self.labels_train)
        cross_val_score = self.ml.cross_val_score_pipeline(self.features_train, self.labels_train)
        print(f"Score: {score}")
        print(f"Cross val score: {cross_val_score}, Mean: {np.mean(cross_val_score)}, Std: {np.std(cross_val_score)}")
        return

    def grid_search_ml(self, param_grid=None):
        self.ml.do_grid_search(self.features_train, self.labels_train, param_grid)

    def evaluate_gridsearch(self, full=None):
        if self.ml.do_grid_search:
            print("Best Parameters:", self.ml.grid_search.best_params_)
            print("Best Score:", self.ml.grid_search.best_score_)
            if full:
                print("All Results:")
                results = pd.DataFrame(self.ml.grid_search.cv_results_)
                print(results)
        else:
            raise ValueError("Grid search has not been performed yet.")


if __name__ == '__main__':

    do_ml = DoMl()
    do_ml.do_ml()
    # do_ml.test_ml()