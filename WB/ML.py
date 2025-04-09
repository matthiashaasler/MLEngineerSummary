import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline


class PrepareData:
    def __init__(self, data_dir="", project_name=None, data_file=None, test_size=0.2):
        self.pipe_line = None
        self.project_name = project_name
        self.data_save_dir = data_dir
        if data_file is None:
            self.data_df = None
        else:
            self.data_df = self.__read_data_to_df(data_file=data_file)


    @property
    def df_columns(self):
        return self.data_df.columns.tolist()

    def __read_data_to_df(self, data_file=''):
        try:
            return pd.read_pickle(os.path.join(self.data_save_dir, data_file))
        except FileNotFoundError:
            raise FileNotFoundError(f"File {data_file} not found in the directory {self.data_save_dir}")

    def split_data_into_train_and_test(self, test_size=None, stratified=None, strat_column=None, number_of_bins=5):
        # Split the data into train and test sets
        # Stratifizieren?
        if stratified:
            self.data_df[strat_column+'_discrete'] = pd.cut(self.data_df[strat_column], bins=number_of_bins, labels=False)
            splitter = StratifiedShuffleSplit(
                n_splits=10,
                test_size=test_size
            )
            for strat_train_index, strat_test_index in splitter.split(self.data_df, self.data_df[strat_column+'_discrete']):

                train_set = self.data_df.loc[strat_train_index]
                test_set = self.data_df.loc[strat_test_index]

            train_set.drop(columns=[strat_column+'_discrete'], inplace=True)
            test_set.drop(columns=[strat_column+'_discrete'], inplace=True)
        else:
            train_set, test_set = train_test_split(self.data_df, test_size=test_size)

        return train_set, test_set

    def split_data(self, test_size=None, stratified=None, strat_column=None, label=None, save=None):

        train_set, test_set = self.split_data_into_train_and_test(
            test_size=test_size,
            stratified=stratified,
            strat_column=strat_column
        )
        if save:
            self.save_data(data_name='train_set', data=train_set)
            self.save_data(data_name='test_set', data=test_set)

        x_train, y_train = self.split_features_and_labels(label=label, data=train_set)
        x_test, y_test = self.split_features_and_labels(label=label, data=test_set)
        return x_train, x_test, y_train, y_test

    def load_features_and_labels(self, label=None):
        # Load the data from the pickle files
        train_set = self.load_data('train_set')
        test_set = self.load_data('test_set')

        x_train, y_train = self.split_features_and_labels(label=label, data=train_set)
        x_test, y_test = self.split_features_and_labels(label=label, data=test_set)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def split_features_and_labels(label=None, data=None):
        # Split the features and labels
        features = data.drop(label, axis=1)
        labels = data[label].copy()
        return features, labels

    def save_data(self, data_name=None, data=None):
        with open(os.path.join(self.data_save_dir,f'{self.project_name}_{data_name}.pkl'), 'wb') as data_file:
            pickle.dump(data, data_file, pickle.HIGHEST_PROTOCOL)

    def load_data(self, data_name=None):
        with open(os.path.join(self.data_save_dir,f'{self.project_name}_{data_name}.pkl'), 'rb') as data_file:
            data = pickle.load(data_file)
        return data



class MlPipeLine:
    
    def __init__(self):
        self.pipe_line = None

    def build_sk_learn_pipeline(self, dict_of_steps=None):
        list_of_steps = []

        for name, step in dict_of_steps.items():
                list_of_steps.append((name, step))
        try:
            self.pipe_line = Pipeline(list_of_steps)
        except ValueError:
            raise ValueError(f"Pipeline could not be built with the given steps: {list_of_steps}")


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



class DoMl:
    def __init__(self, cv_folds=5, scoring_function=None,
                 gs_parameter=None):
        self.output = 'full'
        self.grid_search = None
        self.scoring_function = scoring_function
        self.cv_folds = cv_folds
        self.x_test = None
        self.x_train = None
        self.y_test = None
        self.y_train = None
        self.data = None
        self.ml_pipe = MlPipeLine()

    def do_preprocessing(self, list_of_preprocessors=None):
        # Define the pipeline

        preprocessor = ColumnTransformer(
            transformers=list_of_preprocessors,
        )
        preprocessor_pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        self.x_train = pd.DataFrame(preprocessor_pipeline.fit_transform(self.x_train),
                                    columns=preprocessor_pipeline.named_steps['preprocessor'].get_feature_names_out())
        self.x_test = pd.DataFrame(preprocessor_pipeline.transform(self.x_test),
                                   columns=preprocessor_pipeline.named_steps['preprocessor'].get_feature_names_out())

    def prepare_ml(self, x_train=None, y_train=None, x_test=None, y_test=None, list_of_preprocessors=None):

        self.x_test = x_test
        self.x_train = x_train
        self.y_test = y_test
        self.y_train = y_train

        #
        if list_of_preprocessors:
            # do preprocessing
            self.do_preprocessing(list_of_preprocessors=list_of_preprocessors)


    def do_ml(self, gs_parameter=None, dict_of_steps=None):
        # build pipeline
        self.ml_pipe.build_sk_learn_pipeline(dict_of_steps=dict_of_steps)
        # Train the model
        self.ml_pipe.fit_pipeline(labels=self.y_train, features=self.x_train)
        # Evaluate the model
        self.evaluate_ml()
        if gs_parameter:
            # Do GridSearch
            self.do_grid_search(gs_parameter=gs_parameter)
            # Evaluate the model
            self.evaluate_gridsearch(full=self.output)
        self.test_ml()

    def test_ml(self):
        print(f"Test Score: {self.ml_pipe.score_pipeline(labels=self.y_test, features=self.x_test)}")

    def predict_ml(self, features=None):
        predictions = self.ml_pipe.predict_pipeline(features)
        print(predictions)
        return

    def evaluate_ml(self):
        score = self.ml_pipe.score_pipeline(labels=self.y_train, features=self.x_train)
        cross_val_score = self.cross_val_score_pipeline(labels=self.y_train, features=self.x_train)
        print(f"Train Score: {score}")
        print(f"Cross val score: {cross_val_score}, Mean: {np.mean(cross_val_score)}, Std: {np.std(cross_val_score)}")
        return

    def evaluate_gridsearch(self, full=None):
        if self.grid_search:
            print("Best Parameters:", self.grid_search.best_params_)
            print("Best Score:", self.grid_search.best_score_)
            if full:
                print("All Results:")
                results = pd.DataFrame(self.grid_search.cv_results_)
                print(results)
        else:
            raise ValueError("Grid search has not been performed yet.")

    def cross_val_score_pipeline(self, features=None, labels=None):
        return cross_val_score(
            self.ml_pipe.pipe_line,
            features,
            labels,
            cv=self.cv_folds,
            scoring=self.scoring_function
        )

    def do_grid_search(self, gs_parameter=None):
        grid_search = GridSearchCV(
            self.ml_pipe.pipe_line,
            gs_parameter,
            cv=self.cv_folds,
            scoring=self.scoring_function,
            n_jobs=-1,
            verbose=3
        )
        self.grid_search = grid_search.fit(self.x_train, self.y_train)
        return

    def do_tf(self, list_of_layers=None):

        y_train = self.y_train.to_numpy()
        x_train = self.x_train.to_numpy()

        input_model = tf.keras.Input(shape=(x_train.shape[1],))
        model_list = [input_model] + list_of_layers


        model = tf.keras.models.Sequential(model_list)

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())

        history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
        x_test = self.x_test.to_numpy()
        y_test = self.y_test.to_numpy()
        # Plotting the loss
        plt.plot(history.history['loss'], label='Train Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()

        loss, mae = model.evaluate(x_test, y_test)
        print(f"Test Loss: {loss}")
        print(f"Test Mean Absolute Error: {mae}")