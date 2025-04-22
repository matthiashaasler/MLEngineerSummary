import numpy as np
import tensorflow as tf
# from tf.keras.layers import Dense, Dtopout
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.svm import SVR

from WB.EDA import EDA
from WB.ML import PrepareData, DoMl

import os

from WB.NN import construct_dataset, NN

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define EDA class for Beer data
class BeerEDA(EDA):
    def __init__(self, project_name=None, data_source=None):
        EDA.__init__(self, data_dir="../data", project_name=project_name, data_source=data_source, figure_dir="../images")

        self.print_all_infos()

        self.plot_pandas_histograms()
        self.plot_corr_matrix(columns=self.df_columns)


        # Display the scatter matrix for thr review columns
        attributes = [att for att in self.df_columns if "review_" in att and att != "review_overall"]
        self.print_df_info(attributes, "The review columns are:")
        self.plot_scatter_matrix(columns=attributes)

        self.plot_histogram(column_name="review_overall", number_of_bins=50)

        # Display the scatter matrix for the numerical columns
        self.plot_corr_matrix(columns= self.get_numerical_columns())

        self.plot_boxplot(columns=self.get_numerical_columns())

        attributes =self.get_categorical_columns()
        self.print_df_info(attributes, "The non-numerical columns are:")

        self.data_df = self.data_df.dropna()
        self.data_df = self.data_df.drop_duplicates()
        columns_to_drop = [att for att in self.df_columns if "review_" in att and att != "review_overall"]
        columns_to_drop = columns_to_drop + ["Name", "Beer Name (Full)", "Description", 'number_of_reviews']
        print(f"The columns to drop are: {columns_to_drop}")
        self.drop_columns(columns_to_drop)
        self.data_df = self.data_df.reset_index(drop=True)
        self.save_data(file_name="truncated_data")


if __name__ == '__main__':

    # EDA of the data
    # saves a data file with the name Beer_truncated_data.pkl

    # beer_eda = BeerEDA(project_name="Beer",)

    # Loading of the data file
    data = PrepareData(data_file="Beer_truncated_data.pkl", data_dir='data', project_name='Beer')
    # Define label column and stratified column. Return of train and test set as features and label
    x_train, x_test, y_train, y_test = data.split_data(
        test_size=0.2,
        stratified=False,
        strat_column='review_overall',
        label='review_overall',
        save=True
    )


    # Define numerical and categorical columns
    categorical_cols = x_train.select_dtypes(exclude=np.number).columns.tolist()
    numerical_cols = x_train.select_dtypes(include=np.number).columns.tolist()

    # Start ML
    do_ml = DoMl(
        cv_folds=5,
        scoring_function='r2',
    )
    do_ml.prepare_ml(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                     list_of_preprocessors= [
                         ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
                         # ('passthrough', 'passthrough', numerical_cols),
                         ('scaler', StandardScaler(), numerical_cols),
                         # ('transformer', FunctionTransformer(func=np.log, inverse_func=np.exp), ['ABV', 'Body', 'Alcohol'])
                     ],

                     )
    gs_parameters  = [
        # {
        #     'scaler': ['passthrough', RobustScaler(), MinMaxScaler()],
        #     'pca': ['passthrough', PCA()],
        #     'clf': [MLPRegressor(max_iter=10000)],
        #     'clf__activation': ["logistic",  "relu"],
        #     # 'clf__hidden_layer_sizes': [(5, 2), (10, 5), (20, 10),
        #     #                             (5, 5, 2), (10, 10, 5), (20, 20, 10)],
        #     # 'clf__learning_rate': ['constant', 'adaptive'],
        #     'clf__alpha': [0.0001, 0.001, 0.01],
        # },
        {
            'scaler': ['passthrough', RobustScaler(), MinMaxScaler()],
            'clf': [SVR()],
            'clf__C': [1, 10, 100, 1000],
            'clf__gamma': [0.1, 1.0, 10]
        },
        {
            'scaler': ['passthrough', RobustScaler(), MinMaxScaler()],
            'clf': [Ridge()],
            'clf__alpha': [0.001, 0.1, 1, 10],
            'clf__solver': ['saga']
        }
    ]

    do_ml.do_ml(
        dict_of_steps={
            'scaler':MinMaxScaler(),
            'pca': PCA(),
            'clf': Ridge()
        },
        # gs_parameter=gs_parameters
    )
l2_wert = 0.0001

def get_model(size):
    model = tf.keras.Sequential([])
    model.add(tf.keras.Input(shape=(size,)))
    model.add(tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_wert)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(600, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_wert)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(800, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_wert)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(800, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_wert)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(800, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_wert)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(800, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_wert)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(800, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_wert)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(800, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_wert)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    return model

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_r2_score', factor=0.5, patience=25, cooldown=25,
                                                  min_lr=0.00001)
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_r2_score', patience=50, restore_best_weights=True)

do_ml.do_tf(
    model_function=get_model,
    epochs=1500,
    save_modul=True,
    batch_size=32,
    loss="mse",
    optimizer="adam",
    metrics=["r2_score", "mae"],
    callbacks=[lr_reducer, early_stopper]
    )