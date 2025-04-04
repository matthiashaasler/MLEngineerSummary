import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, lars_path_gram
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, FunctionTransformer, MinMaxScaler

from functions import split_features_and_labels, save_data



if __name__ == '__main__':
    # Load the data from pickle file
    df_beer = pd.read_pickle('data/df_beer_trunc.pkl')

    # Split train and test data
    train_set, test_set = train_test_split(
        df_beer,
        test_size=0.2,
        random_state=42)

    # Split the features and labels
    beer_train_data, beer_train_labels = split_features_and_labels(train_set, 'review_overall')
    beer_test_data, beer_test_labels = split_features_and_labels(test_set, 'review_overall')

    # Save the train and test data
    save_data(beer_train_data, beer_train_labels, beer_test_data, beer_test_labels)

    #===================================================================================================================
    # ML

    #Reset the indices for the train data and labels
    beer_train_data.reset_index(inplace=True)
    # beer_train_labels.reset_index(inplace=True)

    # print_df_info(df_beer['Alcohol'].value_counts() / df_beer.shape[0], 'The proportion of unique values in the Alcohol column are:')
    # print_df_info(df_beer['Alcohol'].value_counts(), 'The number of unique values in the Alcohol column are:')
    # print_df_info(df_beer['Brewery'].value_counts() / df_beer.shape[0], 'The proportion of unique values in the Brewery column are:')
    # print_df_info(df_beer['Brewery'].value_counts(), 'The number of unique values in the Brewery column are:')

    # splitter = StratifiedShuffleSplit(
    #     n_splits=10,
    #     test_size=0.2,
    #     random_state=42
    # )
    #
    # start_train_index, strat_test_index = splitter.split(df_beer, df_beer['Alcohol'])
    # strat_train_set = df_beer.loc[start_train_index]
    # strat_test_set = df_beer.loc[strat_test_index]

    # Encoder
    encoder = OrdinalEncoder()
    beer_col_encoded = encoder.fit_transform(beer_train_data[['Brewery', 'Style']])
    feature_names = encoder.get_feature_names_out(['Brewery', 'Style'])
    beer_col_encoded = (pd.DataFrame(beer_col_encoded, columns=feature_names))
    beer_train_data_encoded = beer_train_data.drop(['Brewery', 'Style'], axis=1)
    beer_train_data_encoded = pd.concat([beer_train_data_encoded, beer_col_encoded], axis=1, )
    #
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Scaler
    scaler = MinMaxScaler()
    beer_train_data_scaled = scaler.fit_transform(beer_train_data_encoded)
    beer_train_data_scaled = pd.DataFrame(beer_train_data_scaled, columns=beer_train_data.columns)

    # Transformer
    # log_transformer = FunctionTransformer(func=np.log, inverse_func=np.exp)
    # beer_train_data_scaled[["ABV", 'Body', 'Alcohol']] = log_transformer.transform(
    #     beer_train_data_scaled[["ABV", 'Body', 'Alcohol']]
    # )

    # --------------------------------------------------------------------------------------------
    # PCA
    data_loss = 0.05
    pca = PCA()
    pca.fit(beer_train_data_scaled)

    beer_train_data_scaled_pca = pca.transform(beer_train_data_scaled)

    pca_column_loss_index = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 1.0 - data_loss)

    cumsum = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(cumsum)
    plt.plot(pca_column_loss_index, cumsum[pca_column_loss_index], 'ro')
    plt.show()

    # print("Eigenvektoren", pca.components_)
    print("Singulärwerte", pca.singular_values_)  # Definiert Wichtigkeit der einzelnen Komponenten
    print("Erklärungskraft", pca.explained_variance_ratio_)
    print(f'Cumulated Variance ratio{cumsum}')
    print(f'Index of the column loss: {pca_column_loss_index}')

    # --------------------------------------------------------------------------------------------
    # Linear Regression

    linear_regression = Ridge()
    # linear_regression.fit(beer_train_data_scaled_pca, beer_train_labels)
    linear_regression_cross_val_score = cross_val_score(linear_regression, beer_train_data_scaled_pca,
                                                        beer_train_labels, cv=5)
    print(f'Cross validation score: {linear_regression_cross_val_score}')

    parameter = {'alpha': np.arange(0.0, 5.0, 0.05)}  #
    #, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None}

    lr_grid_search = GridSearchCV(linear_regression, parameter, cv=5)
    lr_grid_search.fit(beer_train_data_scaled_pca, beer_train_labels)
    print(lr_grid_search.cv_results_)
    print(lr_grid_search.best_params_)
    print(lr_grid_search.best_score_)

    # --------------------------------------------------------------------------------------------

    # Pipeline

    categorical_cols = ['Brewery', 'Style']
    numerical_cols = beer_train_data.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(), categorical_cols),
            ('passthrough', 'passthrough', numerical_cols)  # Pass numerical columns unchanged
        ])

    num_pipeline = Pipeline([
        ('transformer', preprocessor),
        # ("impute", SimpleImputer(strategy="median")),
        # ("spaltenergänzung", AttributesAdder(nummern_paare, liste_neue_spaltennamen)),
        ("standardizeone", MinMaxScaler()),
        ("pca", PCA()),
        # ("standardize_two", StandardScaler()),
        ("linearregression", Ridge())
    ])

    # Define the Parameter Grid
    param_grid = {
        'linearregression__alpha': np.arange(0.0, 5.0, 0.05),
        # 'linear_regression__solver': ['sag', 'saga', 'lbfgs']  # Adjust Ridge solver
    }

    # Setup GridSearchCV
    grid_search = GridSearchCV(num_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(beer_train_data, beer_train_labels)

    # Print the best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Inspect all results
    print("\nAll Results:")
    results = pd.DataFrame(grid_search.cv_results_)
    print(
        results[['linearregression__alpha']])
