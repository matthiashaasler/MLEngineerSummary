import pickle

import pandas as pd
from fontTools.misc.plistlib import start_dict
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from functions import get_proportions, print_df_info, split_data, split_features_and_labels, save_data

if __name__ == '__main__':

    # Load the data from pickle file
    df_beer = pd.read_pickle('data/df_beer_trunc.pkl')

    # Split train and test data
    train_set, test_set = split_data(df_beer)

    # Split the features and labels
    beer_train_data, beer_train_labels = split_features_and_labels(train_set, 'review_overall')
    beer_test_data, beer_test_labels = split_features_and_labels(test_set, 'review_overall')

    # Save the train and test data
    save_data(beer_train_data, beer_train_labels, beer_test_data, beer_test_labels)

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






