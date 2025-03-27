import pickle

import pandas as pd
from fontTools.misc.plistlib import start_dict
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from functions import get_proportions, print_df_info

if __name__ == '__main__':

    # Load the data from pickle file
    df_beer = pd.read_pickle('data/df_beer_trunc.pkl')



    train_set, test_set = train_test_split(
        df_beer,
        test_size=0.2,
        random_state=42)


    print_df_info(df_beer['Alcohol'].value_counts() / df_beer.shape[0], 'The proportion of unique values in the Alcohol column are:')
    print_df_info(df_beer['Alcohol'].value_counts(), 'The number of unique values in the Alcohol column are:')
    print_df_info(df_beer['Brewery'].value_counts() / df_beer.shape[0], 'The proportion of unique values in the Brewery column are:')
    print_df_info(df_beer['Brewery'].value_counts(), 'The number of unique values in the Brewery column are:')

    # splitter = StratifiedShuffleSplit(
    #     n_splits=10,
    #     test_size=0.2,
    #     random_state=42
    # )
    #
    # start_train_index, strat_test_index = splitter.split(df_beer, df_beer['Alcohol'])
    # strat_train_set = df_beer.loc[start_train_index]
    # strat_test_set = df_beer.loc[strat_test_index]

    # Split train und test set in features und labels
    beer_train_data = train_set.drop("review_overall", axis=1)
    beer_train_labels = train_set["review_overall"].copy()
    beer_test_data = test_set.drop("review_overall", axis=1)
    beer_test_labels = test_set["review_overall"].copy()

    # Save the train and test data
    with open('data/beer_train_data.pkl', 'wb') as f:
        # Einpacken von allen vorbereiteten Trainingsdaten
        pickle.dump(beer_train_data, f, pickle.HIGHEST_PROTOCOL)

    with open('data/housing_train_labels.pkl', 'wb') as f:
        # Einpacken der Trainingslabel
        pickle.dump(beer_train_labels, f, pickle.HIGHEST_PROTOCOL)

    with open('data/beer_test_data.pkl', 'wb') as f:
        # Einpacken von allen unvorbereiteten Testdaten
        pickle.dump(beer_test_data, f, pickle.HIGHEST_PROTOCOL)
    with open('data/beer_test_labels.pkl', 'wb') as f:
        # Einpacken von allen unvorbereiteten Testdaten
        pickle.dump(beer_test_labels, f, pickle.HIGHEST_PROTOCOL)
