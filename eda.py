import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from functions import load_data, print_df_info, save_fig

if __name__ == '__main__':

    df_beer = load_data(
        ordner_name='data',
        csv_name='beer_profile_and_ratings.csv'
    )

    # pd.set_option("display.max_rows", None, "display.max_columns", None)

    # Display the first 10 rows of the dataframe
    print_df_info(df_beer.head(10), 'The first 10 rows of the dataframe are:')


    # Display the last 10 rows of the dataframe
    print_df_info(df_beer.tail(10), 'The last 10 rows of the dataframe are:')

    # Display the data types of the columns
    print_df_info(df_beer.info(), 'The data types info:')

    # Display the shape of the dataframe
    print_df_info(df_beer.shape, 'The shape of the dataframe is:')

    # Display the columns of the dataframe
    print_df_info(df_beer.columns.to_list(), 'The columns of the dataframe are:')

    # Display the summary statistics of the dataframe
    print_df_info(df_beer.describe(), 'The summary statistics of the dataframe are:')

    # Display the unique values of the columns
    print_df_info(df_beer.nunique(), 'The number of unique values in the dataframe are:')

    # Display the number of missing values in the dataframe
    print_df_info(df_beer.isna().sum(), 'The number of null values in the dataframe are:')

    # Display the number of duplicates in the dataframe
    print_df_info(df_beer.duplicated().sum(), 'The number of duplicated values in the dataframe are:')

    # Display numerical columns as histograms
    df_beer.hist(bins=50, figsize=(20, 15))
    save_fig('df_beer_hist')
    plt.show()

    # Display the correlation matrix
    corr_matrix = df_beer.corr(numeric_only=True)
    print_df_info(corr_matrix, 'The correlation matrix is:')

    # Display the scatter matrix for thr review columns
    attributes = [att for att in df_beer.select_dtypes(include=np.number).columns.tolist() if 'review' in att]
    print_df_info(attributes, 'The review columns are:')
    scatter_matrix(df_beer[attributes], figsize=(20,15))
    save_fig("scatter_matrix_plot")
    plt.show()

    # Drop the review columns but overall_review
    df_beer_trunc = df_beer.drop(columns=attributes)
    df_beer_trunc['review_overall'] = df_beer['review_overall']

    # Display the scatter matrix
    # Feature columns
    attributes = df_beer_trunc.select_dtypes(include=np.number).columns.tolist()
    print_df_info(attributes, 'The numerical columns of the truncated DF are:')
    scatter_matrix(df_beer_trunc[attributes], figsize=(20,15))
    save_fig("scatter_matrix_plot")
    plt.show()

    # Display scatter plot of IBU MIN und IBU MAX
    df_beer_trunc.plot(kind="scatter", x="Max IBU", y="Min IBU", figsize=(20,15),
             alpha=0.1, grid=True)
    save_fig("scatter_plot_max_vs_min_ibu")  # extra code
    plt.show()


    # Display the boxplot of the numerical columns
    df_beer_trunc.boxplot(figsize=(20, 15))
    save_fig("df_beer_trunc_boxplot")
    plt.show()

    # Dump data to pickle
    df_beer_trunc.to_pickle('data/df_beer_trunc.pkl')











