import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Define the path to the images folder
IMAGES_PATH = Path() / "images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """
    Saves the figure with the given id in the images' folder.

    :param fig_id: name of the figure
    :param tight_layout:  switch to True to avoid clipping the labels
    :param fig_extension: extension of the figure
    :param resolution: resolution of the figure
    """

    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def load_data(ordner_name, csv_name):
    """
    Load the housing data from the given folder and csv file in a pandas Dataframe.

    :param ordner_name: directory name
    :param csv_name: csv file name
    :return: pandas Dataframe
    """
    ordner = Path(ordner_name)
    csv_file = ordner  / csv_name
    return pd.read_csv(csv_file)


def print_df_info(df_output, string):
    """
    Print the dataframe output.

    :param df_output: dataframe output
    :param string: string to be printed
    """
    print(f'{string}')
    print(df_output)
    print(f'-'*100)

def get_proportions(data, column=None):
    return data[column].value_counts() / len(data)


def split_data(df):
    """
    Split the data into train and test set.

    :param df: pandas dataframe
    :return: train and test set
    """
    train_set, test_set = train_test_split(
        df,
        test_size=0.2,
        random_state=42)

    return train_set, test_set


def save_data(train_data, train_labels, test_data, test_labels, file_name='beer'):
    """
    Save the train and test data.

    :param train_data: train data
    :param train_labels: train labels
    :param test_data: test data
    :param test_labels: test labels
    """
    with open(f'data/{file_name}_train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    with open(f'data/{file_name}_train_labels.pkl', 'wb') as f:
        pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)

    with open(f'data/{file_name}_test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

    with open(f'data/{file_name}_test_labels.pkl', 'wb') as f:
        pickle.dump(test_labels, f, pickle.HIGHEST_PROTOCOL)


def split_features_and_labels(df, label=''):
    """
    Split the features and labels from the dataframe.

    :param label: name of the label column
    :param df: pandas dataframe
    :return: features and labels
    """
    features = df.drop(label, axis=1)
    labels = df[label].copy()

    return features, labels