from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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