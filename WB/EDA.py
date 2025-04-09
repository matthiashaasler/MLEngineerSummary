import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px


class HandleData:

    def __init__(self, data_dir="", project_name=None, data_file=None):
        self.project_name = project_name
        self. data_save_dir = data_dir
        self.data_df = self.__read_data_to_df(data_file=data_file)
        self.__create_data_dir()


    @property
    def df_columns(self):
        return self.data_df.columns.tolist()

    def __create_data_dir(self):
        if not os.path.exists(self.data_save_dir):
            os.makedirs(self.data_save_dir)

    def __read_data_to_df(self, data_file=''):
        try:
            return pd.read_csv(os.path.join(self.data_save_dir, data_file))
        except FileNotFoundError:
            raise FileNotFoundError(f"File {data_file} not found in the directory {self.data_save_dir}")

    def get_numerical_columns(self):
        return self.data_df.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_columns(self):
        return self.data_df.select_dtypes(exclude=[np.number]).columns.tolist()


    def drop_columns(self, columns=None):
        if columns is not None:
            self.data_df.drop(columns=columns, inplace=True)
        else:
            raise ValueError("No columns to drop provided.")

    def save_data(self, file_name=''):
        data_file = os.path.join(self.data_save_dir, self.project_name + "_" + file_name + ".pkl")
        if not os.path.exists(data_file):
            self.data_df.to_pickle(data_file)
        else:
            raise IOError(f"File {data_file} already exists!")

    def load_data(self, file_name=''):
        data_file = os.path.join(self.data_save_dir, self.project_name + "_" + file_name + ".pkl")
        if os.path.exists(data_file):
            self.data_df = pd.read_pickle(data_file)
        else:
            raise IOError(f"File {data_file} does not exist!")


class PlotData:

    def __init__(self, data_df=None, project_name=None, figure_save_dir='images'):
        self.project_name = project_name
        self.figure_save_dir = figure_save_dir
        self.data_df = data_df
        self.__create_figure_dir()

    def __create_figure_dir(self):
        if not os.path.exists(self.figure_save_dir):
            os.makedirs(self.figure_save_dir)

    def plot_histogram(self, column_name, number_of_bins=50):
        try:
            fig = px.histogram(self.data_df,
                               x=column_name,
                               nbins=number_of_bins,
                               title=f"Histogram of {column_name}",
                               labels={column_name: column_name.replace('_', ' ')})  # remove underscore
            self.save_figure(file_name=f"Histogram_{column_name}", fig=fig)
            fig.show()
        except KeyError:
            print(f"Column {column_name} does not exist in the dataframe.")

    def plot_pandas_histograms(self):
        self.data_df.hist(bins=50, figsize=(20, 15))
        self.save_figure(file_name=f"All_Histograms_{self.project_name}")
        plt.show()

    def get_corr_matrix(self, columns=None):
        corr_matrix =  self.data_df[columns].corr(numeric_only=True)
        print(corr_matrix)
        return corr_matrix


    def plot_corr_matrix(self, columns=None):
        corr_matrix = self.get_corr_matrix(columns=columns)
        fig = px.imshow(corr_matrix, title="Correlation Matrix", text_auto=True)
        self.save_figure(file_name="CorrelationMatrix", fig=fig)
        fig.show()

    def plot_scatter_matrix(self, columns=None, color_by_column=None):
        fig = px.scatter_matrix(self.data_df,
                                dimensions=columns,
                                color=color_by_column, symbol=color_by_column,
                                title="Scatter matrix",
        labels={col:col.replace('_', ' ') for col in self.data_df.columns}) # remove underscore
        self.save_figure(file_name=f"ScatterMatrix", fig=fig)
        fig.show()

    def plot_scatter(self, x_column, y_column):
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data_df[x_column], self.data_df[y_column], alpha=0.5)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"Scatter Plot of {x_column} vs {y_column}")
            self.save_figure(file_name=f"Scatter_{x_column}_vs_{y_column}.png")
            plt.show()
        except KeyError:
            print(f"Columns {x_column} or {y_column} do not exist in the dataframe.")

    def save_figure(self, file_name=None, fig=None):
        data_file = os.path.join(self.figure_save_dir, f'{self.project_name}_{file_name}.png')
        file_number = 0
        while os.path.exists(data_file):
            file_number += 1
            data_file = os.path.join(self.figure_save_dir, f'{self.project_name}_{file_name}_{file_number}.png')
        if not os.path.exists(data_file):
            if fig:
                fig.write_image(data_file)
            else:
                plt.savefig(data_file)
        else:
            raise IOError(f"File {data_file} already exists!")

    def plot_boxplot(self, columns=None):
        fig = px.box(self.data_df,
                     x=columns,
                     title="Box plot",)
        self.save_figure(file_name="Boxplot", fig=fig)
        fig.show()

class EDA(HandleData, PlotData):
    def __init__(self, data_dir='data', project_name=None, data_source=None, figure_dir='images'):
        HandleData.__init__(self, data_dir=data_dir, project_name=project_name, data_file=data_source)
        PlotData.__init__(self, data_df=self.data_df, project_name=project_name, figure_save_dir=figure_dir)

    @staticmethod
    def print_df_info(df_output, heading):
        """
        Print the dataframe output.

        :param df_output: dataframe output
        :param heading: string to be printed
        """
        print("")
        print(f"-" * (len(heading) + 2))
        print(f" {heading} ")
        print(df_output)
        # print(f"-" * (len(heading) + 2))

    def print_all_infos(self):
        for heading, pandas_func in [
            ("The first 10 rows of the dataframe are:", self.data_df.head(10)),
            ("The last 10 rows of the dataframe are:", self.data_df.tail(10)),
            ("The data types info:", self.data_df.info()),
            ("The shape of the dataframe is:", self.data_df.shape),
            ("The columns of the dataframe are:", self.df_columns),
            ("The summary statistics of the dataframe are:", self.data_df.describe()),
            ("The number of unique values in the dataframe are:", self.data_df.nunique()),
            ("The number of null values in the dataframe are:", self.data_df.isna().sum()),
            ("The number of duplicated values in the dataframe are:", self.data_df.duplicated().sum())
        ]:
            self.print_df_info(pandas_func, heading)
        return
