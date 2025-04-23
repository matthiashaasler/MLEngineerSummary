import datetime
import inspect
import math
import os

import numpy as np
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras


def split_data(x, y, test_size, categorical_size=None):
    x , x_test, y, y_test =  train_test_split(x,y, test_size = test_size, shuffle = True)
    if categorical_size:
        y =  keras.utils.to_categorical(y, categorical_size)
        y_test =  keras.utils.to_categorical(y_test, categorical_size)
    return x, x_test, y, y_test

def construct_dataset(x=None, y=None, shuffel=True, buffer_size=512, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.cache()  # Nur bei kleinen/mittleren Datasets sinnvoll
    if shuffel:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def extract_y(y_input):
    if isinstance(y_input["x"], tf.data.Dataset):
        y_list = []
        for batch in y_input['x']:
            # batch could be (x, y) or just y
            if isinstance(batch, tuple):
                y = batch[1]
            else:
                y = batch
            y_list.append(y.numpy())
        return np.concatenate(y_list, axis=0)
        # If input is a TensorFlow tensor
    elif hasattr(y_input, 'numpy'):
        return y_input.numpy()
        # If input is already a NumPy array
    else:
        return y_input





class NN:

    def __init__(self, log_dir='nn_log', model_name='Model'):
        self.run_time = None
        self.model = None
        self.model_file = "model.keras"
        self.log_dir = ''
        self.log_dir = self.create_log_dir(log_dir, model_name)

    def create_log_dir(self, log_dir=None, model_name=None):
        dir = os.path.join(os.getcwd(), log_dir, model_name)
        try:
            os.makedirs(dir)
        except OSError as e:
            print(f"The log dir {dir} already exists.")
        finally:
            return dir

    def set_run_time(self):
        self.run_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    @property
    def history(self):
        try:
            return self.model.history.history
        except AttributeError:
            print("The Model has no history!")

    @staticmethod
    def call_model_method(model, method_name, **kwargs):
        """
        Führt eine Methode des Models aus (z. B. fit, evaluate, predict),
        filtert automatisch gültige Argumente mit `inspect`.
        """
        # Methode abrufen
        method = getattr(model, method_name)

        # Erlaubte Parameter extrahieren
        valid_params = inspect.signature(method).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        print(f"→ Rufe model.{method_name}() mit Parametern:")
        for k, v in filtered_kwargs.items():
            print(f"    {k}: {v}")
        return method(**filtered_kwargs)

    def save_model(self, file_name=None):
        self.model.save(os.path.join(self.log_dir, file_name + '_' + self.run_time + '.keras' ))
        return

    def load_model(self, model_file=None):
        self.model =  keras.models.load_model(model_file + '.keras')
        return

    #ToDo: save pictures
    def plot_metrics(self, only_loss=False):
        if not self.history:
            print("No history available!")
            return

        if only_loss:
            available_metrics = ['loss']
        else:
            available_metrics = [key for key in self.history.keys()
                                 if not key.startswith('val_')]
        num_metrics = len(available_metrics)
        cols = 2  # z.B. zwei Plots pro Zeile
        rows = math.ceil(num_metrics / cols)
        plt.figure(figsize=(cols * 6, rows * 4))

        for i, metric in enumerate(available_metrics):
            plt.subplot(rows, cols, i + 1)
            plt.plot(self.history[metric], label=f'Train {metric}')

            val_metric = f'val_{metric}'
            if val_metric in self.history:
                plt.plot(self.history[val_metric], label=f'Validation {metric}')

            plt.title(f'{metric.capitalize()} over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir,f'metrics_plot_{self.run_time}.jpg'))

        plt.show()
        return

    def set_model(self, model_file=None, load_dir=None, model=None):
        if load_dir:
            print(f"The model {model_file} is loaded from {load_dir}.")
            self.load_model(os.path.join(load_dir,model_file))
        elif model:
            print(f"The module is loaded from memory.")
            self.model = model
        else:
            raise AttributeError('No model defined!')

    def do_run(self, model_file='Model', model=None, load_dir=None, fit_data=None, eval_data=None,
               pred_data=None,
               **kwargs
               ):
        self.set_run_time()

        self.set_model(model_file=model_file, load_dir=load_dir, model=model)

        print(f"Status of the Model: {self.model.compiled}")
        if not self.model.compiled:
            self.call_model_method(self.model, "compile", **kwargs)
        else:
            print("→ Modell ist bereits kompiliert.")

        self.model.summary()

        if kwargs.get("do_fit", True):
            self.call_model_method(self.model, "fit", **fit_data, **kwargs)
            self.plot_metrics(only_loss=kwargs.get("only_loss", False))
        if kwargs.get("save_model", False):
            self.save_model(file_name=model_file)
        if kwargs.get("do_evaluate", False):
            self.call_model_method(self.model, "evaluate", **eval_data, **kwargs)

        if kwargs.get("do_predict", False):
            y_predict = self.call_model_method(model, "predict", **pred_data, **kwargs)
            y_test_np = extract_y(pred_data)
            y_predict_np = y_predict.reshape(-1)
            sns.scatterplot(x=y_test_np, y=y_predict_np, alpha=0.6)
            plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs. Predicted')
            plt.show()
            residuals = y_test_np - y_predict_np
            plt.scatter(y_predict_np, residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted')
            plt.show()
        return







