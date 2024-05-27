import tensorflow as tf
import pandas as pd
import numpy as np



def train_test(X,y,window=10,split=0.8):
  """
  Splits train-test with index for a window (default=10) data
  """
  length = len(X)

  split_index = int(split * length)
  split_index = (split_index // window) * window

  X_train = X[:split_index] 
  y_train =y[:split_index]
  X_test = X[split_index:]
  y_test = y[split_index:]
  print(f"Shape of X_train = {X_train.shape}")
  print(f"Shape of y_train = {y_train.shape}")
  print(f"Shape of X_test = {X_test.shape}")
  print(f"Shape of y_test = {y_test.shape}")
  return X_train, X_test, y_train, y_test



def window_after(table,split_window_column, condition=True, window=10):
  """
  Given a table and a reference column, it will creat a new dataframe with a window (default=10) rows starting the True condition of the reference column 
  """
  print(f"Lenght of original Dataframe = {len(table)}\n")
  new_table = pd.DataFrame(columns=table.columns)
  for index, row in table.iterrows():
    if row[split_window_column] == condition:
      current_index = table.index.get_loc(index)
      new_table = pd.concat([new_table, table.iloc[current_index:current_index+window]])
  print(f"Lengh of New Dataframe = {len(new_table)}")
  return new_table

def window_before(table,split_window_column, condition=True, window=10):
  """
  Given a table and a reference column, it will creat a new dataframe with a window (default=10) rows starting the True condition of the reference column
  """
  print(f"Lenght of original Dataframe = {len(table)}\n")
  new_table = pd.DataFrame(columns=table.columns)
  for index, row in table.iterrows():
    if row[split_window_column] == condition:
      current_index = table.index.get_loc(index)
      new_table = pd.concat([new_table, table.iloc[current_index-window+1:current_index+1]])
  print(f"Lengh of New Dataframe = {len(new_table)}")
  return new_table



def remove_na(table, axis=0):
  """
  Removes na, default is rows. Shows nro of rows delected.
  """
  len_before = len(table)
  print(f"Before removing NA. N.Rows = {len_before}\n ")
  print(table.isna().sum())
  table = table.dropna(axis=axis)
  len_after =len(table)
  print("")
  print(f"After removing NA. N.Rows = {len_after}\n ")
  print(table.isna().sum())
  print("")
  print(f"Rows delected = {len_before-len_after}")
  return table


def delete_columns(tabla,index_start=False,index_stop=False,columns=False):
  """
  Given a table, it eliminates de columns in the given indeces and the columns (as a list) named
  """
  if index_start is not False and index_stop is not False:
    delected_columns = tabla.columns[index_start:index_stop+1]
    tabla = tabla.drop(columns=delected_columns)
  if columns is not False:
    for column in columns:
      tabla = tabla.drop(columns=column)
  return tabla


def turn_dtype(tabla, columns, dtype="float32"):
  """
  Returns a table with the list of columns in dtype
  """
  for column in columns:
    tabla[column] = tabla[column].astype(dtype)
  return tabla


def turn_onehot(tabla, columns, dtype="float32"):
  """
  Turns a list of objets/bool or categories columns into one_hot
  """
  for column in columns:
      dummies = pd.get_dummies(data=tabla[column],dtype=dtype)
      tabla = pd.concat([tabla, dummies],axis=1)
      tabla = tabla.drop(columns=column)
  return tabla

import os
import datetime

def checkpoint(model_name, path="/content/drive/MyDrive/Fendi Mio/Modelos/",monitor="val_loss"):
  date =datetime.datetime.now().strftime("%Y-%m-%d")
  filepath = os.path.join(path,date,model_name)
  return tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                            monitor=monitor,
                                            verbose=1,
                                            save_best_only=True)


# def create_sequences(data, targets, sequence_length, sequence_stride):
#   """
#   Creates both an array of train and test sequences
#   """
#     sequences = []
#     labels = []
#     for i in range(0, len(data) - sequence_length + 1, sequence_stride):
#         sequences.append(data[i:i+sequence_length])
#         labels.append(targets[i])  # Assuming the target corresponds to the first element of each sequence
#     return np.array(sequences), np.array(labels)

from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay

def pred_eval_cm(model,test_data):
  """
  Predicts, reshapes, evaluates and makes a ConfMatrix

  **Return: pred, y_true, model_eval
  """
  pred_prob = model.predict(test_data)
  pred = np.round(pred_prob)
  pred = np.squeeze(pred)

  y_true_list = []

  for _, targets in test_data:
    y_true_list.append(targets)
  
  y_true = np.concatenate(y_true_list)

  model_eval = classification_report(y_true,pred)
  print(model_eval)

  cm=confusion_matrix(y_true, pred)
  display = ConfusionMatrixDisplay(cm)
  display.plot()

  return pred, y_true, model_eval


from sklearn.preprocessing import StandardScaler

def exclude_normalize(df,excluded):
  """
  Normaliza las columnas en df siempre y cuando no esten en la lista de "excluded"
  """
  for column in df.columns:
    if column not in excluded:
      scaler = StandardScaler()
      column_data = df[column].values.reshape(-1,1)
      df[column] = scaler.fit_transform(column_data).flatten()
  return df


from sklearn.preprocessing import StandardScaler

def normalize(df,del_columns):
  """
  Normaliza las del_columns en df.
  """
  for column in df.columns:
    if column in del_columns:
      scaler = StandardScaler()
      column_data = df[column].values.reshape(-1,1)
      df[column] = scaler.fit_transform(column_data).flatten()
  return df

def model_predict(model,test_data):
  """
  reshapes and predict for test_data tensor. Returns pred y_true
  """
  pred_prob = model.predict(test_data)
  pred = np.round(pred_prob)
  pred = np.squeeze(pred)

  y_true_list = []

  for _, targets in test_data:
    y_true_list.append(targets)
  
  y_true = np.concatenate(y_true_list)

  return pred,y_true


from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay


def true_dict(y_true,pred,ML=True):
  """
  prints a classification report and returns a dict of presicion and accuracy.
  Use ML=True for ML and ML=False for Neural Networks
  """
  if ML:
    cero="0"
    one="1"
  else:
    cero="0.0"
    one="1.0"
  
  model_dict = classification_report(y_true,pred,output_dict=True)

  precision_0 = model_dict[cero]["precision"]
  precision_1 = model_dict[one]["precision"]
  macro_avg_precision = model_dict["macro avg"]["precision"]
  recall_1 = model_dict[one]["recall"]
  accuracy = model_dict["accuracy"]

  model_precision = {"precision_0": precision_0,
                          "precision_1": precision_1,
                          "macro_avg_precision": macro_avg_precision,
                          "recall_1": recall_1,
                          "accuracy": accuracy}

  model_eval = classification_report(y_true,pred)
  print(model_eval)

  cm=confusion_matrix(y_true, pred)
  display = ConfusionMatrixDisplay(cm)
  display.plot()

  return model_precision


# all_model_results = pd.DataFrame({"model_1": model_dict1,
                                  # "model_2": model_dict2})

# all_model_results = all_model_results.transpose()
# all_model_results
# all_model_results.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0));


import joblib
import os

def load_and_scale(df, path="/content/drive/My Drive/Fendi/"):
    """
    Loads the scalers from the specified folder and transforms the corresponding columns in df.
    """
    # Load the scalers from the file
    full_path = os.path.join(path,'scalers.pkl')
    scalers = joblib.load(full_path)

    transformed_df = df.copy()

    # Apply the scalers to the corresponding columns
    for column, scaler in scalers.items():
        if column in transformed_df.columns:
            column_data = transformed_df[column].values.reshape(-1, 1)
            transformed_df[column] = scaler.transform(column_data).flatten()

    return transformed_df

from sklearn.preprocessing import StandardScaler
import os

def save_normalize(data,excluded,folder,save=False,path="/content/drive/MyDrive/Fendi Mio/EMA/Scalers/"):
  """
  Normaliza las columnas en df siempre y cuando no esten en la lista de "excluded"
  """
  scalers = {}
  df = data.copy()

  for column in df.columns:
    if column not in excluded:
      scaler = StandardScaler()
      column_data = df[column].values.reshape(-1,1)
      df[column] = scaler.fit_transform(column_data).flatten()
      scalers[column] = scaler
      
  if save==True:
    full_path = os.path.join(path, folder)
    joblib.dump(scalers, os.path.join(full_path, 'scalers.pkl'))

  return df
