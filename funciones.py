import tensorflow as tf
import pandas as pd
import numpy as np

-----------------------------------------------------------------------
-----------------------------------------------------------------------

# def turn_float32_onehot(dataframe):
#   """
#   Turns all dtypes of a DF into float32 or one hot
#   """
#   for column in dataframe.columns:
#     if dataframe[column].dtype == "float64" or dataframe[column].dtype == "int64":
#       dataframe[column] = dataframe[column].astype("float32")
#     elif dataframe[column].dtype == "bool" or dataframe[column].dtype == "object":
#       dummies = pd.get_dummies(data=tabla[column],dtype="float32")
#       #dummies_float = dummies.astype("float32")
#       dataframe = pd.concat([dataframe, dummies],axis=1) 
#       dataframe = dataframe.drop(columns=column)
#     else:
#       continue
  
#   return dataframe

-----------------------------------------------------------------------
-----------------------------------------------------------------------


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

-----------------------------------------------------------------------
-----------------------------------------------------------------------


def window(table,split_window_column, condition=True, window=10):
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



-----------------------------------------------------------------------
-----------------------------------------------------------------------


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
