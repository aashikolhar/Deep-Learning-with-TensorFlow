# Function to create tensorflow windowed datasets with a horizon of 1 for time series analysis
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
  dataset = dataset.shuffle(shuffle_buffer)
  dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset


# Function to convert the windowed dataset created into separate numpy arrays of data features and data labels to use separately in the fit function for training
def dataset_to_numpy(ds):
    """
    Convert tensorflow dataset to numpy arrays
    """
    features = []
    labels = []

    # Iterate over a dataset
    for i, (feature, label) in enumerate(ds):
        features.append(feature.numpy())
        labels.append(label.numpy())
    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)
    return features, labels
  
  
  # Function to plot time series data
 def plot_time_series(timesteps, values, format=".", start=0, end=None, label=None):
   """
   Plots time steps (a series of points in time) against values (a series of values across timesteps)
   Parameters
   ---------
   timesteps : array of timestep values
   values : array of values across time
   format : style of plot (default is scatter)
   start : where to start the plot (setting a value will index from start of timesteps)
   end : where to end the plot (setting a value will indec the end of timesteps)
   label : labels to show on plot about values, default= None
   """
   #plot the series
   plt.plot(timesteps[start:end], values[start:end], format, label=label)
   plt.xlabel("Time")
   plt.ylabel("BTC Price")
   if label:
     plt.legend(fontsize=14) # make label bigger
   plt.grid(True)
  
  # Function to calculate the mean_absolute_scaled error in time series anal
 def mean_absolute_scaled_error(y_true, y_pred):
  """
  Implementing MASE (assuming no seasonality of data).
  """
  mae = tf.reduce_mean(tf.abs(y_true-y_pred))

  # Find mae of Naive forecast (no seasonality)
  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:]-y_true[:-1])) # our seasonality is 1 day (hence the shift of 1)

  return mae/mae_naive_no_season 


# Function to take in model predictions and truth values and return evaluation metrics
def evaluate_preds(y_true, y_pred):
  # Make sure float32 datatype for metric calculations
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Function to calculate various evaluation metrics
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)

  return {"mae" : mae.numpy(),
          "mse" : mse.numpy(),
          "rmse" : rmse.numpy(),
          "mape" : mape.numpy(),
          "mase" : mase.numpy()}

# Function to create the train and test splits
def make_train_test_splits(windows, labels, test_split=0.2):
   """
   Splits matching pairs of windows and labels into train and test splits.
   """
   split_size = int(len(windows) * (1-test_split)) # this will default to 80% train and 20% test
   train_windows = windows[:split_size]
   train_labels = labels[:split_size]
   test_windows = windows[split_size:]
   test_labels = labels[split_size:]
   return train_windows, test_windows, train_labels, test_labels

# Function to create and implement ModelCheckpoint callback with a specific filename
def create_model_checkpoint(model_name, save_path="model_experiments"):
  return tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_path, model_name),
                                            verbose=0, # only output a limited amount of text
                                            save_best_only=True)

# Function to make predictions on the input data
def make_preds(model, input_data):
  """
  Uses model to make predictions on input data
  """
  forecast = model.predict(input_data)
  return tf.squeeze(forecast) # retrun 1D array of predictions
