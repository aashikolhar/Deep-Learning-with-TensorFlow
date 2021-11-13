# Creates tensorflow windowed datasets with a horizon of 1 for time series analysis
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
  dataset = dataset.shuffle(shuffle_buffer)
  dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

# Converts the windowed dataset created into separate numpy arrays of data features and data labels to use separately in the fit function for training
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
