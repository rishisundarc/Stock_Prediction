import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras_tuner import Hyperband  # Correct import for Hyperband


def prepare_data(file_path, time_step):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Print column names for debugging
    print("Columns in dataset:", df.columns)
    
    # Ensure columns are correctly named
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # Handle cases where 'Date' column is missing
        print("Warning: 'Date' column not found. Make sure the dataset contains a Date column.")
    
    if 'Close' not in df.columns:
        raise KeyError("'Close' column not found in the dataset.")
    
    # Prepare features and labels
    data = df[['Close']].values
    
    # Display original data statistics
    print("Original data - Min: {:.2f}, Max: {:.2f}".format(np.min(data), np.max(data)))
    
    # Initialize and apply MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Display scaled data statistics
    print("Scaled data - Min: {:.2f}, Max: {:.2f}".format(np.min(scaled_data), np.max(scaled_data)))
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i + time_step])
        y.append(scaled_data[i + time_step])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def build_model(hp, time_step):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=50, max_value=100, step=10),
                   return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units', min_value=50, max_value=100, step=10), return_sequences=False))
    model.add(Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))
    model.add(Dense(units=1))
    
    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                  loss='mean_squared_error')
    return model

# Define time_step
time_step = 30  # Adjust based on dataset size

# Initialize the tuner
tuner = Hyperband(
    hypermodel=lambda hp: build_model(hp, time_step),
    objective='val_loss',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='intro_to_kt'
)
