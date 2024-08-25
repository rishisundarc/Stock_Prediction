import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import data_preparation as dp

# Load the trained model
model = load_model('lstm_stock_model.h5')

# Prepare the data again (should match what was used in training)
X, y, scaler = dp.prepare_data('AAPL_last_two_months_data.csv')

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
# Note: scaler should be fitted on the data used for the model, make sure scaler is fitted to the same column data
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse transform the true values
# Reshape y_train and y_test before inverse transformation
y_train_actual = scaler.inverse_transform(y_train)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate RMSE
train_score = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_score = np.sqrt(mean_squared_error(y_test_actual, test_predict))

print(f'Train RMSE: {train_score}')
print(f'Test RMSE: {test_score}')

# Plot the results
plt.figure(figsize=(14, 7))

# Adjust plot ranges to match actual vs predicted data
plt.plot(np.arange(len(y_train_actual)), y_train_actual, label='Actual Train')
plt.plot(np.arange(len(train_predict)), train_predict, label='Predicted Train')
plt.plot(np.arange(len(train_predict), len(train_predict) + len(test_predict)), test_predict, label='Predicted Test')
plt.plot(np.arange(len(y_train_actual), len(y_train_actual) + len(y_test_actual)), y_test_actual, label='Actual Test')

plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
