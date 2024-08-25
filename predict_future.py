import numpy as np

def predict_future_price(model, last_sequence, scaler):
    # Predict future price
    predicted_price = model.predict(last_sequence)
    # Inverse transform the prediction
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]
