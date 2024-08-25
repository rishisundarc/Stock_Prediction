from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import data_preparation as dp
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64
import predict_future as pfp

app = Flask(__name__)

# Dictionary of available companies and their CSV file names
COMPANIES = {
    'APPLE': 'AAPL_last_two_months_data.csv',
    'GOOGLE': 'GOOGL_last_two_months_data.csv',
    'MICROSOFT': 'MSFT_last_two_months_data.csv',
    'Samsung' : '005930.KS_last_two_months_data.csv',
    'AMAZON' : 'AMZN_last_two_months_data.csv',
    'TESLA' : 'TSLA_last_two_months_data.csv',
    'X' : 'X_last_two_months_data.csv',
    'Facebook' : 'META_last_two_months_data.csv',
    'NVIDIA Corporation' : 'NVDA_last_two_months_data.csv',
    'Johnson & Johnson' : 'JNJ_last_two_months_data.csv',
    'Toyota Motor Corporation (Japan)' : '7203.T_last_two_months_data.csv',
    'Sony Group Corporation (Japan)' : '6758.T_last_two_months_data.csv',
    'Nestlé S.A. (Switzerland)' : 'NESN.SW_last_two_months_data.csv',
    'Tata Consultancy Services' : 'TCS_last_two_months_data.csv',
    'Infosys' : 'INFY_last_two_months_data.csv',
}

# Load the model
model = load_model('lstm_stock_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    company = request.form.get('company', 'AAPL')  # Default to AAPL if no company selected
    file_name = COMPANIES.get(company, 'AAPL_last_two_months_data.csv')
    
    # Prepare data for the selected company
    time_step = 30  # Ensure this matches the time_step used in training
    X, y, scaler = dp.prepare_data(file_name, time_step)
    
    # Predict future price using the `predict_future_price.py` module
    last_sequence = X[-1].reshape(1, time_step, 1)
    predicted_price = pfp.predict_future_price(model, last_sequence, scaler)
    
    # Convert the prediction to a string
    predicted_price_str = f"{predicted_price:.2f}"
    
    # Generate a plot
    plt.figure(figsize=(10, 6))
    
    # Plot actual prices
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
    plt.plot(range(len(actual_prices)), actual_prices, label='Actual Price', color='blue')
    
    # Plot predicted price
    plt.plot(len(actual_prices), predicted_price, marker='o', markersize=10, color='red', label='Predicted Price')
    
    plt.title(f'Stock Price Prediction for {company}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template('index.html', predicted_price=predicted_price_str, plot_url=plot_url, companies=COMPANIES.keys(), selected_company=company)

if __name__ == '__main__':
    app.run(debug=True)
