# Stock Prediction Project

## Overview
This project provides a web application to predict stock prices using an LSTM (Long Short-Term Memory) model. Users can select a company, view the predicted stock price for the next day, and visualize historical price data.

## Project Structure

1. **`app.py`**: The main Flask application that serves the web interface, handles user input, and displays predictions and plots.

2. **`data_preparation.py`**: Contains functions for preparing and scaling stock data, and for hyperparameter tuning of the LSTM model.

3. **`model_training.py`**: Defines and trains the LSTM model on historical stock data.

4. **`model_evaluation.py`**: Evaluates the trained model’s performance and generates metrics such as RMSE (Root Mean Squared Error).

5. **`predict_future_price.py`**: Makes predictions for future stock prices using the trained model.

6. **`fetch_stock_data.py`**: (If used) Fetches historical stock data from a specified source, such as an API or CSV file.

7. **`templates/index.html`**: The HTML template for the web interface, allowing users to select a company and view predictions and plots.

8. **`static/styles.css`**: Contains styling for the web application, enhancing the user interface.

## Usage

1. **Prepare Data**: Use historical stock data files in CSV format. Ensure they include columns such as `Date` and `Close`.

2. **Train Model**: Run `model_training.py` to train the LSTM model on the prepared data.

3. **Evaluate Model**: Use `model_evaluation.py` to assess the model's performance.

4. **Predict Future Prices**: Run `predict_future_price.py` to generate predictions for the next day's stock price.

5. **Run Web Application**:
   - Execute `app.py` to start the Flask server.
   - Open a web browser and navigate to `http://127.0.0.1:5000`.
   - Select a company from the dropdown menu to see predictions and historical price plots.

6. **Fetch Stock Data**: If using `fetch_stock_data.py`, ensure it correctly retrieves and formats stock data from your data source.

## Requirements

- Python 3.x
- Flask
- TensorFlow/Keras
- scikit-learn
- Pandas
- Matplotlib
- (Optional) keras_tuner for hyperparameter tuning

Install dependencies using:
```bash
pip install -r requirements.txt
