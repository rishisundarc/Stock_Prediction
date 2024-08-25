# Stock Prediction Project

## Overview

This project predicts stock prices using an enhanced LSTM (Long Short-Term Memory) model. It includes additional features and hyperparameter tuning to improve the prediction accuracy. The application provides a web interface for users to select a company and view the predicted stock price along with a plot of historical prices.

## Features

- **Stock Price Prediction**: Predicts the next day's stock price based on historical data.
- **Model**: Enhanced LSTM model with additional features and hyperparameter tuning.
- **Web Interface**: Allows users to choose a company and view predictions and historical price plots.
- **Data Visualization**: Displays a plot of historical stock prices.

## Technologies Used

- **Python**: Programming language used for data processing and model training.
- **TensorFlow/Keras**: Machine learning library for building and training the LSTM model.
- **Flask**: Web framework for creating the web application.
- **Pandas**: Data manipulation and analysis library.
- **Scikit-Learn**: For preprocessing and model evaluation.
- **Matplotlib**: For plotting stock price data.

## Project Structure

- `app.py`: Main Flask application file.
- `fetch_stock_data` : Retrieve historical stock prices and other relevant data for a given company
- `data_preparation.py`: Contains functions for data preprocessing, including scaling and sequence creation.
- `model_training.py`: Script to build, train, and save the LSTM model.
- `model_evaluation.py`: Script to evaluate the model's performance.
- `predict_future_price.py`: Script to make predictions using the trained model.
- `index.html` : main webpage for a stock prediction application, providing a user interface for selecting a company and viewing predictions and plots

## Setup

### Prerequisites

- Python 3.7 or higher
- Install required Python libraries:

```bash
pip install pandas numpy scikit-learn tensorflow flask matplotlib keras-tuner
