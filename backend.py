"""
Dynamic Pricing Dashboard - Backend API

This module contains the backend API for the dynamic pricing dashboard built with Flask.
It handles data processing, machine learning model training, and plot generation for
the pricing optimisation engine.

The backend implements:
- XGBoost regression for demand forecasting
- Price elasticity analysis
- Revenue optimisation calculations
- Data caching with Redis
- Plot generation for visualisation

Author: Felix Brochier
Institution: Imperial College London
Supervisor: Professor Pierre Pinson
Project: "Dynamic Pricing Made Accessible: A Dashboard Which Provides Optimal Price 
         Recommendations Through a Transparent and Seamless Process"
"""

# backend.py
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for web deployment
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import base64
from io import BytesIO
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import redis
import pickle
import logging
import hashlib

# Initialise Flask application
app = Flask(__name__)

# Connect to Redis for data caching and session management
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=False)


def dataframe_hash(df):
    """
    Generate a SHA256 hash for a DataFrame to enable caching and change detection.
    
    Args:
        df (pd.DataFrame): DataFrame to hash
        
    Returns:
        str: SHA256 hash of the DataFrame contents
    """
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def generate_demand_plot(train_index, train_data):
    """
    Generate a time series plot showing demand variations over time.
    
    Args:
        train_index: Time index for x-axis
        train_data: Demand values for y-axis
        
    Returns:
        str: Base64-encoded PNG image of the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_index, train_data, label='Training Data', color='blue')
    plt.title('Demand variation over time')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def demand_price_plot(price_range, final_demand_price_function):
    """
    Generate a plot showing the relationship between price and demand.
    
    Args:
        price_range: Array of price values
        final_demand_price_function: Corresponding demand values
        
    Returns:
        str: Base64-encoded PNG image of the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(price_range, final_demand_price_function, marker='o', linestyle='-', color='b')
    plt.title('Demand vs. Price')
    plt.xlabel('Price')
    plt.ylabel('Demand')
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def revenue_price_plot(price_range, final_demand_price_function):
    """
    Generate a plot showing the relationship between price and revenue.
    
    Args:
        price_range: Array of price values
        final_demand_price_function: Revenue values at each price point
        
    Returns:
        str: Base64-encoded PNG image of the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(price_range, final_demand_price_function, marker='o', linestyle='-', color='b')
    plt.title('Revenue vs. Price')
    plt.xlabel('Price')
    plt.ylabel('Revenue')
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def actual_predicted_plot(y_test, y_pred):
    """
    Generate a scatter plot comparing actual vs predicted values for model validation.
    
    Args:
        y_test: Actual values from test set
        y_pred: Predicted values from model
        
    Returns:
        str: Base64-encoded PNG image of the plot
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def train_and_evaluate_XGBoost_model(data):
    """
    Train and evaluate an XGBoost model for demand forecasting.
    
    This function performs the core machine learning pipeline:
    1. Feature engineering (date components, etc.)
    2. Train/test split (90/10)
    3. XGBoost model training
    4. Model evaluation with RMSE
    5. Price analysis for optimisation
    
    Args:
        data (pd.DataFrame): Historical sales data with Date, Number of sales, Price columns
        
    Returns:
        tuple: (trained_model, rmse_score, most_frequent_price, price_distribution, 
                predictions, actual_values)
    """
    # Convert 'Date' column to datetime and preserve original
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data['OriginalDate'] = data['Date']
    
    # Feature Engineering: Extract temporal features
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Date'] = data['Date'].apply(lambda x: x.toordinal())
    
    # Split data into training (90%) and validation (10%) sets
    split_point = int(len(data) * 0.9)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    # Prepare features and target variables
    X_train = train_data.drop(['Number of sales', 'OriginalDate'], axis=1)
    y_train = train_data['Number of sales']
    
    # Prepare validation data
    X_test = test_data.drop(['Number of sales', 'OriginalDate'], axis=1)
    y_test = test_data['Number of sales']
    
    # Initialise and train XGBoost model with regression objective
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    model.fit(X_train, y_train)

    # Analyse price distribution for optimisation strategy selection
    most_frequent_value = data['Price of product'].mode()[0]
    price_counts = data['Price of product'].value_counts(normalize=True)
    
    # Generate predictions on test set for model evaluation
    y_pred = model.predict(X_test)
    
    # Calculate Root Mean Square Error for model performance assessment
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, rmse, most_frequent_value, price_counts, y_pred, y_test

def feature_plot(model):
    """
    Generate a feature importance plot for the trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        
    Returns:
        str: Base64-encoded PNG image of the feature importance plot
    """
    plt.figure(figsize=(20, 10))
    ax = xgb.plot_importance(model)
    plt.title('Feature Importance')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(ax.figure)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

def create_linear_function_with_coefficients(C):
    """
    Create a linear price-demand function based on economic theory.
    
    This function creates a linear demand function where:
    - At price C, demand = 1 (baseline)
    - At price 0.5*C, demand = 1.3 (30% increase when price halves)
    
    Args:
        C (float): Reference price point
        
    Returns:
        tuple: (linear_function, slope, intercept)
    """
    # Calculate the slope (a) using two points: (C, 1) and (0.5*C, 1.3)
    a = (1.3 - 1) / (0.5 * C - C)
    
    # Calculate the intercept (b) using point (C, 1)
    b = 1 - a * C
    
    # Define the linear function using calculated coefficients
    def linear_function(x):
        return a * x + b
    
    return linear_function, a, b

def dynamic_price_calculation(company_data, data_input):
    """
    Calculate optimal pricing based on user inputs and historical data.
    
    This function implements two pricing strategies:
    1. Linear approximation (when price variation is low)
    2. ML-based prediction (when price variation is high)
    
    Args:
        company_data (pd.DataFrame): Historical sales data
        data_input (pd.DataFrame): User-specified parameters for optimisation
        
    Returns:
        tuple: (price_range, demand_predictions, revenue_predictions)
    """
    # Train model and get price distribution analysis
    model, RMSE, most_frequent_value, price_counts, y_train, y_test = train_and_evaluate_XGBoost_model(company_data)

    # Extract price range from user input
    min_price = data_input['Price of product'].iloc[0]
    max_price = data_input['Number of sales'].iloc[0]
    price_range = list(range(min_price, max_price))

    # Remove target column to align with training features
    data_input = data_input.drop(['Number of sales',], axis=1, errors='ignore')

    # Apply same feature engineering as training data
    data_input['Date'] = pd.to_datetime(data_input['Date'], dayfirst=True)
    data_input['Year'] = data_input['Date'].dt.year
    data_input['Month'] = data_input['Date'].dt.month
    data_input['Day'] = data_input['Date'].dt.day
    data_input['DayOfWeek'] = data_input['Date'].dt.dayofweek
    data_input['Date'] = data_input['Date'].apply(lambda x: x.toordinal())

    # Strategy selection based on price variation in historical data
    if price_counts.iloc[0] > 0.8:
        print("Running code A...")  # Linear approximation for low price variation
        f, a, b = create_linear_function_with_coefficients(most_frequent_value)
        price_demand = []
        for price in price_range:
            demand = f(price)
            price_demand.append(demand)
        
        # Predict external demand factors
        external_demand = model.predict(data_input)
        final_demand_price_function = price_demand * external_demand
        revenue_price_function = [x * y for x, y in zip(final_demand_price_function, price_range)]
    else:
        print('Running code B...')  # ML-based prediction for high price variation
        final_demand_price_function = []
        for price in price_range:
            input_data = data_input.copy()
            input_data['Price of product'] = price

            # Predict demand for each price point
            external_demand = model.predict(input_data)
            final_demand_price_function.append(external_demand[0])

        # Calculate revenue for each price point
        revenue_price_function = [x * y for x, y in zip(final_demand_price_function, price_range)]

    return price_range, final_demand_price_function, revenue_price_function

# Flask API Endpoints

@app.route('/process_data', methods=['POST'])
def process_data():
    """
    Process uploaded dataset and cache it in Redis.
    
    This endpoint receives the initial dataset upload, performs data validation,
    caches the data in Redis, and triggers initial data processing.
    
    Returns:
        JSON response with processing status and summary statistics
    """
    try:
        data2 = request.get_json()

        if not data2:
            return jsonify({'status': 'error', 'message': 'No data received'}), 400

        new_data = pd.DataFrame(data2)
        
        # Compute hash of the new data for change detection
        new_data_hash = dataframe_hash(new_data)
        print("Hash of new data:", new_data_hash)

        # Retrieve current data hash from Redis for comparison
        current_data_hash = redis_client.get('data1_hash')

        # Print the data before saving to Redis for debugging
        print("Data before saving to Redis:")
        print(new_data.head())

        # Serialise and save DataFrame to Redis with hash
        redis_client.set('data1', pickle.dumps(new_data))
        redis_client.set('data1_hash', new_data_hash)

        # Check if the new data is different from the current data
        if new_data_hash == current_data_hash:
            print("No change in data. Data hash matches the current data in Redis.")
        else:
            print("New data received. Updated in Redis.")

        # Perform data processing regardless of change status
        summary_stats = process_new_data(new_data)

        return jsonify({'status': 'success', 'message': 'Data successfully processed', 'summary_stats': summary_stats}), 200
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def process_new_data(df):
    """
    Process new data and generate summary statistics.
    
    Args:
        df (pd.DataFrame): New data to process
        
    Returns:
        dict: Summary statistics of the processed data
    """
    # Generate summary statistics for the new data
    summary_stats = df.describe().to_dict()
    print("Summary statistics of the new data:")
    print(summary_stats)
    # Additional processing logic can be added here
    return summary_stats










@app.route('/user_choice', methods=['POST'])
def user_choice():
    """
    Process user-specified parameters for price optimisation.
    
    This endpoint receives user inputs including external parameters and
    price range for optimisation, processes and caches them in Redis.
    
    Returns:
        JSON response confirming successful parameter storage
    """
    data2 = request.get_json()
    new_data = pd.DataFrame(data2)
    # Convert numeric columns where possible
    new_data = new_data.apply(pd.to_numeric, errors='ignore')
    
    print(new_data.head(5))
    
    # Serialise and store user parameters in Redis
    serialised_df = pickle.dumps(new_data)
    redis_client.set('data2', serialised_df)

    return jsonify({'status': 'success', 'message': 'Data successfully processed'}), 200

@app.route('/demand_plot', methods=['POST'])
def demand_plot():
    """
    Generate demand variation plot over time.
    
    This endpoint retrieves historical data from Redis and generates
    a time series plot showing demand variations.
    
    Returns:
        JSON response containing base64-encoded plot image
    """
    try:
        # Retrieve the serialised data from Redis
        raw_data = redis_client.get('data1')
        if not raw_data:
            return jsonify({'status': 'error', 'message': 'No data found'}), 404

        # Deserialise the data from its pickled form
        test_data = pickle.loads(raw_data)
        data = pd.DataFrame(test_data)

        # Debug: Print the first 5 rows of the DataFrame
        print("First 5 rows of the DataFrame:")
        print(data.head(5))

        # Prepare data for plotting: parse dates and set as index
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        data.set_index('Date', inplace=True)
        df = data['Number of sales']

        # Generate plot data
        plot_data = generate_demand_plot(df.index, df)

        return jsonify({
            'plot': plot_data,
            'message': 'Plot generated successfully'
        })
    except Exception as e:
        logging.error(f"Error in generating demand plot: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/feature_importance_plot', methods=['POST'])
def feature_importance_plot():
    """
    Generate feature importance plot for the trained model.
    
    This endpoint trains the XGBoost model and generates a plot showing
    the importance of different features in predicting demand.
    
    Returns:
        JSON response containing base64-encoded plot image
    """
    try:
        # Retrieve historical data from Redis
        raw_data = redis_client.get('data1')
        if raw_data:
            data1 = pickle.loads(raw_data)
        
        # Train model and generate feature importance plot
        model, RMSE, a, b, y_train, y_test = train_and_evaluate_XGBoost_model(data1)
        print("Currently checking if the model is valid")
        
        if isinstance(model, xgb.XGBRegressor):
            print("Model is valid.")
            feature_import_plot = feature_plot(model)
            return jsonify({'plot': feature_import_plot})
        else:
            print("Invalid model instance.")
            return jsonify({'error': 'Invalid model instance'}), 500
            
    except Exception as e:
        print(f"Error generating feature importance plot: {str(e)}")
        return jsonify({'error': str(e)}), 500




@app.route('/price_demand_function_plot', methods=['POST'])
def price_demand_function_plot():
    """
    Generate price-demand relationship plot.
    
    This endpoint combines historical data with user parameters to calculate
    and plot the relationship between price and demand.
    
    Returns:
        JSON response containing base64-encoded plot image
    """
    try:
        # Retrieve historical data from Redis
        serialised_df1 = redis_client.get('data1')
        data1 = pickle.loads(serialised_df1)

        # Retrieve user parameters from Redis
        serialised_df2 = redis_client.get('data2')
        data2 = pickle.loads(serialised_df2)

        # Calculate price-demand relationship
        price_range, final_demand_price_function, revenue_price_function = dynamic_price_calculation(data1, data2)

        # Generate plot
        plot_data = demand_price_plot(price_range, final_demand_price_function)

        return jsonify({
            'plot': plot_data,
            'message': 'DataFrame saved to redis'
        })
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/revenue_demand_function_plot', methods=['POST'])
def revenue_demand_function_plot():
    """
    Generate revenue-price relationship plot.
    
    This endpoint calculates and plots the relationship between price and revenue,
    which is used to identify the optimal price point for revenue maximisation.
    
    Returns:
        JSON response containing base64-encoded plot image
    """
    try:
        # Retrieve historical data from Redis
        serialised_df1 = redis_client.get('data1')
        data1 = pickle.loads(serialised_df1)

        # Retrieve user parameters from Redis
        serialised_df2 = redis_client.get('data2')
        data2 = pickle.loads(serialised_df2)

        # Calculate revenue-price relationship
        price_range, final_demand_price_function, revenue_price_function = dynamic_price_calculation(data1, data2)

        # Generate plot
        plot_data = revenue_price_plot(price_range, revenue_price_function)

        return jsonify({
            'plot': plot_data,
            'message': 'DataFrame saved to redis'
        })
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_accuracy', methods=['POST'])
def model_accuracy():
    """
    Generate model accuracy visualisation.
    
    This endpoint creates a scatter plot comparing actual vs predicted values
    to visualise model performance and accuracy.
    
    Returns:
        JSON response containing base64-encoded plot image
    """
    try:
        # Retrieve historical data from Redis
        serialised_df1 = redis_client.get('data1')
        data1 = pickle.loads(serialised_df1)

        # Train model and get predictions
        model, rmse, most_frequent_value, price_counts, y_pred, y_test = train_and_evaluate_XGBoost_model(data1)

        # Generate accuracy plot
        plot_data = actual_predicted_plot(y_test, y_pred)

        return jsonify({
            'plot': plot_data,
            'message': 'DataFrame saved to redis'
        })
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/RMSE', methods=['POST'])
def RMSE():
    """
    Calculate and return Root Mean Square Error for model evaluation.
    
    This endpoint trains the model and returns the RMSE score as a
    quantitative measure of model accuracy.
    
    Returns:
        JSON response containing RMSE score
    """
    try:
        # Retrieve historical data from Redis
        serialised_df1 = redis_client.get('data1')
        data1 = pickle.loads(serialised_df1)

        # Train model and calculate RMSE
        model, rmse, most_frequent_value, price_counts, y_pred, y_test = train_and_evaluate_XGBoost_model(data1)

        return jsonify({
            'RMSE': rmse,
            'message': 'DataFrame saved to redis'
        })
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Flask application entry point
if __name__ == "__main__":
    app.run(port=5001)
