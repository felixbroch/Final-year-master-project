# backend.py
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' for non-interactive use
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


app = Flask(__name__)

# Connect to Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=False)


def dataframe_hash(df):
    """Generate a hash for a dataframe."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def generate_demand_plot(train_index, train_data):
    plt.figure(figsize=(10, 5))  # Set the size of the plot here as well
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
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(price_range, final_demand_price_function, marker='o', linestyle='-', color='b')  # Plot with blue line and circle markers
    plt.title('Demand vs. Price')  # Title of the plot
    plt.xlabel('Price')  # X-axis label
    plt.ylabel('Demand')  # Y-axis label
    plt.grid(True)  # Enable grid for better readability
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def revenue_price_plot(price_range, final_demand_price_function):
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.plot(price_range, final_demand_price_function, marker='o', linestyle='-', color='b')  # Plot with blue line and circle markers
    plt.title('Revenue vs. Price')  # Title of the plot
    plt.xlabel('Price')  # X-axis label
    plt.ylabel('Revenue')  # Y-axis label
    plt.grid(True)  # Enable grid for better readability
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def actual_predicted_plot(y_test, y_pred):
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
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data['OriginalDate'] = data['Date']
    
    # Feature Engineering
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Date'] = data['Date'].apply(lambda x: x.toordinal())
    
    # Define the point to split the data into training and validation sets (last 10%)
    split_point = int(len(data) * 0.9)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]
    
    # Prepare data for training
    X_train = train_data.drop(['Number of sales', 'OriginalDate'], axis=1)
    y_train = train_data['Number of sales']
    
    # Prepare data for validation
    X_test = test_data.drop(['Number of sales', 'OriginalDate'], axis=1)
    y_test = test_data['Number of sales']
    
    # Initialize and train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    model.fit(X_train, y_train)

    most_frequent_value = data['Price of product'].mode()[0]
    price_counts = data['Price of product'].value_counts(normalize=True)
    
    # Generate out-of-sample predictions (Forecast on the test set)
    y_pred = model.predict(X_test)
    
    # Calculate RMSE for evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, rmse, most_frequent_value, price_counts, y_pred, y_test

def feature_plot(model):
    plt.figure(figsize=(20, 10))
    ax = xgb.plot_importance(model)
    plt.title('Feature Importance')
    plt.tight_layout()  # Automatically adjust subplot parameters to give padding
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Ensure no part is clipped
    plt.close(ax.figure)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

def create_linear_function_with_coefficients(C):
    # Calculate the slope (a)
    a = (1.3 - 1) / (0.5 * C - C)
    
    # Calculate the intercept (b) using either point, here using (C, 1)
    b = 1 - a * C
    
    # Define the function using the coefficients a and b
    def linear_function(x):
        return a * x + b
    
    # Return the defined function and coefficients
    return linear_function, a, b

def dynamic_price_calculation(company_data, data_input):
    model, RMSE, most_frequent_value, price_counts, y_train, y_test = train_and_evaluate_XGBoost_model(company_data)

    min_price = data_input['Price of product'].iloc[0]
    max_price = data_input['Number of sales'].iloc[0]
    price_range = list(range(min_price, max_price))

    # Drop columns immediately to align with training data feature set
    data_input = data_input.drop(['Number of sales',], axis=1, errors='ignore')

    # Feature Engineering
    data_input['Date'] = pd.to_datetime(data_input['Date'], dayfirst=True)
    data_input['Year'] = data_input['Date'].dt.year
    data_input['Month'] = data_input['Date'].dt.month
    data_input['Day'] = data_input['Date'].dt.day
    data_input['DayOfWeek'] = data_input['Date'].dt.dayofweek
    data_input['Date'] = data_input['Date'].apply(lambda x: x.toordinal())

    # Check if the most frequent price occurs more than 80% of the time
    if price_counts.iloc[0] > 0.8:
        print("Running code A...")
        f, a, b = create_linear_function_with_coefficients(most_frequent_value)
        price_demand = []
        for price in price_range:
            demand = f(price)
            price_demand.append(demand)
        
        external_demand = model.predict(data_input)
        final_demand_price_function = price_demand * external_demand
        revenue_price_function = [x * y for x, y in zip(final_demand_price_function, price_range)]
    else:
        print('Running code B...')
        final_demand_price_function = []
        for price in price_range:
            input_data = data_input.copy()
            input_data['Price of product'] = price  # temporarily add this column if needed for the prediction

            external_demand = model.predict(input_data)
            final_demand_price_function.append(external_demand[0])

        revenue_price_function = [x * y for x, y in zip(final_demand_price_function, price_range)]

    return price_range, final_demand_price_function, revenue_price_function




@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        data2 = request.get_json()

        if not data2:
            return jsonify({'status': 'error', 'message': 'No data received'}), 400

        new_data = pd.DataFrame(data2)
        
        # Compute hash of the new data
        new_data_hash = dataframe_hash(new_data)
        print("Hash of new data:", new_data_hash)

        # Retrieve current data hash from Redis
        current_data_hash = redis_client.get('data1_hash')

        # Print the data before saving to Redis
        print("Data before saving to Redis:")
        print(new_data.head())

        # Serialize and save to Redis
        redis_client.set('data1', pickle.dumps(new_data))
        redis_client.set('data1_hash', new_data_hash)

        # Check if the new data is different from the current data
        if new_data_hash == current_data_hash:
            print("No change in data. Data hash matches the current data in Redis.")
        else:
            print("New data received. Updated in Redis.")

        # Perform further processing regardless of data change
        summary_stats = process_new_data(new_data)

        return jsonify({'status': 'success', 'message': 'Data successfully processed', 'summary_stats': summary_stats}), 200
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def process_new_data(df):
    # Example function to perform further processing on the new data
    # Add your specific data processing steps here
    summary_stats = df.describe().to_dict()
    print("Summary statistics of the new data:")
    print(summary_stats)
    # Add additional processing logic as needed
    return summary_stats










@app.route('/user_choice', methods=['POST'])
def user_choice():
    data2 = request.get_json()
    new_data = pd.DataFrame(data2)
    new_data = new_data.apply(pd.to_numeric, errors='ignore')
    
    # new_data = new_data.apply(pd.to_numeric, errors='ignore')  # Convert all columns where possible
    # new_data['Date'] = pd.to_datetime(new_data['Date'], format='%d/%m/%Y', errors='coerce')
    print(new_data.head(5))
    # Serialize the DataFrame using pickle
    serialized_df = pickle.dumps(new_data)

    # Save the serialized DataFrame in Redis
    redis_client.set('data2', serialized_df)
    # print(data)

    return jsonify({'status': 'success', 'message': 'Data successfully processed'}), 200




@app.route('/demand_plot', methods=['POST'])
def demand_plot():
    try:
        # Retrieve the serialized data from Redis
        raw_data = redis_client.get('data1')
        if not raw_data:
            return jsonify({'status': 'error', 'message': 'No data found'}), 404

        # Deserialize the data from its pickled form
        test_data = pickle.loads(raw_data)

        # Convert the deserialized data into a DataFrame
        data = pd.DataFrame(test_data)

        # Print the first 5 rows of the DataFrame to the console
        print("First 5 rows of the DataFrame:")
        print(data.head(5))

        # Ensure 'Date' column is parsed as datetime and set it as index
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        data.set_index('Date', inplace=True)

        # Extract the 'Number of sales' column for plotting
        df = data['Number of sales']

        # Generate plot data
        plot_data = generate_demand_plot(df.index, df)

        # Return the plot data as JSON
        return jsonify({
            'plot': plot_data,
            'message': 'Plot generated successfully'
        })
    except Exception as e:
        # Log the error
        logging.error(f"Error in generating demand plot: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500






@app.route('/feature_importance_plot', methods=['POST'])
def feature_importance_plot():
    try:
        raw_data = redis_client.get('data1')
        if raw_data:
            data1 = pickle.loads(raw_data)
        
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
    try:
        # Retrieve the serialized DataFrame from Redis
        serialized_df1 = redis_client.get('data1')
        # Deserialize the DataFrame
        data1 = pickle.loads(serialized_df1)

        # Retrieve the serialized DataFrame from Redis
        serialized_df2 = redis_client.get('data2')
        # Deserialize the DataFrame
        data2 = pickle.loads(serialized_df2)

        price_range, final_demand_price_function, revenue_price_function = dynamic_price_calculation(data1, data2)

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
    try:
        # Retrieve the serialized DataFrame from Redis
        serialized_df1 = redis_client.get('data1')
        # Deserialize the DataFrame
        data1 = pickle.loads(serialized_df1)

        # Retrieve the serialized DataFrame from Redis
        serialized_df2 = redis_client.get('data2')
        # Deserialize the DataFrame
        data2 = pickle.loads(serialized_df2)

        price_range, final_demand_price_function, revenue_price_function = dynamic_price_calculation(data1, data2)

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
    try:
        # Retrieve the serialized DataFrame from Redis
        serialized_df1 = redis_client.get('data1')
        # Deserialize the DataFrame
        data1 = pickle.loads(serialized_df1)

        model, rmse, most_frequent_value, price_counts, y_pred, y_test = train_and_evaluate_XGBoost_model(data1)

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
    try:
        # Retrieve the serialized DataFrame from Redis
        serialized_df1 = redis_client.get('data1')
        # Deserialize the DataFrame
        data1 = pickle.loads(serialized_df1)

        model, rmse, most_frequent_value, price_counts, y_pred, y_test = train_and_evaluate_XGBoost_model(data1)


        return jsonify({
            'RMSE': rmse,
            'message': 'DataFrame saved to redis'
        })
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return jsonify({'error': str(e)}), 500



if __name__ == "__main__":
    app.run(port = 5001)
