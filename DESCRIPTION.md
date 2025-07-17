# Technical Documentation: Dynamic Pricing Dashboard

## Table of Contents
1. [Frontend Architecture (app.py)](#frontend-architecture-apppy)
2. [Backend Architecture (backend.py)](#backend-architecture-backendpy)
3. [Data Flow](#data-flow)
4. [API Endpoints](#api-endpoints)
5. [User Interface Components](#user-interface-components)
6. [Database and Caching](#database-and-caching)
7. [Error Handling](#error-handling)

---

## Frontend Architecture (app.py)

### Core Structure

The frontend is built using Shiny for Python and follows a reactive programming paradigm with three main components:

- **UI Layout**: Defines the visual structure including sidebars, tabs, and input forms
- **Server Logic**: Handles user interactions, data processing, and backend communication
- **Application Instance**: Combines UI and server logic into a deployable web application

### UI Components

#### 1. **Layout Structure**
```
┌─────────────────────────────────────────────────────────────┐
│                    Page Header & Navigation                 │
├─────────────────┬───────────────────────────────────────────┤
│                 │                                           │
│    Sidebar      │             Main Content                  │
│   - File Upload │        ┌─────────────────────────┐        │
│   - Test Data   │        │     Tab 1: Demand       │        │
│   - Column List │        │      Analysis           │        │
│                 │        ├─────────────────────────┤        │
│                 │        │  Tab 2: Revenue Mgmt    │        │
│                 │        ├─────────────────────────┤        │
│                 │        │  Tab 3: Methodology     │        │
│                 │        └─────────────────────────┘        │
└─────────────────┴───────────────────────────────────────────┘
```

#### 2. **Dynamic Input Generation**
The system automatically generates input fields based on the uploaded dataset's columns. This component:
- **Input**: CSV file with column headers
- **Process**: Analyses column structure and creates corresponding form fields
- **Output**: Interactive form with fields for each data parameter plus price range inputs

### Server Logic

#### 1. **Reactive Calculations**
These monitor data changes and trigger dependent computations:
- **Input**: File upload events or user interactions
- **Process**: Parses uploaded files and maintains data state
- **Output**: Clean pandas DataFrames ready for analysis

#### 2. **Render Functions**
These generate visual outputs based on current application state:
- **Input**: Processed data and user parameters
- **Process**: Requests visualisations from backend, handles image decoding
- **Output**: Interactive plots and text explanations

### Communication with Backend

The frontend communicates with the Flask backend through RESTful API calls:
- **Input**: User data, parameters, and visualisation requests
- **Process**: HTTP POST requests with JSON payloads to specific endpoints
- **Output**: JSON responses containing processed data, plots (base64-encoded), and status information

---

## Backend Architecture (backend.py)

### Core Structure

The backend is built using Flask and provides RESTful API endpoints that handle:
- **Data Processing**: Validates and preprocesses uploaded datasets
- **Machine Learning**: Trains models and generates predictions
- **Visualisation**: Creates plots and charts for frontend display
- **Session Management**: Maintains user state through Redis caching

### Key Components

#### 1. **Data Processing Pipeline**
The core machine learning pipeline processes historical sales data:
- **Input**: CSV data with Date, Number of sales, Price, and external parameters
- **Process**: Data cleaning, feature engineering, train/test split (90/10), model training, evaluation
- **Output**: Trained XGBoost model, RMSE score, price distribution analysis, predictions

#### 2. **Feature Engineering**
Temporal features are extracted from date information to improve model performance:
- **Input**: Date column in DD/MM/YYYY format
- **Process**: Extracts year, month, day, day of week, and converts to ordinal format
- **Output**: Enhanced dataset with temporal features ready for machine learning

#### 3. **Price Optimisation Strategies**

The system employs two distinct approaches based on historical price variation:

**Strategy A: Linear Approximation** (Low price variation scenarios)
- **Condition**: When >80% of historical data uses the same price
- **Input**: Most frequent price point and user parameters
- **Process**: Applies economic theory with linear demand function (demand decreases linearly with price)
- **Output**: Price-demand relationship based on theoretical elasticity

**Strategy B: ML-Based Prediction** (High price variation scenarios)  
- **Condition**: When historical data shows diverse pricing patterns
- **Input**: User parameters and price range for testing
- **Process**: Uses trained XGBoost model to predict demand at each price point
- **Output**: Data-driven price-demand relationship based on actual historical patterns

### Machine Learning Pipeline

#### 1. **Model Training**
The system uses XGBoost regression for demand prediction:
- **Input**: Preprocessed features (excluding price) and historical demand data
- **Process**: Gradient boosting with squared error optimisation and fixed random seed for reproducibility
- **Output**: Trained model capable of predicting demand based on external factors

#### 2. **Model Evaluation**
Performance assessment ensures model reliability:
- **Input**: Test dataset (10% of historical data) and trained model
- **Process**: Generates predictions and calculates Root Mean Square Error (RMSE)
- **Output**: Quantitative accuracy score displayed to users for transparency

#### 3. **Feature Importance Analysis**
Understanding which factors most influence demand:
- **Input**: Trained XGBoost model
- **Process**: Extracts and ranks feature importance scores
- **Output**: Visualisation showing relative influence of each parameter on demand predictions

### Plot Generation

All visualisations are generated server-side and delivered to the frontend:
- **Input**: Processed data, model results, or user parameters
- **Process**: Creates matplotlib figures, saves to memory buffer, converts to base64 encoding
- **Output**: Base64-encoded PNG images embedded in JSON responses for web display

---

## Data Flow

### 1. **Data Upload Flow**
```
User uploads CSV → Frontend parses → Backend processes → Redis stores → Model trains
```

### 2. **Price Optimisation Flow**
```
User inputs parameters → Backend calculates → Price/demand curves → Revenue optimisation → Results display
```

### 3. **Visualisation Flow**
```
Backend generates plots → Base64 encoding → JSON response → Frontend decodes → Display
```

---

## API Endpoints

### 1. **Data Processing**
Handles initial dataset upload and validation:
- **Input**: CSV data as JSON from frontend
- **Process**: Validates data format, generates hash for caching, stores in Redis, computes summary statistics
- **Output**: Success/error status and basic dataset statistics

### 2. **User Parameters**
Manages user-specified optimisation parameters:
- **Input**: User form data including external parameters and price range
- **Process**: Validates inputs, applies data type conversion, stores in Redis for later use
- **Output**: Confirmation message indicating successful parameter storage

### 3. **Visualisation Endpoints**
Generate specific plots for dashboard display:

**Demand Plot**: Shows historical demand variation over time
- **Input**: Historical sales data from Redis
- **Output**: Time series plot as base64-encoded image

**Feature Importance Plot**: Displays which factors most influence demand
- **Input**: Trained model and feature names
- **Output**: Horizontal bar chart showing feature importance scores

**Price-Demand Function Plot**: Shows relationship between price and predicted demand
- **Input**: User parameters and price range
- **Output**: Curve plot showing demand elasticity

**Revenue-Price Function Plot**: Identifies optimal price point for revenue maximisation
- **Input**: Calculated price-demand relationship
- **Output**: Revenue curve with peak indicating optimal price

**Model Accuracy Plot**: Validates model performance
- **Input**: Actual vs predicted values from test set
- **Output**: Scatter plot comparing model predictions to reality

### 4. **Model Metrics**
Provides quantitative model performance assessment:
- **Input**: Test dataset and trained model
- **Process**: Calculates Root Mean Square Error (RMSE)
- **Output**: Numeric accuracy score for user evaluation

---

## User Interface Components

### 1. **File Upload Component**
Handles dataset input from users:
- **Input**: CSV files through web interface
- **Process**: Validates file format, triggers data processing pipeline
- **Output**: Column preview and data ready for analysis

### 2. **Dynamic Input Fields**
Automatically adapts to uploaded data structure:
- **Input**: Dataset column structure
- **Process**: Generates appropriate input fields for each parameter
- **Output**: Customised form for user parameter specification

### 3. **Tabbed Interface**
Organises analysis results into logical sections:
- **Tab 1 - Demand Analysis**: Historical patterns and feature importance
- **Tab 2 - Revenue Management**: Price optimisation and revenue curves  
- **Tab 3 - Methodology**: Model accuracy and algorithmic explanation

### 4. **Plot Containers**
Display visualisations with explanatory text:
- **Input**: Base64-encoded images from backend
- **Process**: Decodes images and arranges with contextual explanations
- **Output**: Professional dashboard layout with plots and interpretations

---

## Database and Caching

### Redis Usage

#### 1. **Data Storage**
Persistent session management and data caching:
- **Input**: Pandas DataFrames and user parameters
- **Process**: Serialises data using pickle, stores with unique keys
- **Output**: Cached data accessible across multiple requests

#### 2. **Data Retrieval**
Fast access to previously processed information:
- **Input**: Redis keys for specific datasets
- **Process**: Retrieves serialised data and deserialises back to pandas format
- **Output**: Original DataFrames ready for computation

#### 3. **Cache Management**
Intelligent caching with change detection:
- **Input**: New datasets for comparison
- **Process**: Generates unique hashes, compares with stored versions
- **Output**: Prevents redundant processing, ensures data consistency