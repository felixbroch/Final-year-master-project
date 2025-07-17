# Dynamic Pricing Made Accessible: A Dashboard for Optimal Price Recommendations

**Master's Thesis Project**  
Imperial College London  
Design Engineering Department  
Author: Felix Brochier  
Supervisor: Professor Pierre Pinson  

## Project Overview

This repository contains the implementation of a dynamic pricing dashboard that makes sophisticated pricing optimisation accessible to retailers through a transparent and user-friendly interface. The project addresses the challenge of making advanced pricing strategies available to businesses without requiring extensive technical expertise or resources.

### Background and Motivation

Dynamic pricing has become increasingly important in modern retail, allowing businesses to optimise revenue by adjusting prices based on market conditions, demand patterns, and external factors. However, many retailers lack the technical infrastructure or expertise to implement such systems effectively. This project bridges that gap by providing an accessible, web-based dashboard that delivers optimal price recommendations through a clear, transparent process.

### Objective

The primary objective of this project is to democratise dynamic pricing by creating a dashboard that:
- Provides optimal price recommendations based on historical data and market parameters
- Offers complete transparency in the pricing decision process
- Requires minimal technical knowledge to operate
- Delivers actionable insights through intuitive visualisations
- Supports data-driven decision making for retailers of all sizes

## Methodology

### Technical Approach

The system employs a sophisticated machine learning approach centered around XGBoost (Extreme Gradient Boosting) regression to model demand patterns and optimise pricing strategies. The methodology follows a four-step process:

1. **Demand Function Learning**: The XGBoost model learns the relationship between external parameters (seasonality, market conditions, etc.) and demand, explicitly excluding price to isolate demand drivers.

2. **Feature Analysis**: The system analyses feature importance to identify which external factors most significantly impact demand, providing transparency into the decision-making process.

3. **Price Optimisation**: Using the learned demand function, the system predicts demand across a user-specified price range, creating both demand-to-price and revenue-to-price functions.

4. **Revenue Maximisation**: The optimal price is identified as the point that maximises revenue in the revenue-to-price function.

### Algorithmic Strategy

The system implements two distinct pricing strategies based on historical price variation:

- **Linear Approximation**: For datasets with low price variation (>80% of prices at the same level), the system uses a linear demand function based on economic theory.
- **ML-Based Prediction**: For datasets with high price variation, the system leverages the full predictive power of the XGBoost model to estimate demand at each price point.

## Technologies and Tools

### Frontend
- **Shiny for Python**: Interactive web application framework
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Visualisation and plotting
- **PIL (Python Imaging Library)**: Image processing for plot display

### Backend
- **Flask**: RESTful API framework
- **XGBoost**: Machine learning for demand prediction
- **Scikit-learn**: Model evaluation and metrics
- **Redis**: Data caching and session management
- **NumPy**: Numerical computing

### Development Environment
- **Python 3.x**: Core programming language
- **Git**: Version control
- **Jupyter Notebooks**: Prototyping and analysis (if applicable)

## Repository Structure

```
.
├── README.md           # Project documentation
├── app.py             # Frontend Shiny application
├── backend.py         # Flask API backend
└── .gitignore         # Git ignore file
```

### Key Files

- **`app.py`**: Contains the complete frontend implementation using Shiny for Python, including the user interface, interactive elements, and communication with the backend API.

- **`backend.py`**: Implements the Flask API that handles data processing, machine learning model training, price optimisation calculations, and plot generation.

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- Redis server

### Required Libraries

Install the required Python packages:

```bash
pip install shiny pandas matplotlib flask xgboost scikit-learn redis pillow numpy seaborn
```

### Redis Setup

#### macOS (using Homebrew):
```bash
brew install redis
brew services start redis
```

#### Windows:
Download and install Redis from the official Redis website or use Windows Subsystem for Linux (WSL).

#### Linux:
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

### Running the Application

1. **Start Redis Server**:
   ```bash
   redis-server
   ```

2. **Launch the Backend API**:
   ```bash
   python backend.py
   ```
   The API will be available at `http://localhost:5001`

3. **Start the Frontend Dashboard**:
   ```bash
   shiny run app.py
   ```
   The dashboard will be available at `http://localhost:8000`

## Usage Guide

### Data Requirements

The system expects CSV files with the following columns:
- **Date**: Date of the transaction (DD/MM/YYYY format)
- **Number of sales**: Quantity sold
- **Price of product**: Price at which the product was sold
- **Additional columns**: Any external parameters (seasonality, weather, promotions, etc.)

### Dashboard Navigation

The dashboard features three main analysis tabs:

1. **Demand Analysis**: 
   - Displays historical demand variations over time
   - Shows feature importance analysis to identify key demand drivers
   - Provides insights into seasonal patterns and trends

2. **Revenue Management**:
   - Presents the price-demand relationship curve
   - Displays the revenue-price function
   - Identifies the optimal price point for revenue maximisation

3. **Methodology**:
   - Shows model accuracy through actual vs predicted comparisons
   - Provides RMSE score for model evaluation
   - Explains the algorithmic approach and methodology

### Workflow

1. Upload your historical sales data via the file upload interface
2. Review the column mapping to ensure proper data interpretation
3. Input external parameters for the optimisation scenario
4. Specify the minimum and maximum price range for analysis
5. Click "Submit" to generate optimal price recommendations
6. Navigate through the tabs to explore demand patterns, revenue optimisation, and model accuracy

## Results and Deliverables

### Key Outputs

- **Optimal Price Recommendation**: Data-driven price suggestion that maximises revenue
- **Demand Elasticity Analysis**: Understanding of how price changes affect demand
- **Feature Importance Insights**: Identification of key factors driving demand
- **Model Accuracy Metrics**: Quantitative assessment of prediction reliability
- **Interactive Visualisations**: Clear, interpretable charts and plots

### Business Value

The dashboard provides immediate business value by:
- Enabling data-driven pricing decisions
- Reducing reliance on intuition-based pricing
- Providing transparency in pricing methodology
- Offering scalable pricing optimisation for various product categories
- Supporting strategic business planning with demand insights

## Limitations and Considerations

### Current Limitations

- **Data Quality Dependency**: Results are highly dependent on the quality and completeness of historical data
- **External Factor Modeling**: The system's accuracy depends on the inclusion of relevant external parameters
- **Market Dynamics**: The model assumes historical patterns will continue, which may not hold during market disruptions
- **Price Elasticity Assumptions**: Linear approximation may not capture complex demand relationships

### Future Enhancements

Potential areas for improvement include:
- Integration with real-time market data
- Advanced time series forecasting capabilities
- Multi-product pricing optimisation
- Automated feature engineering
- Enhanced user interface with additional customisation options
- Integration with e-commerce platforms and POS systems
