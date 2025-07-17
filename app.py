"""
Dynamic Pricing Dashboard - Frontend Application

This module contains the frontend code for the dynamic pricing dashboard built with Shiny for Python.
The dashboard provides an interactive interface for retailers to analyse demand patterns and 
receive optimal price recommendations through a transparent, data-driven process.

Author: Felix Brochier
Institution: Imperial College London
Supervisor: Professor Pierre Pinson
Project: "Dynamic Pricing Made Accessible: A Dashboard Which Provides Optimal Price 
         Recommendations Through a Transparent and Seamless Process"
"""

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web deployment
import matplotlib.pyplot as plt
import re
from requests import post
import io
import base64
from PIL import Image
import hashlib
import asyncio

def file_hash(data_frame):
    """
    Generate a hash for a pandas DataFrame for caching and comparison purposes.
    
    Args:
        data_frame (pd.DataFrame): The DataFrame to hash
        
    Returns:
        str: MD5 hash of the DataFrame contents
    """
    return hashlib.md5(pd.util.hash_pandas_object(data_frame, index=True).values).hexdigest()


# UI Layout Definition
# Define the main user interface layout with custom CSS styling for professional appearance
app_ui = ui.page_fluid(
    ui.panel_title("Accessible dynamic pricing for retailers", "Dynamic Pricing"),
    ui.tags.head(
        # JavaScript for tab switching functionality
        ui.tags.script("""
            Shiny.addCustomMessageHandler('switch_tab', function(message) {
                var tabId = message.id;
                var tabElement = document.querySelector(`[data-value="${tabId}"]`);
                if (tabElement) {
                    tabElement.click();
                }
            });
        """),
        # Custom CSS styling for dashboard appearance and responsiveness
        ui.tags.style("""
            body {
                font-family: Arial, sans-serif;
                background-color: #f8f9fa;
                color: #333;
                margin: 0;
                padding: 3%;
            }
            #dynamic-input-container {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                align-items: flex-end;
                margin-top: 20px;
            }
            .small-input {
                flex: 1 1 200px;
                max-width: 200px;
            }
            .small-input input {
                width: 100%;
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            #submit-button {
                margin-top: 20px;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            #submit-button:hover {
                background-color: #0056b3;
            }
            .tab-content {
                padding: 10px;
                background-color: white;
                border: 1px solid #ddd;
                border-top: none;
            }
            .nav-tabs .nav-link {
                border: 1px solid #ddd;
                border-radius: 4px 4px 0 0;
                background-color: #f8f9fa;
                margin-right: 2px;
                padding: 10px;
            }
            .nav-tabs .nav-link.active {
                background-color: white;
                border-bottom: 1px solid white;
            }
            .plot-frame {
                border: 2px solid #ddd; /* Frame border */
                padding: 10px; /* Padding inside the frame */
                margin-bottom: 20px; /* Space below the frame */
                background-color: white; /* Background color for the frame */
            }
            #dynamic-input-container {
            display: flex;
            align-items: flex-end;
            }
            .small-input input {
                height: 40px; /* Adjust this value as needed */
                box-sizing: border-box;
                margin-bottom: 0px;
            }
            .submit-button {
                height: 40px; /* Adjust this value to match the input height */
                box-sizing: border-box;
                margin-bottom: 17px
            }
            .text-middle {
                display: flex;
                align-items: center;
                height: 100%;
                justify-content: center;
                text-align: center;
            }
        """)
    ),
    ui.tags.div(class_="navbar", children="Welcome to Our Professional Dashboard"),
    # Main dashboard description and instructions
    ui.markdown("""
    This project aims to make dynamic pricing accessible and transparent for retailers. In just a few clicks, have an optimal price recommendation, with a transparent, seamless process.
    - **Tab 1:** Interpretation of demand, how it fluctuates and the driving parameters behind it.
    - **Tab 2:** Revenue Management and optimisation.
    - **Tab 3:** Methodology and model accuracy.
    
    Upload your data, define the parameters from your data and find your optimal price!
    """),
    # Dynamic input container for user parameters
    ui.output_ui("dynamic_inputs"),
    # Main layout with sidebar for data upload and main panel for analysis tabs
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file("file1", "Choose CSV File", accept=".csv"),
            ui.output_text("file_upload_explanation"),
            ui.input_action_button("upload_test", "Upload Test Dataset"),
            ui.output_table("column_list"),
            width=2.7
        ),
        ui.panel_main(
            # Tab navigation for different analysis views
            ui.navset_tab(
                # Tab 1: Demand Analysis - Shows demand patterns and feature importance
                ui.nav("Demand Analysis", ui.div(
                    ui.row(
                        ui.column(8,  # Adjust width as necessary
                            ui.output_ui("plot1_ui")
                        ),
                        ui.column(4,
                            ui.div(ui.output_text("text1"), class_="text-middle")
                        )
                    ),
                    ui.row(
                        ui.column(8,  # Adjust width as necessary
                            ui.output_ui("plot1_2_ui")
                        ),
                        ui.column(4,
                            ui.div(ui.output_text("text1_2"), class_="text-middle")
                        )
                    ),
                    class_="tab-content"
                )),
                # Tab 2: Revenue Management - Shows price-demand relationship and revenue optimisation
                ui.nav("Revenue Management", ui.div(
                    ui.row(
                        ui.column(8,  # Adjust width as necessary
                            ui.output_ui("plot2_ui")
                        ),
                        ui.column(4,
                            ui.div(ui.output_text("text2"), class_="text-middle")
                        )
                    ),
                    ui.row(
                        ui.column(8,  # Adjust width as necessary
                            ui.output_ui("plot2_2_ui")
                        ),
                        ui.column(4,
                            ui.div(ui.output_text("text2_2"), class_="text-middle")
                        )
                    ),
                    class_="tab-content"
                )),
                # Tab 3: Methodology - Shows model accuracy and explains the algorithmic approach
                ui.nav("Methodology", ui.div(
                    ui.row(
                        ui.column(8,  # Adjust width as necessary
                            ui.output_ui("plot3_ui")
                        ),
                        ui.column(4,
                            ui.div(ui.output_text("RMSE"), class_="text-middle")
                        )
                    ),
                    ui.output_text("output3"),
                    ui.row(  # Methodology explanation section
                        ui.column(12,  # Full width for methodology description
                            ui.markdown("""
                            The code works as following:
                            - **Step 1:** The Model uses XGBoost (Extreme Gradient Boosting) to learn the demand function only from the external parameters (all parameters apart the price).
                            - **Step 2:** Once it has done so, it outputs the plot in the "Demand Analysis" tab and asks the user to set up external parameters in which he wants to calculate the optimal price, as well as the minimum and maximum price to find the optimal price.
                            - **Step 3:** Once the user clicks on the "Submit button", the demand is predicted using the external demand defined by the user and the different prices in the range given by the user. From these demand, a demand to price function and a revenue to price function is defined.
                            - **Step 4:** From the revenue to price function, the user can find the optimal price.

                            """),
                        ),
                    ),
                    class_="tab-content"
                )),
            )
        )   
    )
)

# Server Logic Definition
# Define the server logic that handles user interactions and data processing
def server(input, output, session):
    """
    Server function that handles all backend logic for the Shiny application.
    
    This function manages:
    - File upload and parsing
    - Communication with the Flask backend
    - Plot generation and display
    - User input handling and validation
    - Tab switching and UI updates
    
    Args:
        input: Shiny input object containing user inputs
        output: Shiny output object for rendering UI elements
        session: Shiny session object for managing user sessions
    """

    @reactive.calc
    def parsed_file():
        """
        Parse uploaded CSV file or load test dataset.
        
        Returns:
            pd.DataFrame: Parsed data ready for analysis
        """
        file_info = input.file1()
        if not file_info:
            return pd.DataFrame()

        if input.upload_test() > 0:
            # Load mock dataset for testing purposes
            df = pd.read_csv('/Users/felix/Desktop/FYP Code copy/Dashboard/Mock_data/test_data.csv')
            print("we are getting the mock dataset")
        else:
            # Load user-uploaded file
            file_path = file_info[0]["datapath"]
            df = pd.read_csv(file_path)
        
        return df

    @reactive.Effect
    def summary():
        """
        Process uploaded data and send to backend for initial analysis.
        
        This function handles the initial data processing by sending the parsed
        data to the Flask backend for summary statistics computation.
        """
        df = parsed_file()

        if df.empty:
            return pd.DataFrame()
        
        # Send data to backend for processing
        response = post('http://localhost:5001/process_data', json=df.to_dict(orient='records'))
        
        if response.status_code == 200:
            summary_stats = response.json()
        else:
            print(f"Error sending data to backend: {response.status_code}, {response.text}")

    @reactive.Effect
    async def handle_submit():
        """
        Handle user parameter submission and trigger price optimisation.
        
        This function collects user inputs, sends them to the backend for
        price optimisation calculations, and automatically switches to the
        Revenue Management tab to display results.
        """
        if input.submit_button() > 0:
            # Collect user input values from dynamic form
            input_values = {col: input[f"input_{re.sub(r'[^a-zA-Z0-9]', '_', col)}"]() for col in session.user_df.columns}
            user_df = pd.DataFrame([input_values])
            
            # Send user parameters to backend
            response = post('http://localhost:5001/user_choice', json=user_df.to_dict(orient='records'))

            if response.status_code == 200:
                summary_stats = response.json()
                # Automatically switch to Revenue Management tab to show results
                await session.send_custom_message("switch_tab", {"id": "Revenue Management"})
            else:
                print(f"Error sending data to backend: {response.status_code}, {response.text}")

    @reactive.Effect
    def handle_upload_test():
        """
        Handle test dataset upload and processing.
        
        This function loads a predefined test dataset for demonstration purposes
        and processes it through the backend pipeline.
        """
        if input.upload_test() > 0:
            print("I am running here")
            # Load and process mock dataset
            mock_dataset = pd.read_csv('/Users/felix/Desktop/FYP Code copy/Dashboard/Mock_data/test_data.csv')
            response = post('http://localhost:5001/process_data', json=mock_dataset.to_dict(orient='records'))
            
            if response.status_code == 200:
                result = response.json()
                print("Backend response:", result)
                summary_stats = result.get('summary_stats', {})
                print("Summary statistics received:", summary_stats)
                session.send_custom_message("switch_tab", {})
            else:
                print(f"Error sending data to backend: {response.status_code}, {response.text}")



    @output
    @render.table
    def column_list():
        """
        Display the column names of the uploaded CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing column names for user reference
        """
        file_info = input.file1()
        if not file_info:
            return pd.DataFrame()

        file_path = file_info[0]["datapath"]
        try:
            df = pd.read_csv(file_path)
            return pd.DataFrame(df.columns, columns=["Columns in the uploaded file"])
        except Exception as e:
            print(f"Error reading the file: {e}")
            return pd.DataFrame()

    @output
    @render.ui
    def demand_plot_ui():
        """
        Create UI container for demand plot and associated text.
        
        Returns:
            ui.row: Row containing plot output and text explanation
        """
        plot_output = ui.output_plot("demand_plot")
        text_output = ui.output_text("demand_plot_text")
        return ui.row(plot_output, text_output)

    @output
    @render.plot
    def demand_plot():
        """
        Generate demand variation plot from backend.
        
        This function communicates with the Flask backend to generate
        a plot showing demand variations over time.
        
        Returns:
            matplotlib.figure.Figure: Plot figure or None if error
        """
        df = parsed_file()
        if df.empty:
            return None

        try:
            # Request demand plot from backend
            response = post('http://localhost:5001/demand_plot', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                # Decode base64 image and display
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()
                return fig
            else:
                return None
        except Exception as e:
            print("Error:", e)
            return None

    @output
    @render.text
    def text1():
        """
        Provide explanatory text for the demand analysis plot.
        
        Returns:
            str: Explanation of the demand variation plot
        """
        df = parsed_file()
        if df.empty == False or input.upload_test() > 0:
            return "This plot shows the demand variations. The number of sales and the demand associated to the product changes over time, why not change your prices to this demand?"

    @output
    @render.ui
    def plot1_ui():
        plot_output = ui.output_plot("plot1")
        text_output = ui.output_text("plot1_text")
        return ui.row(plot_output, text_output)

    @output
    @render.plot
    def plot1():
        df = parsed_file()
        if df.empty == True and input.upload_test() > 0:
            response = post('http://localhost:5001/demand_plot', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))  # Adjust figure size as needed
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()  # Ensure tight layout
                return fig
            else:
                return None
        elif df.empty == False:
            response = post('http://localhost:5001/demand_plot', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))  # Adjust figure size as needed
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()  # Ensure tight layout
                return fig
            else:
                return None

    @output
    @render.ui
    def plot1_2_ui():
        plot_output = ui.output_plot("plot1_2")
        text_output = ui.output_text("plot1_2_text")
        return ui.row(plot_output, text_output)



    @output
    @render.plot
    def plot1_2():
        df = parsed_file()
        if df.empty == True and input.upload_test() > 0:
            response = post('http://localhost:5001/feature_importance_plot', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))  # Adjust figure size as needed
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()  # Ensure tight layout
                return fig
            else:
                return None
        elif df.empty == False:
            response = post('http://localhost:5001/feature_importance_plot', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))  # Adjust figure size as needed
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()  # Ensure tight layout
                return fig
            else:
                return None

    @output
    @render.text
    def text1_2():
        """
        Provide explanatory text for the feature importance plot.
        
        Returns:
            str: Explanation of the feature importance analysis
        """
        df = parsed_file()
        if df.empty == False or input.upload_test() > 0:
            return "This plot shows the feature importance in the demand analysis. From the parameters you have input, the aim of this graph is to show what parameters affect the most the changes over demand. The higher the 'F score', the stronger the influence."

    @output
    @render.ui
    def plot2_ui():
        """
        Create UI container for price-demand plot and associated text.
        
        Returns:
            ui.row: Row containing plot output and text explanation
        """
        plot_output = ui.output_plot("plot2")
        text_output = ui.output_text("plot2_text")
        return ui.row(plot_output, text_output)

    @output
    @render.plot
    def plot2():
        """
        Generate price-demand relationship plot from backend.
        
        Returns:
            matplotlib.figure.Figure: Plot figure or None if error
        """
        df = parsed_file()
        if df.empty == True and input.upload_test() > 0:
            response = post('http://localhost:5001/price_demand_function_plot', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()
                return fig
            else:
                return None
        elif df.empty == False:
            response = post('http://localhost:5001/price_demand_function_plot', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()
                return fig
            else:
                return None

    @output
    @render.text
    def text2():
        """
        Provide explanatory text for the price-demand relationship plot.
        
        Returns:
            str: Explanation of the price elasticity visualisation
        """
        df = parsed_file()
        if df.empty == False or input.upload_test() > 0:
            return "This plot shows the price to demand function. This plot represents the price elasticity of the product: if the price increases by 1, how much people are going to buy this? The curve should be decreasing, as the more expensive a product is, the less buyer there will be."

    @output
    @render.ui
    def plot2_2_ui():
        """
        Create UI container for revenue-price plot and associated text.
        
        Returns:
            ui.row: Row containing plot output and text explanation
        """
        plot_output = ui.output_plot("plot2_2")
        text_output = ui.output_text("plot2_2_text")
        return ui.row(plot_output, text_output)

    @output
    @render.plot
    def plot2_2():
        """
        Generate revenue-price relationship plot from backend.
        
        Returns:
            matplotlib.figure.Figure: Plot figure or None if error
        """
        df = parsed_file()
        if df.empty == True and input.upload_test() > 0:
            response = post('http://localhost:5001/revenue_demand_function_plot', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()
                return fig
            else:
                return None
        elif df.empty == False:
            response = post('http://localhost:5001/revenue_demand_function_plot', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()
                return fig
            else:
                return None

    @output
    @render.ui
    def plot3_ui():
        """
        Create UI container for model accuracy plot and associated text.
        
        Returns:
            ui.row: Row containing plot output and text explanation
        """
        plot_output = ui.output_plot("plot3")
        text_output = ui.output_text("plot3_text")
        return ui.row(plot_output, text_output)

    @output
    @render.plot
    def plot3():
        """
        Generate model accuracy plot from backend.
        
        Returns:
            matplotlib.figure.Figure: Plot figure or None if error
        """
        df = parsed_file()
        if df.empty == True and input.upload_test() > 0:
            response = post('http://localhost:5001/model_accuracy', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()
                return fig
            else:
                return None
        elif df.empty == False:
            response = post('http://localhost:5001/model_accuracy', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result['plot'])
                image = Image.open(io.BytesIO(image_data))
                fig, ax = plt.subplots(figsize=(20, 6))
                ax.imshow(image)
                ax.axis('off')
                plt.tight_layout()
                return fig
            else:
                return None

    @output
    @render.text
    def text2_2():
        """
        Provide explanatory text for the revenue-price relationship plot.
        
        Returns:
            str: Explanation of the revenue optimisation visualisation
        """
        df = parsed_file()
        if df.empty == False or input.upload_test() > 0:
            return "This plot shows the revenue-demand function. From the price to demand function, the demand is multiplied to its associated price to calculate the revenue at each price. The highest point in this plot determines the biggest revenue possible, the optimal price is the one associated to the biggest revenue. "

    @output
    @render.text
    def RMSE():
        """
        Calculate and display the Root Mean Square Error for model evaluation.
        
        This function communicates with the backend to get the RMSE score
        and formats it for display with explanatory text.
        
        Returns:
            str: RMSE score with explanation
        """
        df = parsed_file()
        if df.empty == False or input.upload_test() > 0:
            response = post('http://localhost:5001/RMSE', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                rmse_value = round(result["RMSE"], 2)
            return "This plot shows the model accuracy. This aims to show from the data you are using, how accurate is the recommendation from the model. Using Root Mean Squared Error (RMSE), the score is: " + str(rmse_value) + '.'

    @output
    @render.text
    def file_upload_explanation():
        """
        Provide instructions for file upload.
        
        Returns:
            str: User guidance for file upload
        """
        return "Use button above to upload your data and get started."

    @output
    @render.ui
    def dynamic_inputs():
        """
        Generate dynamic input fields based on uploaded data columns.
        
        This function creates input fields for each column in the uploaded dataset,
        allowing users to specify values for external parameters and price ranges.
        
        Returns:
            ui.div: Container with dynamic input fields and submit button
        """
        df = parsed_file()
        
        # Handle test dataset case
        if df.empty == True and input.upload_test() > 0:
            session.user_df = pd.read_csv('/Users/felix/Desktop/FYP Code copy/Dashboard/Mock_data/test_data.csv')
            inputs = [
                ui.div(
                    ui.input_text(
                        f"input_{re.sub(r'[^a-zA-Z0-9]', '_', col)}",
                        f"Enter: {col}" if i < len(session.user_df.columns) - 2 else "Minimum Price" if i == len(session.user_df.columns) - 2 else "Maximum Price"
                    ),
                    class_="small-input"
                )
                for i, col in enumerate(session.user_df.columns)
            ]
            submit_button = ui.input_action_button("submit_button", "Submit All", class_="submit-button")
            return ui.div(*inputs, submit_button, id="dynamic-input-container")

        # Handle uploaded file case
        elif df.empty == False:
            file_info = input.file1()
            file_path = file_info[0]["datapath"]
            session.user_df = pd.read_csv(file_path)
            inputs = [
                ui.div(
                    ui.input_text(
                        f"input_{re.sub(r'[^a-zA-Z0-9]', '_', col)}",
                        f"Enter: {col}" if i < len(session.user_df.columns) - 2 else "Minimum Price" if i == len(session.user_df.columns) - 2 else "Maximum Price"
                    ),
                    class_="small-input"
                )
                for i, col in enumerate(session.user_df.columns)
            ]
            submit_button = ui.input_action_button("submit_button", "Submit All", class_="submit-button")
            return ui.div(*inputs, submit_button, id="dynamic-input-container")

        # Return empty div if no data
        else:
            return ui.div()

# Create and run the Shiny application
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
