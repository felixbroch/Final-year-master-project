from shiny import App, Inputs, Outputs, Session, reactive, render, ui
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from requests import post
import io
import base64
from PIL import Image
import hashlib
import asyncio

def file_hash(data_frame):
    return hashlib.md5(pd.util.hash_pandas_object(data_frame, index=True).values).hexdigest()



# Define the UI layout with additional CSS for styling input boxes and aligning them
# Define the UI layout with additional CSS for styling input boxes and aligning them
app_ui = ui.page_fluid(
    ui.panel_title("Accessible dynamic pricing for retailers", "Dynamic Pricing"),
    ui.tags.head(
        ui.tags.script("""
            Shiny.addCustomMessageHandler('switch_tab', function(message) {
                var tabId = message.id;
                var tabElement = document.querySelector(`[data-value="${tabId}"]`);
                if (tabElement) {
                    tabElement.click();
                }
            });
        """),
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
    ui.markdown("""
    This project aims to make dynamic pricing accessible and transparent for retailers. In just a few clicks, have an optimal price recommendation, with a transparent, seamless process.
    - **Tab 1:** Interpretation of demand, how it fluctuates and the driving parameters behind it.
    - **Tab 2:** Revenue Management and optimisation.
    - **Tab 3:** Methodology and model accuracy.
    
    Upload your data, define the parameters from your data and find your optimal price!
    """),
    ui.output_ui("dynamic_inputs"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file("file1", "Choose CSV File", accept=".csv"),
            ui.output_text("file_upload_explanation"),
            ui.input_action_button("upload_test", "Upload Test Dataset"),
            ui.output_table("column_list"),
            width=2.7
        ),
        ui.panel_main(
            ui.navset_tab(
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
                    ui.row(  # This is the new row added
                        ui.column(12,  # This can be adjusted as necessary
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

# Define the server logic
def server(input, output, session):

    @reactive.calc
    def parsed_file():
        file_info = input.file1()
        if not file_info:
            return pd.DataFrame()

        if input.upload_test() > 0:
            df = pd.read_csv('/Users/felix/Desktop/FYP Code copy/Dashboard/Mock_data/test_data.csv')
            print("we are getting the mock dataset")
        else:
            file_path = file_info[0]["datapath"]
            df = pd.read_csv(file_path)
        
        return df

    @reactive.Effect
    def summary():

        df = parsed_file()

        if df.empty:
            return pd.DataFrame()
        response = post('http://localhost:5001/process_data', json=df.to_dict(orient='records'))
        
        if response.status_code == 200:
            summary_stats = response.json()
        else:
            print(f"Error sending data to backend: {response.status_code}, {response.text}")

    @reactive.Effect
    async def handle_submit():
        if input.submit_button() > 0:
            input_values = {col: input[f"input_{re.sub(r'[^a-zA-Z0-9]', '_', col)}"]() for col in session.user_df.columns}
            user_df = pd.DataFrame([input_values])
            response = post('http://localhost:5001/user_choice', json=user_df.to_dict(orient='records'))

            if response.status_code == 200:
                summary_stats = response.json()
                # Send a custom message to switch to the "Revenue Management" tab
                await session.send_custom_message("switch_tab", {"id": "Revenue Management"})
            else:
                print(f"Error sending data to backend: {response.status_code}, {response.text}")


    @reactive.Effect
    def handle_upload_test():
        if input.upload_test() > 0:
            print("I am running here")
            mock_dataset = pd.read_csv('/Users/felix/Desktop/FYP Code copy/Dashboard/Mock_data/test_data.csv')
            response = post('http://localhost:5001/process_data', json=mock_dataset.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                print("Backend response:", result)
                summary_stats = result.get('summary_stats', {})
                print("Summary statistics received:", summary_stats)
                # Perform any necessary updates with summary_stats here
                session.send_custom_message("switch_tab", {})
            else:
                print(f"Error sending data to backend: {response.status_code}, {response.text}")



    @output
    @render.table
    def column_list():
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
        plot_output = ui.output_plot("demand_plot")
        text_output = ui.output_text("demand_plot_text")
        return ui.row(plot_output, text_output)

    @output
    @render.plot
    def demand_plot():
        df = parsed_file()
        if df.empty:
            return None

        try:
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
        except Exception as e:
            print("Error:", e)
            return None

    @output
    @render.text
    def text1():
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
        df = parsed_file()
        if df.empty == False or input.upload_test() > 0:
            return "This plot shows the feature importance in the demand analysis. From the parameters you have input, the aim of this graph is to show what parameters affect the most the changes over demand. The higher the 'F score', the stronger the influence."

    @output
    @render.ui
    def plot2_ui():
        plot_output = ui.output_plot("plot2")
        text_output = ui.output_text("plot2_text")
        return ui.row(plot_output, text_output)

    @output
    @render.plot
    def plot2():
        df = parsed_file()
        if df.empty == True and input.upload_test() > 0:
            response = post('http://localhost:5001/price_demand_function_plot', json=df.to_dict(orient='records'))
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
            response = post('http://localhost:5001/price_demand_function_plot', json=df.to_dict(orient='records'))
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
    def text2():
        df = parsed_file()
        if df.empty == False or input.upload_test() > 0:
            return "This plot shows the price to demand function. This plot represents the price elasticity of the product: if the price increases by 1, how much people are going to buy this? The curve should be decreasing, as the more expensive a product is, the less buyer there will be."

    @output
    @render.ui
    def plot2_2_ui():
        plot_output = ui.output_plot("plot2_2")
        text_output = ui.output_text("plot2_2_text")
        return ui.row(plot_output, text_output)

    @output
    @render.plot
    def plot2_2():
        df = parsed_file()
        if df.empty == True and input.upload_test() > 0:
            response = post('http://localhost:5001/revenue_demand_function_plot', json=df.to_dict(orient='records'))
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
            response = post('http://localhost:5001/revenue_demand_function_plot', json=df.to_dict(orient='records'))
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
    def text2_2():
        df = parsed_file()
        if df.empty == False or input.upload_test() > 0:
            return "This plot shows the revenue-demand function. From the price to demand function, the demand is multiplied to its associated price to calculate the revenue at each price. The highest point in this plot determines the biggest revenue possible, the optimal price is the one associated to the biggest revenue. "

    @output
    @render.ui
    def plot3_ui():
        plot_output = ui.output_plot("plot3")
        text_output = ui.output_text("plot3_text")
        return ui.row(plot_output, text_output)

    @output
    @render.plot
    def plot3():
        df = parsed_file()
        if df.empty == True and input.upload_test() > 0:
            response = post('http://localhost:5001/model_accuracy', json=df.to_dict(orient='records'))
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
            response = post('http://localhost:5001/model_accuracy', json=df.to_dict(orient='records'))
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
    def RMSE():
        df = parsed_file()
        if df.empty == False or input.upload_test() > 0:
            response = post('http://localhost:5001/RMSE', json=df.to_dict(orient='records'))
            if response.status_code == 200:
                result = response.json()
                rmse_value = round(result["RMSE"],2)
            return "This plot shows the model accuracy. This aims to show from the data you are using, how accurate is the recommendation from the model. Using Root Mean Squared Error (RMSE), the score is: " + str(rmse_value) + '.'

    @output
    @render.text
    def file_upload_explanation():
        return "Use button above to upload your data and get started."


    @output
    @render.text
    def Explanation_text():
        return """The code works as following:

        - The Model uses XGBoost (Extreme Gradient Boosting) to learn the demand function only from the external parameters (all parameters apart the price).
        - Once it has done so, it outputs the plot in the "Demand Analysis" tab and asks the user to set up external parameters in which he wants to calculate the optimal price, as well as the minimum and maximum price to find the optimal price.
        - Once the user clicks on the "Submit button", the demand is predicted using the external demand defined by the user and the different prices in the range given by the user. From these demand, a demand to price function and a revenue to price function is defined.
        - From the revenue to price function, the user can find the optimal price.

        """





    @output
    @render.ui
    def dynamic_inputs():
        file_info = input.file1()
        if not file_info:
            return ui.div()
        
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

    
    @output
    @render.ui
    def dynamic_inputs():
        df = parsed_file()
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







# Create the Shiny app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
