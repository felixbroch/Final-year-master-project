# Final-year-master-project
This repository is the code to the dashboard during my final year project during my Design Engineering degree. 

The code consists of 2 files: 
app.py which stores the front-end and the code of my dashboard.
backend.py which contains the calculations and process made by the dashboard to output the plots avaiable in the front-end.

The front-end has been coded using the Shiny Library in Python.
The back-end is stored using a Redis server.
The front-end and back-end communicate using the Flask library.


preliminary set up:

Download redis:

Using homebrew this can be done running in your terminal the followig command:
brew install redis

Once this is done, start the redis server using the following command:
brew services start redis

The server can be stopped using"
brew services stop redis

Now download redis for python (pip is a good way to do so):
pip install redis


Now in a separate terminal, run the following command to lauch the redis server for the web-app:
redis-server





Once this is done, run the backend.py file (make sure all the libraries used are installed)

Finally, run the app.py using shiny. This is done by in a separate terminal entering:
shiny start

Then the code can be run easily

