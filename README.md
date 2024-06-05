# Final-year-master-project

This repository contains the code for the dashboard created during my final year project for my Design Engineering degree.

## Code Structure

The code consists of two files:
- **`app.py`**: Contains the front-end code for the dashboard.
- **`backend.py`**: Contains the calculations and processes used by the dashboard to generate the plots available in the front-end.

## Main Libraries Used

- The front-end is built using the **Shiny** library in Python.
- The back-end is managed using a **Redis** server.
- Communication between the front-end and back-end is handled by the **Flask** library.

## Preliminary Setup

### Download Redis

Using Homebrew, you can install Redis by running the following command in your terminal:
```bash
brew install redis
```

Once the installation is complete, start the Redis server using the command:
```bash
brew services start redis
```

To stop the Redis server, use:
```bash
brew services stop redis
```

### Other libraries

There are other libraries used in the code. Make sure you have them all downloaded and up to date.

### Install Redis for Python

You can install Redis for Python using pip:
```bash
pip install redis
```

## Launching the dashboard

### Launch Redis Server

In a separate terminal, run the following command to launch the Redis server for the web app:
```bash
redis-server
```

### Run the Backend

Once Redis is running, execute the backend.py file. Ensure that all the required libraries are installed.

### Run the Frontend

Finally, run the app.py using Shiny. In a separate terminal, enter:
```bash
shiny start
```

Then app.py should be able to run as planned.
