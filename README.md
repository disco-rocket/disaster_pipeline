# Disaster Response Pipeline Project
### Summary
The purpose of this project is to demonstrate knowledge of Pipelines for a Udacity course.
It takes two CSV files containing data on messages received during disasters, merges and cleans the data, then stores this in an SQLite database.
It then takes the messages and performs case normalisation, lemmatisation, and tokenisation the text. TF-IDF is performed on the resulting tokens.
This is then fed through a logistic regression model to categorise each message based on the text.
This is then presented in a Flask Webapp with two pages - one displays the message types and categories in the SQLite database, and the other categorises messages input by the user.

### File structure

├── data                    
│   ├── disaster_categories.csv  # CSV containing the categories for disaster messages
│   ├── disaster_messages.csv    # CSV containing the actual disaster messages
│   ├── disaster_project.db      # Database containing the SQLite database clean_data, which is merged CSV files cleaned
│   └── process_data.py          # Python code to merge, clean and output the csv files into the database
├── disasterapp
│   ├── templates    
│   │   ├──go.html               # HTML for the output of the categorisation of the users input message
│   │   └──master.html           # HTML for main splash page for Webapp
│   ├── __init__.py              # Required, standard init file for Flask
│   └── run.py                   # Code to launch the webapp. Plotly graphs are also set up in this code.
├── models
│   ├── message_model.pickle     # Pickly of the resulting model from train_classifier.py
│   └── train_classifier,py      # Code to transform and train the data in the database
├── disasterapp.py               # Required code for Flask
└── README.md                    # README


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_project.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_project.db models/message_model.pickle`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Model selection
Logistic regression, Support Vector Machine Classification and Randomforest were tried
The untuned SVM took 20 minutes to train and focused on accuracy by setting the prediction for most columns to 0.
Due to the amount of time it took to run, this was dropped.
Logistic regression took less than a minute to train, and gave reasonable precision, recall and accuracy for some columns.
However there were 7 message types with zero presicion or recall
Randomforest took around a minute to train, but had 8 message types with zero recall and precision, as well as poorer results for the majority of message types
