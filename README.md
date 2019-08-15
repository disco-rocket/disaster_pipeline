# disaster_pipeline
# Purpose
This was for a Udacity Data Science Nano-Degree program

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Method
Logistic regression, Support Vector Machine Classification and Randomforest were tried
The untuned SVM took 20 minutes to train and focused on accuracy by setting the prediction for most columns to 0.
Due to the amount of time it took to run, this was dropped.
Logistic regression took less than a minute to train, and gave reasonable precision, recall and accuracy for some columns.
However there were 7 message types with zero presicion or recall
Randomforest took around a minute to train, but had 8 message types with zero recall and precision, as well as poorer results for the majority of message types


