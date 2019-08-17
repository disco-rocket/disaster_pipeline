import sys
# import libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import pickle
nltk.download(['punkt', 'wordnet'])

def drop_zero_cols(df):
    """
    Input: Dataframe
    Purpose: Drops numeric columns from the input dataframe that sum to zero.
    This is to ensure training doesn't break, and without targets the model would never predict them anyway.
    Output: None
    """
    #Drops columns that don't have any '1' values
    counts = df.sum()
    columns_to_drop = counts[counts == 0].index.tolist()
    if len(columns_to_drop) == 0:
        pass
    else:
        df.drop(columns_to_drop, inplace=True, axis=1)
        print("Columns {0} dropped for having no positives".format(columns_to_drop))

def load_data(database_filepath):
    """
    Input: name of database(database must be stored in this directory.
    Output: X - the message
            y - the dependent variables (one column per category)
            List of column headers for y
    """
    #Example 'sqlite:///disaster_project.db'
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from clean_data', engine)
    #Dropping response columns with no records
    drop_zero_cols(df)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = df.columns[4:]
    return X, y, category_names


def tokenize(text):
    """
    Input: Text to tokenize
    Output: List of individual lemitised words.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for word in tokens:
        clean_word = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(clean_word)

    return clean_tokens


def build_model():
    """
    Input: None
    Output: GridSearch object containing a pipeline that vectorizers, performs TFIDF, and Logistic regression
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize))
        ,('tfidf', TfidfTransformer())
        ,('clf', MultiOutputClassifier(LogisticRegression()))
        ])
    parameters = {
        'clf__estimator__C': [1, 5, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input: 1) Trained model
           2) The X_test population
           3) The Y_Test population
           4) The category names for the T_test columns
    Output: doesn't return anything, but prints the precision, recall, F1 score and accuracy for each category
    """
    f1_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []
    col_name_list = []
    y_pred = model.predict(X_test)
    for i, col_name in enumerate(category_names):
        y_col_test = Y_test.iloc[:,i].values
        y_col_pred = y_pred[:,i]
        precision_list.append(precision_score(y_col_test, y_col_pred))
        recall_list.append(recall_score(y_col_test, y_col_pred))
        f1_list.append(f1_score(y_col_test, y_col_pred))
        accuracy_list.append(accuracy_score(y_col_test, y_col_pred))
        col_name_list.append(col_name)

        column_report = pd.DataFrame({'column':col_name_list
                                      , 'precision':precision_list
                                      ,'recall':recall_list
                                      ,'f1':f1_list
                                      ,'accuracy':accuracy_list}
                                    )
    print(column_report)

def save_model(model, model_filepath):
    """
    Input: 1) trained model to save
           2) the filepath to save the pickle file (e.g. message_model.pickle)
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main function to call the other functions in the correct order
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 ) 
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
