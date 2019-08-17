import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Input:  Takes the fle paths of the two CSV files. The message filepath must come first, followed by the categories one.
    Output: The merged CSV files
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how='left',left_on='id',right_on='id')

def clean_data(df):
    """
    Purpose: This takes the categories column and creates binary columns for each category.
             It also removes duplicates
    Input: the merged message and category dataframe
    Output: the cleaned dataframe
    """
    #splitting out the category values into different columns
    categories = df['categories'].str.split(';', expand=True)
    #getting category headers
    row = categories.iloc[0,:]
    category_colnames = row.str.split('-', expand=True)[0].tolist()
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]   
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        #restrict to just 0 or 1
        categories[column] = (categories[column] != 0).astype(int)		

    #appending new category columns back onto origional dataframe
    df.drop(['categories'],inplace=True,axis=1)
    df = pd.concat([df,categories],axis=1)

    # check number of duplicates
    counts = pd.DataFrame(df['message'].value_counts()).reset_index(level=0, inplace=False)
    duplicates = counts[counts['message'] > 1].shape[0]
    print('starting with {0} duplicate messages'.format(duplicates))
    # drop duplicates
    df.drop_duplicates(subset ="message", keep = 'first', inplace = True) 
    # check number of duplicates
    counts = pd.DataFrame(df['message'].value_counts()).reset_index(level=0, inplace=False)
    duplicates = counts[counts['message'] > 1].shape[0]
    print('reduced to {0} duplicate messages'.format(duplicates))
    
    return df


def save_data(df, database_filename):
    """
    Purpose: This saves a dataframe in the specified database. Nothing is returned.
    Input: 1) The dataframe you want to save
           2) The name of the database to save it into
    """
    #example database_filename 'disaster_project.db'
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('clean_data', engine, index=False, if_exists='replace')  


def main():
    """
    Main function to call other functions.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
