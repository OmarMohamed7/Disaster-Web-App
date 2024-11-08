import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    Load data
    Loads data from csv to save it into database

    Args:
        messages_filepath (str): Path of messages csv file
        categories_filepath (str): Path of categories csv file

    Returns:
        df (Series): Data frame of two csvs
       
    """
    
    # Step1 : Load csv files
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    
    # Step 2: Merge two databases
    df = pd.merge(messages_df, categories_df, on='id')
    return df


def clean_data(df: pd.DataFrame):
    """
    Clean data

    split word by semicolons 
    extract category names
    
    Args:
        df (DataFrame): data framed to be cleaned

    Returns:
        df (Series): Cleaned data frame
       
    """
    # Step 1: Convert row of categories into columns
    categories_df = df['categories'].str.split(';', expand=True)
    
    # Step 2: Extract the category names (from the first row)
    categories_names = categories_df.iloc[0].apply(lambda x: x.split('-')[0]).tolist()
    
    # Step 3: Rename columns with category names
    categories_df.columns = categories_names
    
    # Step 4: Convert all values to integers (e.g., 'related-1' becomes 1)
    categories_df = categories_df.applymap(lambda x: int(x.split('-')[-1]))
    
    # Step 5: Drop the original 'categories' column
    df.drop(columns=['categories'], inplace=True)
    
    # Step 6: Merge the original dataframe with the transformed category columns
    df = pd.concat([df, categories_df], axis=1)
    
    # Step 7: Remove duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
        Save Data
        Saves a data into database

        Args:
        df: data to be saved
        database_filename: Path to save the data
       
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')
    return df  


def main():
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