import sys
import joblib
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sqlalchemy import create_engine
import re

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV


import matplotlib.pyplot as plt 



def load_data(database_filepath):
    """
    Load data
    Loads data from database to apply transformations

    Args:
        database_filepath (str): Path of database file

    Returns:
        X (Series): Data frame of input
        y (Series): Data frame of target
        categories (list): Categories names
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    original_df = pd.read_sql('SELECT * FROM messages', engine)

    df = original_df.copy()

    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    df.drop('child_alone', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    # df.drop('id',axis=1, inplace=True)
    
    X = df['message']
    y = df.iloc[:, 4:]
    
    df.hist(bins=16)
    
    plt.title('Histogram for Length Column') 
    plt.xlabel('Length') 
    plt.ylabel('Frequency') 
  
    # Display the histogram 
    plt.show() 
    

    return X,y,list(df.columns)


def tokenize(text):
    """
    Tokenize text
    Remove stop words frommm text and apply lemmatization (return the word into their original word (running => run))

    Args:
        text (list): list of phrases

    Returns:
        list: tokenized list
    """
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    words = [w.lower() for w in words if w not in stopwords.words("english")]
    
    # Lemmatizing
    words = [WordNetLemmatizer().lemmatize(w.lower()) for w in words]
    return words


def build_model():
    """
    Build moddel return the fine tuned model

    Returns:
        fine tunedd model
    """
    pipeline = Pipeline([
            # ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfVectorizer(tokenizer=tokenize,use_idf=True, smooth_idf=True, sublinear_tf=False)),
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42,n_jobs=-1), n_jobs=-1))
        ]
    )
        
    parameters = {
        # 'clf__criterion': ['gini', 'entropy'],
        # 'clf__max_features': ['auto', 'sqrt'],
        'clf__estimator__n_estimators': [40, 50, 60],
        # 'clf__random_state': [42]
        }
    
    grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring='f1_weighted', verbose=10)

    return grid_search


def get_metrics(test_value, predicted_value):
    """
    get_metrics calculates f1 score, accuracy and recall

    Args:
        test_value (list): list of actual values
        predicted_value (list): list of predicted values

    Returns:
        dictionray: a dictionary with accuracy, f1 score, precision and recall
    """
    accuracy = accuracy_score(test_value, predicted_value)
    
    precision = round(precision_score(test_value, predicted_value))
    
    recall = recall_score(test_value, predicted_value)
    
    f1_scoree = f1_score(test_value, predicted_value)
    return {'Accuracy': accuracy, 'f1 score': f1_scoree, 'Precision': precision, 'Recall': recall}



def evaluate_model(model, X_test, Y_test, category_names):
    """
        evaluate_model 

        Args:
        model: Model to evaluate
        X_test: liist test data
        Y_test: list of actual values
        category_names: list of categories
    """
    
    y_pred = model.predict(X_test)
    
    test_results = []
    for i, column in enumerate(Y_test.columns):
        result = get_metrics(Y_test.loc[:, column].values, y_pred[:, i])
        test_results.append(result)
    
    test_results_df = pd.DataFrame(test_results, columns=category_names)
    
    print("Result for Each Category")
    print(test_results_df)
    
    print("Overall Evaluation Result")
    print(test_results_df.mean())
    


def save_model(model, model_filepath):
    """
        Save Model
        Saves a model on pkl file 

        Args:
        model: Model to evaluate
        model_filepath: Path to save the model
       
    """
    model = model.best_estimator_
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
