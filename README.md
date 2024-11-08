# Disaster Response Pipeline Project

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to http://0.0.0.0:3001/

### Summary:

This project is part of the Data Science Nanodegree Program by Udacity, in collaboration with Figure Eight. The goal is to build a Natural Language Processing (NLP) model to categorize disaster-related messages in real time.

The dataset consists of real messages sent during disaster events. In this project, I've developed a machine learning pipeline that categorizes these messages to help route them to the appropriate disaster relief agencies.

Key Sections:

- ETL Pipeline: Extracts data, cleans it, and stores it in a SQLite database.
- Machine Learning Pipeline: Trains a classifier to categorize messages into various disaster categories.
- Web Application: Displays real-time classification results using a Flask web app.

### File Description:

## app

app/templates/ : templates/html files for web app.

app/run.py: This file can be used to launch the Flask web app used to classify disaster messages.

## data

data/process_data.py: Extract Transform Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database.

data/ETL Pipeline Preparation.ipynb: Notebook contains ETL Pipeline.

data/DisasterResponse.db: DisasterResponse database.

## models

models/ML Pipeline Preparation.ipynb: Notebook contains ML Pipeline.

models/train_classifier.py: Machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file.

models/classifier.pkl: Trained model.

![image](https://github.com/user-attachments/assets/8c5bfd71-ade1-4f85-a237-b55632f6b3d2)

![image](https://github.com/user-attachments/assets/fefd6f2e-a914-4e0d-97b4-a215c8acd1c1)

![image](https://github.com/user-attachments/assets/9fe9a0e8-d7b6-48df-9405-7315cb48628b)

![image](https://github.com/user-attachments/assets/a79c524f-da5a-4c0e-bf75-6493cc20cefe)


