# Phishing Website Detection

This project aims to detect phishing websites using machine learning techniques. The project is divided into several components, each responsible for a specific task in the pipeline.

## Components

1. **Data Ingestion**: This component is responsible for reading the dataset and splitting it into training and testing sets.

2. **Data Transformation**: This component preprocesses the data, handling tasks such as missing value imputation and feature scaling.

3. **Model Training**: This component trains various machine learning models on the preprocessed data and selects the best model based on performance.

4. **Prediction Pipeline**: This component uses the trained model to make predictions on new data.

5. **Logging**: This component logs important events and errors during the execution of the pipeline.

6. **Exception Handling**: This component handles exceptions that may occur during the execution of the pipeline.

7. **Application**: This component is a Flask application that uses the prediction pipeline to predict whether a given website is a phishing website.

## Usage

To use this project, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Run the training pipeline to train the model.
4. Start the Flask application.

## Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- Flask
- XGBoost
- 
**Deployment**
model is deploted in AWS used Elastic Beanstalk &  CodePipeline<br>
url: http://phishsingdomain-env.eba-bnpny4zw.ap-south-1.elasticbeanstalk.com

