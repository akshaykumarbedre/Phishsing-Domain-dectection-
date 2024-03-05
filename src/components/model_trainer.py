import os
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from src.utils import evaluate_model , save_object
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self) -> None:
        self.model_train_config=ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'RandomForestClassifier':RandomForestClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'XGBClassifier':XGBClassifier(),
            'SVC':SVC(),
            'AdaBoostClassifier':AdaBoostClassifier(),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            # 'KNeighborsClassifier':KNeighborsClassifier(),
            'XGBClassifier':XGBClassifier()
            }
            logging.info("Training the model started , place wait untill all model get train , And choose the best one ")
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy: {best_model_score}')
            logging.info(f'Model Saving : {best_model_name}')

            save_object(
                    file_path=self.model_train_config.trained_model_file_path,
                    obj=best_model
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)