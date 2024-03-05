import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class Training_pipeline:
    def __init__(self) -> None:
        self.DataIngestion_obj=DataIngestion()
        self.data_transformation_obj=DataTransformation()
        self.model_trainer_obj=ModelTrainer()

    def initiate_training_pipeline(self):
        train_data_path,test_data_path=self.DataIngestion_obj.initiate_data_ingestion()
        print(train_data_path,test_data_path)

        train_arr,test_arr,_=self.data_transformation_obj.initiate_data_transformation(train_data_path,test_data_path)
        self.model_trainer_obj.initate_model_training(train_arr,test_arr)



if __name__=='__main__':
    training_pipeline_obj=Training_pipeline()
    training_pipeline_obj.initiate_training_pipeline()



