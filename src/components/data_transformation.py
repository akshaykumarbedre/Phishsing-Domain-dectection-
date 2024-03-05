import os
import sys
import numpy as np

from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import pandas as pd
from dataclasses import dataclass
from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_object(self):
         
         try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled

         

            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            preprocessor=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent', missing_values=-1)),
                ('scaler',StandardScaler())

                ]

            )
      
            return preprocessor

            logging.info('Pipeline Completed')

         except Exception as e:
            
            logging.info("Error in  Pipeline")
            raise CustomException(e,sys)



    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            # logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            # logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()
            
            numerical_cols = ["qty_dot_url","qty_hyphen_url","qty_underline_url","qty_slash_url","qty_questionmark_url","qty_equal_url","qty_at_url","qty_and_url","qty_exclamation_url","qty_space_url","qty_tilde_url","qty_comma_url","qty_plus_url","qty_asterisk_url","qty_hashtag_url","qty_dollar_url","qty_percent_url","qty_tld_url","length_url","qty_dot_domain","qty_hyphen_domain","qty_underline_domain","qty_slash_domain","qty_questionmark_domain","qty_equal_domain","qty_at_domain","qty_and_domain","qty_exclamation_domain","qty_space_domain","qty_tilde_domain","qty_comma_domain","qty_plus_domain","qty_asterisk_domain","qty_hashtag_domain","qty_dollar_domain","qty_percent_domain","qty_vowels_domain"]


            
            target_column_name = 'phishing'
            
            ## features into independent and dependent features

            input_feature_train_df = train_df[numerical_cols]
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df[numerical_cols]
            target_feature_test_df=test_df[target_column_name]

            ## apply the transformation

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info('Processsor pickle in created and saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)


