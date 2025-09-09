# Purpose of the data Transformation >>>>>  Feature Engineering , Data Cleaning , Encoding, do Exception handling here

import sys 
from dataclasses import dataclass

import numpy as np 
import pandas as pd 
import os

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifects', "preprocessor.pkl")


class DataTransformation:
    def __init(self) :
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        
        '''
        This Function is responsible for Data Transformation 
        '''

        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            ## We create a pipeline  >>>> This pipeline will run on the training dataset.

            num_pipeline = Pipeline(
                steps=[

                    # This is for handling the missing values.
                    ("imputer", SimpleImputer(strategy="median")),  
                     # Doing the standard scalling .
                    ("scaler", StandardScaler())  
                ]
            )

            cat_pipeline = Pipeline(

                steps = [
                    # 1'st step is for to handle the missing values .
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    # This is for the encoding [ categorical features >>> numerical features].
                    ("one-hot-encoder", OneHotEncoder()),
                    # This one is for Standard Scalling .
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Categorical Columns : {categorical_columns}")
            logging.inof(f"Numerical Columns : {numerical_columns}")

            # Combining the numerical pipeline to this categorical pipeline 
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)   # Raising Custom Exception 
        
    # Lets Starts our Data Transformation technique 

    def initialte_data_transformation(self, train_path, test_path):
        """
        I am starting the Data Transformation inside this function.
        """

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data completed. ")

            logging.info("Obtaining Preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Appling the preprocessor object on the training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved Preprocessing object")

            # save_object is just used for saving the pickle file.
            save_object(   

                file_path = self.data_transformation_config.preprocessor_obj_file_path ,
                obj = preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
