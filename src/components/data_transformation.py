import os
import sys
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CostumException
from dataclasses import dataclass 
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def data_transformer(self):
        try:
            numerical_columns = ['writing score','reading score']
            cat_columns = ['gender',
                           'race/ethnicity',
                           'parental level of education',
                           'lunch',
                           'test preparation course']
            
            num_pipeline = Pipeline(
                steps = [
                    ('Imputer', SimpleImputer(strategy = 'median')),
                    ('Scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ('Imputer', SimpleImputer(strategy = 'most_frequent')),
                    ('OneHotEncoder', OneHotEncoder()),
                    ('Scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Numerical Columns {numerical_columns}")
            logging.info(f"Categorical Columns {cat_columns}")
            
            transformer = ColumnTransformer(
                [
                ('Numerice Pipeline', num_pipeline, numerical_columns),
                ('Categorical Pipeline', cat_pipeline, cat_columns)
                ]
            )
            
            return transformer
        
        except Exception as e:
            raise CostumException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.data_transformer()

            target_column_name="math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CostumException(e,sys)