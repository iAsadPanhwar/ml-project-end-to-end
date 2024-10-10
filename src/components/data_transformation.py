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

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
    
class DataTransformation:
    def __int__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def data_transformer():
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
                    ('Scaler', StandardScaler())
                ]
            )
            
            logging.info(f"Numerical Columns {numerical_columns}")
            logging.info(f"Categorical Columns {cat_columns}")
            
            transformer = ColumnTransformer(
                ('Numerice Pipeline', num_pipeline, numerical_columns),
                ('Categorical Pipeline', cat_pipeline, cat_columns)
            )
            
            return transformer
        
        except Exception as e:
            raise CostumException(e, sys)