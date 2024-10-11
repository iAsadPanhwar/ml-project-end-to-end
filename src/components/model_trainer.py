import os
import sys
from src.logger import logging
from src.exception import CostumException  # Corrected the typo
from dataclasses import dataclass 
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_fie_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            ) 
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            if not model_report:  # Check if model_report is empty or None
                raise ValueError("model_report is either None or empty.")
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CostumException("No best model found")
            
            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_fie_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)
            return score

        except Exception as e:
            raise CostumException(e, sys)
