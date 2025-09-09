# Here we are going to train multiple model and check the accuracy 
import os 
import sys 
from dataclasses import dataclass 
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBoostClassi
from catboost import CatBoostRegressor
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifects", "model.pkl")

class ModelTrainer:     # This class is responsible for the model trainnig. 
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data ")
            X_train, y_train, X_test, y_test = (
                train_array [:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest " : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Adaboost Regressor" : AdaBoostRegressor(),
                "Gradient Boosting " : GradientBoostingRegressor(),
                "linear Regression" : LinearRegression(),
                "K-Neighbours " : KNeighborsRegressor(),
                "XGBClassifier" : XGBoostRegressor(),
                "CatBoost Classifier": CatBoostRegressor(verbose = False)
            }

            model_report : dict = evaluate_models(X_train=X_train, X_test = X_test ,y_train=y_train, y_test = y_test,
                                                  models=models)
            
            # To get best score from dictionary 
            best_model_score = max(sorted( model_report.values()))

            # To get best model name from dictionary 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException ("No best model found")
            logging.info(f"Best found model on both training and testing dataset ")

            # Lets save the model path 
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted) 
            return r2_square
        
        except Exception as e:
            raise CustomException (e, sys)
            
