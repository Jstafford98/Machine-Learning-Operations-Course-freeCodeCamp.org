import logging
import pandas as pd
from zenml import step

from sklearn.base import RegressorMixin
from .config import ModelNameConfig
from src.model_development import LinearRegressionModel

@step
def train_model(
    X_train : pd.DataFrame, 
    X_test  : pd.DataFrame,
    y_train : pd.DataFrame,
    y_test  : pd.DataFrame,
    config  : ModelNameConfig
) -> RegressorMixin :
    
    try:
        
        model, trained_model = None, None
        
        if config.model_name == "LinearRegression" : 
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported.")
        
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e