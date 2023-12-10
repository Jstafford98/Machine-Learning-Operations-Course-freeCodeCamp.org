import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    ''' abstract class defining strategy for evaluating models '''
    
    @abstractmethod
    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray):
        ''' calculates scores for the model '''
        pass
    
class MSE(Evaluation):
    ''' evaluation strategy that uses Mean Squared Error '''
    
    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray):
        try:
            logging.info("Entered calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f'Error in calculating MSE: {e}')
            raise e
        
class R2(Evaluation):
    ''' evaluation strategy that uses r2_score '''
    
    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray):
        try:
            logging.info("Entered calculating R2")
            mse = r2_score(y_true, y_pred)
            logging.info(f"R2: {mse}")
            return mse
        except Exception as e:
            logging.error(f'Error in calculating R2: {e}')
            raise e
        
class RMSE(Evaluation):
    ''' evaluation strategy using root mean squared error '''
    
    def calculate_scores(self, y_true : np.ndarray, y_pred : np.ndarray):
        try:
            logging.info("Entered calculating RMSE")
            mse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f'Error in calculating RMSE: {e}')
            raise e