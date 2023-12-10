import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    ''' abstract class defining strategy for handling data '''
    
    @abstractmethod
    def handle_data(self, df : pd.DataFrame) -> Union[pd.DataFrame, pd.Series] :
        pass
    
class DataPreProcessStrategy(DataStrategy):
    ''' strategy for preprocessing data '''
    
    def _impute_median(self, s : pd.Series) -> pd.Series :
        median = s.median()
        return s.fillna(median)
    
    def handle_data(self, df : pd.DataFrame) -> Union[pd.DataFrame, pd.Series] :
        ''' preprocesses df '''
        try:
            
            df = df.drop(
                [
                    'order_approved_at',
                    'order_delivered_carrier_date',
                    'order_delivered_customer_date',
                    'order_estimated_delivery_date',
                    'order_purchase_timestamp'
                ],
                axis=1
            )
            
            for column in ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']:
                df[column] = self._impute_median(df[column])
                
            df['review_comment_message'] = df['review_comment_message'].fillna('No review')
            
            df = df.select_dtypes(include=[np.number])
            
            columns_to_drop = ['customer_zip_code_prefix', 'order_item_id']
            df = df.drop(columns_to_drop, axis=1)
            
            return df
        
        except Exception as e :
            logging.error(f"Error processing data: {e}")
            raise e
    
class DataDivideStrategy(DataStrategy):
    ''' strategy for dividing data into train and test '''
    
    def handle_data(self, df : pd.DataFrame) -> Union[pd.DataFrame, pd.Series] :
        ''' Divide data into train/test split '''
        
        try:
            X = df.drop(["review_score"], axis=1)
            y = df['review_score']
            return train_test_split(X, y, test_size=.2, random_state=0)
        except Exception as e:
            logging.error("Error in dividing data: {e}")
            raise e
        
class DataCleaning:
    ''' class for cleaning which preprocesses the data and divides it into a train/test split '''
    
    def __init__(self, df : pd.DataFrame, strategy : DataStrategy) -> None:
        self.df = df
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series] :
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e