import logging
import pandas as pd
from zenml import step
from typing import Tuple, Union
from typing_extensions import Annotated

from src.data_cleaning import (
    DataCleaning, 
    DataDivideStrategy, 
    DataPreProcessStrategy
)

@step
def clean_data(df : pd.DataFrame) -> Tuple[
    Annotated[Union[pd.DataFrame, pd.Series], "X_train"],
    Annotated[Union[pd.DataFrame, pd.Series], "X_test" ],
    Annotated[Union[pd.DataFrame, pd.Series], "y_train"],
    Annotated[Union[pd.DataFrame, pd.Series], "y_test" ],
]:
    ''' cleans the data and divides it into train/test split '''
    try:
        
        process_strategy = DataPreProcessStrategy()
        data_divide_strategy = DataDivideStrategy()
        
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        data_cleaning = DataCleaning(processed_data, data_divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        logging.info("Data cleaning complete.")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"Error in clean data: {e}")
        raise e