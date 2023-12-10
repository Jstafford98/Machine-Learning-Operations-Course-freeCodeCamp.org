from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    ''' Model configurations '''
    model_name : str = "LinearRegression"