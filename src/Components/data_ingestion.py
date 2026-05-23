import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.Components.data_transformation import DataTransformation
from src.Components.data_transformation import DataTransformationConfig
from src.Components.model_trainer import ModelTrainerConfig
from src.Components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        """An object can store multiple values in it, so here the ingestion_config will have 3 values in it, 
        and they are stored as class object which exactly looks like dictionary
        (donot assume values would be stored inform of tuple)
        --> ingestion_config = {
                                "train_data_path": "artifacts/train.csv",
                                "test_data_path": "artifacts/test.csv",
                                "raw_data_path": "artifacts/data.csv"
                            }       
        """
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv("data/stud.csv")
            logging.info("Read the dataset as dataframe")

            #It extracts the folder (artifacts) from the file path and creates it if it doesn’t already exist.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path)

            #logging the event again
            logging.info("Train test split is initiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
    
            #logging th event again
            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path  
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation  = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


