import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def Predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            features.columns = features.columns.str.strip().str.lower().str.replace(" ", "_")
            expected_cols = preprocessor.feature_names_in_
            
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = 0
            features = features[expected_cols]

           
            expected_cols = preprocessor.feature_names_in_
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
         gender : str,
        race_ethnicity : str,
        parental_level_of_education : str,
        lunch :str,
        test_preparation_course : str,
        reading_score : int,
        writing_score : int,
        ):
        self.gender  =  gender
        self.race_ethnicity =race_ethnicity 
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender" : [self.gender],
                "Race or Ethnicity" : [self.race_ethnicity],
                "parental level of education" : [self.parental_level_of_education],
                "Lunch Type" : [self.lunch],
                "Test_preparation_Course" : [self.test_preparation_course],
                "Reading Score out of 100Reading Score out of 100" : [self.reading_score],
                "Writing Score out of 100" : [self.writing_score],
            }
            df = pd.DataFrame(custom_data_input_dict)
           
            import re
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(r"[^\w]+", "_", regex=True)  
            )

            return df
        except Exception as e:
            raise CustomException(e,sys)

