import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class HealthcareDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        
    def process_vital_signs(self, vital_signs_dict):
        """Convert vital signs dictionary to numerical values"""
        bp_systolic = float(vital_signs_dict['blood_pressure'].split('/')[0])
        bp_diastolic = float(vital_signs_dict['blood_pressure'].split('/')[1])
        temperature = float(vital_signs_dict['temperature'])
        return pd.Series({
            'bp_systolic': bp_systolic,
            'bp_diastolic': bp_diastolic,
            'temperature': temperature
        })
    
    def process_lab_results(self, lab_results_dict):
        """Process laboratory results"""
        return pd.Series({
            'wbc_count': lab_results_dict['wbc_count'],
            'crp': lab_results_dict['crp']
        })
    
    def process_symptoms(self, symptoms_str):
        """Convert symptoms string to binary features"""
        all_symptoms = ['fever', 'cough', 'fatigue', 'shortness_of_breath', 
                       'loss_of_taste', 'body_ache']
        patient_symptoms = symptoms_str.split(',')
        return pd.Series({
            symptom: 1 if symptom in patient_symptoms else 0 
            for symptom in all_symptoms
        })
    
    def preprocess_data(self, df):
        """Main preprocessing function"""
        # Process categorical variables
        df['Gender'] = self.label_encoder.fit_transform(df['Gender'])
        
        # Process vital signs
        vital_signs_df = pd.DataFrame(df['Vital Signs'].apply(self.process_vital_signs).tolist())
        
        # Process lab results
        lab_results_df = pd.DataFrame(df['Lab Results'].apply(self.process_lab_results).tolist())
        
        # Process symptoms
        symptoms_df = pd.DataFrame(df['Symptoms'].apply(self.process_symptoms).tolist())
        
        # Combine all features
        processed_df = pd.concat([
            df[['Age', 'Gender']],
            vital_signs_df,
            lab_results_df,
            symptoms_df
        ], axis=1)
        
        # Handle missing values
        processed_df = pd.DataFrame(
            self.imputer.fit_transform(processed_df),
            columns=processed_df.columns
        )
        
        # Scale the features
        processed_df = pd.DataFrame(
            self.scaler.fit_transform(processed_df),
            columns=processed_df.columns
        )
        
        return processed_df 