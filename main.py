import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import ast
import os
from models.model_evaluation import ModelEvaluator
from utils import setup_logging, load_and_validate_data
from config import MODEL_CONFIG, PATHS
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Setup logging
logger = setup_logging()

# Add this near the top of the file with your other imports and configurations
PREPROCESSING_CONFIG = {
    'symptoms': [
        'fever', 'cough', 'fatigue', 'shortness_of_breath', 'headache',
        'sore_throat', 'body_aches', 'nausea', 'diarrhea', 'loss_of_taste_smell'
    ]
}

class HealthcarePredictiveModel:
    def __init__(self):
        self.model = RandomForestClassifier(**MODEL_CONFIG['random_forest'])
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.gender_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
    def extract_vital_signs(self, vital_signs_str):
        """Safely extract vital signs from string"""
        try:
            # Handle empty or invalid input
            if not vital_signs_str or pd.isna(vital_signs_str):
                return {'systolic_bp': 120.0, 'diastolic_bp': 80.0, 'temperature': 37.0}
            
            # If the input is a lab result (wrong column), return defaults
            if isinstance(vital_signs_str, str) and '"wbc_count"' in vital_signs_str:
                return {'systolic_bp': 120.0, 'diastolic_bp': 80.0, 'temperature': 37.0}
            
            # If string is already a dict, convert to string first
            if isinstance(vital_signs_str, dict):
                vital_signs = vital_signs_str
            else:
                # Clean the string and parse JSON
                vital_signs_str = vital_signs_str.replace("'", '"').strip()
                vital_signs = json.loads(vital_signs_str)
            
            # Extract blood pressure with better error handling
            bp = vital_signs.get('blood_pressure', '120/80')
            try:
                systolic, diastolic = map(float, bp.split('/'))
            except (ValueError, AttributeError):
                systolic, diastolic = 120.0, 80.0
            
            # Extract temperature with validation
            temp = vital_signs.get('temperature', 37.0)
            temperature = float(temp) if temp else 37.0
            
            return {
                'systolic_bp': systolic,
                'diastolic_bp': diastolic,
                'temperature': temperature
            }
        except Exception as e:
            logger.error(f"Error parsing vital signs: {str(e)} for value: {vital_signs_str}")
            return {'systolic_bp': 120.0, 'diastolic_bp': 80.0, 'temperature': 37.0}

    def extract_lab_results(self, lab_results_str):
        """Safely extract lab results from string"""
        try:
            if not lab_results_str or pd.isna(lab_results_str):
                return {'wbc_count': 0, 'crp': 0}
            
            # If the input is a vital sign (wrong column), return defaults
            if isinstance(lab_results_str, str) and '"blood_pressure"' in lab_results_str:
                return {'wbc_count': 0, 'crp': 0}
            
            # Clean and parse JSON
            if isinstance(lab_results_str, dict):
                lab_results = lab_results_str
            else:
                # Handle partial JSON strings
                if '"crp":' in lab_results_str and '"wbc_count":' not in lab_results_str:
                    crp_value = float(lab_results_str.split('"crp":')[1].split('}')[0].strip())
                    return {'wbc_count': 0, 'crp': crp_value}
                
                if '"wbc_count":' in lab_results_str and '"crp":' not in lab_results_str:
                    wbc_value = float(lab_results_str.split('"wbc_count":')[1].split('}')[0].strip())
                    return {'wbc_count': wbc_value, 'crp': 0}
                
                lab_results_str = lab_results_str.replace("'", '"').strip()
                lab_results = json.loads(lab_results_str)
            
            return {
                'wbc_count': float(lab_results.get('wbc_count', 0)),
                'crp': float(lab_results.get('crp', 0))
            }
        except Exception as e:
            logger.error(f"Error parsing lab results: {str(e)} for value: {lab_results_str}")
            return {'wbc_count': 0, 'crp': 0}
        
    def prepare_data(self, df, is_training=True):
        """Prepare data for training or prediction"""
        try:
            df = df.copy()
            
            # Create numerical features DataFrame
            numerical_features = pd.DataFrame(index=df.index)
            
            # Process vital signs
            vital_signs_data = df['Vital Signs'].apply(self.extract_vital_signs)
            numerical_features['systolic_bp'] = vital_signs_data.apply(lambda x: x['systolic_bp'])
            numerical_features['diastolic_bp'] = vital_signs_data.apply(lambda x: x['diastolic_bp'])
            numerical_features['temperature'] = vital_signs_data.apply(lambda x: x['temperature'])
            
            # Process lab results
            lab_results_data = df['Lab Results'].apply(self.extract_lab_results)
            numerical_features['wbc_count'] = lab_results_data.apply(lambda x: x['wbc_count'])
            numerical_features['crp'] = lab_results_data.apply(lambda x: x['crp'])
            
            # Add Age
            numerical_features['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            
            # Fill NaN values with mean for numerical features
            for col in numerical_features.columns:
                numerical_features[col] = numerical_features[col].fillna(numerical_features[col].mean())
            
            # Process symptoms
            symptoms_df = pd.DataFrame(0, index=df.index, columns=PREPROCESSING_CONFIG['symptoms'])
            df['Symptoms'] = df['Symptoms'].fillna('')
            
            for idx, symptoms in df['Symptoms'].items():
                if isinstance(symptoms, str) and symptoms:
                    symptom_list = [s.strip().lower() for s in symptoms.split(',')]
                    for symptom in symptom_list:
                        # Map similar symptoms
                        if symptom in ['loss_of_taste', 'loss_of_taste_smell']:
                            symptom = 'loss_of_taste'
                        if symptom in ['body_ache', 'body_aches']:
                            symptom = 'body_ache'
                        
                        if symptom in PREPROCESSING_CONFIG['symptoms']:
                            symptoms_df.at[idx, symptom] = 1
            
            # Process Gender using OneHotEncoder
            gender_data = df[['Gender']]
            if is_training:
                gender_encoded = self.gender_encoder.fit_transform(gender_data)
            else:
                gender_encoded = self.gender_encoder.transform(gender_data)
            
            gender_feature_names = [f'gender_{cat}' for cat in self.gender_encoder.categories_[0]]
            gender_df = pd.DataFrame(
                gender_encoded,
                columns=gender_feature_names,
                index=df.index
            )
            
            # Combine all features
            X = pd.concat([numerical_features, symptoms_df, gender_df], axis=1)
            
            # Ensure no NaN values remain
            X = X.fillna(0)
            
            if is_training:
                # Clean the Diagnosis column
                df['Diagnosis'] = df['Diagnosis'].apply(lambda x: 
                    'COVID-19' if isinstance(x, str) and 'temperature' in x and float(x.split(':')[1].strip('} ')) >= 38.0 
                    else ('Normal' if isinstance(x, str) and 'temperature' in x else x)
                )
                
                y = df['Diagnosis']
                logger.info(f"Processed features shape: {X.shape}")
                logger.info(f"Feature names: {X.columns.tolist()}")
                logger.info(f"Unique diagnoses: {y.unique().tolist()}")
                logger.info(f"Class distribution:\n{y.value_counts()}")
                return X, y
            else:
                return X
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise

    def train(self, X, y):
        """Train the model"""
        try:
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert to DataFrame to preserve column names
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            
            # Apply SMOTE for class balancing
            try:
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            except Exception as e:
                logger.warning(f"SMOTE failed: {str(e)}. Using original data.")
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
            
            # Train the model
            self.model.fit(X_train_balanced, y_train_balanced)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Convert predictions back to original labels
            y_pred_labels = self.label_encoder.inverse_transform(y_pred)
            y_test_labels = self.label_encoder.inverse_transform(y_test)
            
            # Log results
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_test_labels, y_pred_labels))
            
            return X_test_scaled, y_test_labels, y_pred_labels
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def save_model(self, filename):
        """Save the model and its components"""
        try:
            os.makedirs(PATHS['models'], exist_ok=True)
            base_path = os.path.join(PATHS['models'], filename)
            
            # Save the main model
            joblib.dump(self.model, f'{base_path}_model.pkl')
            
            # Save the encoders and other components
            components = {
                'label_encoder': self.label_encoder,
                'gender_encoder': self.gender_encoder,
                'scaler': self.scaler,
            }
            joblib.dump(components, f'{base_path}_components.pkl')
            
            logger.info(f"Model and components saved to {base_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

def main():
    try:
        # Load and validate data
        data_path = os.path.join(PATHS['data'], 'healthcare_dataset.csv')
        df = load_and_validate_data(data_path)
        
        logger.info(f"Initial dataset size: {len(df)}")
        logger.info(f"Initial diagnosis distribution:\n{df['Diagnosis'].value_counts()}")
        
        # Initialize model
        model = HealthcarePredictiveModel()
        
        # Prepare data
        X_processed, y = model.prepare_data(df)
        
        # Train model
        X_test, y_test, y_pred = model.train(X_processed, y)
        
        # Save model
        model.save_model('healthcare_predictor')
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
