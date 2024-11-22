import logging
import os
import pandas as pd
from datetime import datetime
from config import PATHS, VITAL_SIGNS_RANGES

def setup_logging():
    """Set up logging configuration"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'healthcare_predictor_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_vital_signs(value, vital_type):
    """Validate if a vital sign is within normal range"""
    ranges = VITAL_SIGNS_RANGES[vital_type]
    return ranges['min'] <= value <= ranges['max']

def load_and_validate_data(file_path):
    """Load and perform initial data validation"""
    try:
        df = pd.read_csv(file_path)
        required_columns = [
            'Patient ID', 'Age', 'Gender', 'Symptoms',
            'Diagnosis', 'Vital Signs', 'Lab Results'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise 