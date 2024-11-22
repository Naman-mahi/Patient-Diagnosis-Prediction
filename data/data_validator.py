import pandas as pd
import numpy as np
from datetime import datetime
import json

class DataValidator:
    def __init__(self, config):
        self.config = config
        self.validation_results = []
        
    def validate_patient_data(self, df):
        """Validate patient data against rules"""
        # Check age range
        age_issues = df[~df['Age'].between(0, 120)]['Patient ID'].tolist()
        if age_issues:
            self.validation_results.append({
                'issue': 'Invalid age',
                'patients': age_issues
            })
            
        # Check gender values
        gender_issues = df[~df['Gender'].isin(['M', 'F'])]['Patient ID'].tolist()
        if gender_issues:
            self.validation_results.append({
                'issue': 'Invalid gender',
                'patients': gender_issues
            })
            
        # Validate vital signs
        vital_signs_issues = []
        for idx, row in df.iterrows():
            vital_signs = row['Vital Signs']
            if isinstance(vital_signs, str):
                vital_signs = json.loads(vital_signs)
            warnings = self._validate_vital_signs(vital_signs)
            if warnings:
                vital_signs_issues.append({
                    'patient_id': row['Patient ID'],
                    'warnings': warnings
                })
                
        if vital_signs_issues:
            self.validation_results.append({
                'issue': 'Vital signs warnings',
                'details': vital_signs_issues
            })
            
    def _validate_vital_signs(self, vital_signs):
        """Validate individual vital signs"""
        warnings = []
        
        # Validate temperature
        temp = vital_signs.get('temperature')
        if temp and (temp < 35 or temp > 42):
            warnings.append(f'Temperature {temp} is outside normal range')
            
        # Validate blood pressure
        bp = vital_signs.get('blood_pressure')
        if bp:
            try:
                systolic, diastolic = map(int, bp.split('/'))
                if systolic < 70 or systolic > 200:
                    warnings.append(f'Systolic BP {systolic} is outside normal range')
                if diastolic < 40 or diastolic > 120:
                    warnings.append(f'Diastolic BP {diastolic} is outside normal range')
            except:
                warnings.append('Invalid blood pressure format')
                
        return warnings
        
    def generate_validation_report(self):
        """Generate a validation report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'data/validation_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'validation_results': self.validation_results
            }, f, indent=4)
            
        return report_path 