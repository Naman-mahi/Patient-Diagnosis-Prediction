# Model configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'random_state': 42
    }
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'numerical_features': [
        'Age', 'systolic_bp', 'diastolic_bp', 
        'temperature', 'wbc_count', 'crp'
    ],
    'categorical_features': ['Gender'],
    'symptoms': [
        'fever',
        'cough',
        'fatigue',
        'shortness_of_breath',
        'loss_of_taste',
        'body_ache',
        'headache',
        'sore_throat',
        'nausea',
        'diarrhea'
    ]
}

# Vital signs normal ranges
VITAL_SIGNS_RANGES = {
    'systolic_bp': {'min': 60, 'max': 200},
    'diastolic_bp': {'min': 40, 'max': 130},
    'temperature': {'min': 35, 'max': 42},
    'wbc_count': {'min': 4000, 'max': 20000},
    'crp': {'min': 0, 'max': 100}
}

# File paths
PATHS = {
    'data': 'data',
    'models': 'models',
    'results': 'results'
} 