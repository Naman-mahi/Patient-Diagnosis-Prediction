import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
try:
    import joblib
except ImportError:
    st.error("Error: joblib not installed. Please run 'pip install joblib'")
    st.stop()

try:
    import plotly.express as px
except ImportError:
    st.error("Error: plotly not installed. Please run 'pip install plotly'")
    st.stop()

from main import HealthcarePredictiveModel, PREPROCESSING_CONFIG

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Initialize prediction history file if it doesn't exist
if not os.path.exists('prediction_history.json'):
    with open('prediction_history.json', 'w') as f:
        json.dump([], f)

def load_model():
    """Load the trained model and its components"""
    try:
        model = joblib.load('models/healthcare_predictor_model.pkl')
        components = joblib.load('models/healthcare_predictor_components.pkl')
        return model, components
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first using main.py")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def create_prediction_input():
    """Create input fields for prediction"""
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["M", "F"])
        temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
        
    with col2:
        systolic_bp = st.number_input("Systolic Blood Pressure", min_value=60, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=130, value=80)
        
    st.subheader("Laboratory Results")
    col3, col4 = st.columns(2)
    
    with col3:
        wbc_count = st.number_input("WBC Count", min_value=0, max_value=50000, value=10000)
    with col4:
        crp = st.number_input("CRP Level", min_value=0, max_value=100, value=10)
    
    st.subheader("Symptoms")
    symptoms = st.multiselect(
        "Select Symptoms",
        PREPROCESSING_CONFIG['symptoms'],
        default=[]
    )
    
    # Create a DataFrame with a single row
    data = {
        'Patient ID': 'NEW',
        'Age': age,
        'Gender': gender,
        'Vital Signs': {
            'blood_pressure': f"{systolic_bp}/{diastolic_bp}",
            'temperature': temperature
        },
        'Lab Results': {
            'wbc_count': wbc_count,
            'crp': crp
        },
        'Symptoms': ",".join(symptoms),
        'Diagnosis': 'Unknown'
    }
    return pd.DataFrame([data])

def initialize_app():
    """Initialize application files and directories"""
    try:
        # Create necessary directories
        for directory in ['data', 'models', 'logs']:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize history file if it doesn't exist
        history_file = 'prediction_history.json'
        if not os.path.exists(history_file):
            with open(history_file, 'w') as f:
                json.dump([], f)
        
        # Validate history file
        try:
            with open(history_file, 'r') as f:
                json.load(f)
        except json.JSONDecodeError:
            # If file is corrupted, create new one
            with open(history_file, 'w') as f:
                json.dump([], f)
                
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        logger.error(f"Error initializing application: {str(e)}")

def add_footer():
    """Add a custom footer to the app"""
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: white;
                color: black;
                text-align: center;
                padding: 20px;
                font-size: 14px;
            }
            .footer a {
                color: #4CAF50;
                text-decoration: none;
            }
            .heart {
                color: #e25555;
            }
        </style>
        <div class="footer">
            Developed with <span class="heart">❤</span> by 
            <a href="https://github.com/Naman-mahi" target="_blank">Namanmahi</a>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    initialize_app()
    st.set_page_config(
        page_title="Healthcare Prediction System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Healthcare Prediction System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict", "View History", "Analytics"])
    
    if page == "Predict":
        st.header("Patient Diagnosis Prediction")
        
        # Input form
        input_df = create_prediction_input()
        
        if st.button("Predict Diagnosis"):
            try:
                # Initialize model and prepare data
                model = HealthcarePredictiveModel()
                
                # Load trained model and components
                trained_model, components = load_model()
                model.model = trained_model
                model.label_encoder = components['label_encoder']
                model.gender_encoder = components['gender_encoder']
                model.scaler = components['scaler']
                
                # Prepare the input data for prediction
                X_processed = model.prepare_data(input_df, is_training=False)
                
                # Make prediction
                prediction = model.model.predict(X_processed)
                
                # Display prediction
                st.success(f"Predicted Diagnosis: {prediction[0]}")
                
                # Display confidence scores
                if hasattr(model.model, 'predict_proba'):
                    probabilities = model.model.predict_proba(X_processed)
                    st.write("Confidence Scores:")
                    for class_label, prob in zip(model.model.classes_, probabilities[0]):
                        st.write(f"{class_label}: {prob:.2%}")
                
                # Save prediction to history
                save_to_history(input_df.iloc[0].to_dict(), prediction[0])
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please ensure all required fields are filled correctly")
    
    elif page == "View History":
        st.header("Prediction History")
        history = load_history()
        if history:
            df = pd.DataFrame(history)
            st.dataframe(df)
            
            # Download button for history
            csv = df.to_csv(index=False)
            st.download_button(
                "Download History",
                csv,
                "prediction_history.csv",
                "text/csv"
            )
        else:
            st.info("No prediction history available")
    
    elif page == "Analytics":
        st.header("Analytics Dashboard")
        history = load_history()
        if history:
            df = pd.DataFrame(history)
            
            # Show diagnosis distribution
            fig1 = px.pie(df, names='Prediction', title='Diagnosis Distribution')
            st.plotly_chart(fig1)
            
            # Show age distribution by diagnosis
            fig2 = px.box(df, x='Prediction', y='Age', title='Age Distribution by Diagnosis')
            st.plotly_chart(fig2)
            
            # Show symptom frequency
            symptom_counts = analyze_symptoms(df)
            fig3 = px.bar(symptom_counts, title='Common Symptoms')
            st.plotly_chart(fig3)
        else:
            st.info("No data available for analytics")
    
    # Add footer at the end
    add_footer()

def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    return obj

def save_to_history(patient_data, prediction):
    """Save prediction to history"""
    try:
        history = load_history()
        
        # Convert all data to native Python types
        cleaned_data = convert_numpy_types(patient_data)
        
        # Add timestamp and prediction
        cleaned_data['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cleaned_data['Prediction'] = str(prediction)
        
        # Ensure Vital Signs and Lab Results are properly formatted
        if isinstance(cleaned_data.get('Vital Signs'), str):
            cleaned_data['Vital Signs'] = json.loads(cleaned_data['Vital Signs'].replace("'", '"'))
        if isinstance(cleaned_data.get('Lab Results'), str):
            cleaned_data['Lab Results'] = json.loads(cleaned_data['Lab Results'].replace("'", '"'))
        
        history.append(cleaned_data)
        
        # Create directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save to file
        with open('prediction_history.json', 'w') as f:
            json.dump(history, f, indent=4)
            
        st.success("Prediction saved to history successfully!")
        
    except Exception as e:
        st.error(f"Error saving to history: {str(e)}")
        logger.error(f"Error saving to history: {str(e)}")

def load_history():
    """Load prediction history"""
    try:
        history_file = 'prediction_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                content = f.read()
                if not content:  # If file is empty
                    return []
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    st.warning("Error reading history file. Creating new history.")
                    # Backup corrupted file
                    backup_file = f'prediction_history_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    os.rename(history_file, backup_file)
                    return []
        return []
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")
        logger.error(f"Error loading history: {str(e)}")
        return []

def analyze_symptoms(df):
    """Analyze symptom frequency"""
    try:
        all_symptoms = []
        for symptoms in df['Symptoms']:
            if symptoms and isinstance(symptoms, str):
                all_symptoms.extend([s.strip() for s in symptoms.split(',') if s.strip()])
        
        symptom_counts = pd.Series(all_symptoms).value_counts()
        return symptom_counts
    except Exception as e:
        st.error(f"Error analyzing symptoms: {str(e)}")
        logger.error(f"Error analyzing symptoms: {str(e)}")
        return pd.Series()

if __name__ == "__main__":
    main() 