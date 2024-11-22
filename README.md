# Healthcare Predictive Modeling System

## Overview
This project is a machine learning-based healthcare prediction system that helps diagnose various conditions based on patient symptoms, vital signs, and laboratory results. The system uses advanced AI algorithms to provide accurate disease predictions and risk assessments.

## Features
- Real-time disease prediction
- Interactive web interface
- Patient history tracking
- Analytics dashboard
- Automated reporting
- Multi-disease classification
- Confidence score for predictions

## Diseases Covered
- COVID-19
- Heart Disease
- Influenza
- Pneumonia

## Technical Stack
- **Backend**: Python 3.9+
- **ML Framework**: scikit-learn
- **Web Interface**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model**: Random Forest Classifier with SMOTE balancing

## Project Structure
healthcare_predictor/
├── app.py                    # Streamlit web application for user interface
├── main.py                   # Main script for model training and evaluation
├── config.py                 # Configuration and environment settings
├── utils.py                  # Helper functions and utilities
├── requirements.txt          # Python package dependencies
├── setup.py                  # Installation and setup configuration
├── Dockerfile               # Container configuration for deployment
├── data/
│   ├── healthcare_dataset.csv  # Training and validation dataset
│   └── data_validator.py       # Data validation and preprocessing
├── models/
│   ├── model_evaluation.py     # Model performance evaluation tools
│   └── model_registry.py       # Model versioning and management
└── logs/                     # Application and error logs


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Naman-mahi/Patient-Diagnosis-Prediction
```


2. Create and activate virtual environment:
```bash
python -m venv healthcare_predictor_env
source healthcare_predictor_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage

1. Train the model:
```bash
python main.py
```

2. Run the web application:
```bash
streamlit run app.py
```

3. Access the application at `http://localhost:8501`

## Model Features
- **Patient Information**: Age, Gender
- **Vital Signs**: Blood Pressure, Temperature
- **Lab Results**: WBC Count, CRP Levels
- **Symptoms**: Multiple symptoms including fever, cough, fatigue, etc.

## Deployment Options

### Streamlit Cloud (Recommended)
1. Push to GitHub
2. Visit share.streamlit.io
3. Connect your repository
4. Deploy

### Alternative Deployment Options
- **Railway.app**: Using railway.json configuration
- **Render**: Using render.yaml configuration
- **Docker**: Using provided Dockerfile
- **Google Cloud Run**: Container-based deployment

## Continuous Integration
The project includes GitHub Actions workflow for:
- Automated testing
- Dependency checking
- Code quality verification

## Model Performance
- Accuracy metrics
- Confusion matrix
- Feature importance analysis
- Class distribution reports

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset contributors
- Medical professionals for validation
- Open source community

## Contact
Sunil K. Khobragade - [@Naman-mahi](https://github.com/Naman-mahi)
Project Link: [https://github.com/Naman-mahi/Patient-Diagnosis-Prediction](https://github.com/Naman-mahi/Patient-Diagnosis-Prediction)

## Future Enhancements
- Additional disease categories
- Integration with electronic health records
- Mobile application
- API development
- Multi-language support
- Enhanced visualization features

## Screenshots

### Healthcare Prediction System
![Healthcare Prediction System](heathcare%20Predication%20System.png)

### Prediction History
![Prediction History](Prediction%20History.png)

### Common Symptoms Analysis
![Common Symptoms](Common%20Symptoms.png)

### Analytics Dashboard
![Analytics Dashboard](Analytics%20Dashboard.png)


## Documentation
For detailed documentation, please visit the [Wiki](https://github.com/Naman-mahi/Patient-Diagnosis-Prediction/wiki)