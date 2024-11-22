try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    raise ImportError("Please install matplotlib and seaborn: pip install matplotlib seaborn")

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
import os
import warnings

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, y_pred):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Add diagnostic information
        self.analyze_class_distribution()
        
    def analyze_class_distribution(self):
        """Analyze and log class distribution in test and predicted data"""
        unique_test, counts_test = np.unique(self.y_test, return_counts=True)
        unique_pred, counts_pred = np.unique(self.y_pred, return_counts=True)
        
        print("\nClass Distribution Analysis:")
        print("Test Data Distribution:")
        for label, count in zip(unique_test, counts_test):
            print(f"Class {label}: {count} samples")
        
        print("\nPredicted Data Distribution:")
        for label, count in zip(unique_pred, counts_pred):
            print(f"Class {label}: {count} samples")
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        try:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(self.y_test, self.y_pred)
            
            # Get unique labels
            labels = np.unique(np.concatenate([self.y_test, self.y_pred]))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(f'models/confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix: {str(e)}")
        
    def plot_feature_importance(self, feature_names):
        """Plot feature importance"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                indices = np.argsort(importance)[::-1]
                
                plt.figure(figsize=(15, 8))
                plt.title("Feature Importances")
                plt.bar(range(self.X_test.shape[1]), importance[indices])
                plt.xticks(range(self.X_test.shape[1]), 
                          [feature_names[i] for i in indices], 
                          rotation=45, ha='right')
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(f'models/feature_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.close()
        except Exception as e:
            print(f"Error plotting feature importance: {str(e)}")
            
    def generate_evaluation_report(self):
        """Generate a text report with model performance metrics"""
        try:
            # Suppress the specific warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # Get class labels and their counts
            unique_classes = np.unique(np.concatenate([self.y_test, self.y_pred]))
            
            # Generate the classification report with zero_division parameter
            report = classification_report(
                self.y_test, 
                self.y_pred,
                labels=unique_classes,
                zero_division=1,  # Handle zero division cases
                digits=4  # Increase precision in the report
            )
            
            # Save the report with additional information
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f'models/evaluation_report_{timestamp}.txt'
            
            with open(report_path, 'w') as f:
                f.write("Model Evaluation Report\n")
                f.write("=====================\n\n")
                
                # Add class distribution information
                f.write("Class Distribution Analysis:\n")
                f.write("-------------------------\n")
                unique_test, counts_test = np.unique(self.y_test, return_counts=True)
                f.write("\nTest Data Distribution:\n")
                for label, count in zip(unique_test, counts_test):
                    f.write(f"Class {label}: {count} samples\n")
                
                unique_pred, counts_pred = np.unique(self.y_pred, return_counts=True)
                f.write("\nPredicted Data Distribution:\n")
                for label, count in zip(unique_pred, counts_pred):
                    f.write(f"Class {label}: {count} samples\n")
                
                f.write("\nClassification Report:\n")
                f.write("--------------------\n")
                f.write(report)
                
            print(f"Evaluation report saved to: {report_path}")
            
        except Exception as e:
            print(f"Error generating evaluation report: {str(e)}") 