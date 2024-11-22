import os
import json
from datetime import datetime

class ModelRegistry:
    def __init__(self, registry_path='models/registry.json'):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        
    def _load_registry(self):
        """Load existing registry or create new one"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
        
    def register_model(self, model_name, model_path, metrics, parameters):
        """Register a new model with its metadata"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        model_info = {
            'model_path': model_path,
            'timestamp': timestamp,
            'metrics': metrics,
            'parameters': parameters,
            'status': 'active'
        }
        
        if model_name not in self.registry:
            self.registry[model_name] = []
            
        self.registry[model_name].append(model_info)
        self._save_registry()
        
    def _save_registry(self):
        """Save registry to file"""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)
            
    def get_latest_model(self, model_name):
        """Get the latest version of a model"""
        if model_name in self.registry and self.registry[model_name]:
            return self.registry[model_name][-1]
        return None
        
    def get_model_history(self, model_name):
        """Get all versions of a model"""
        return self.registry.get(model_name, []) 