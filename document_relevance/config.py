"""
Configuration module for Document Relevance Classification System.
Contains default settings and configuration loading functions.
"""

import os
import json
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    'model_name': 'all-mpnet-base-v2',
    'similarity_threshold': 0.65,
    'reference_dir': './reference_documents',
    'kb_path': './data/knowledge_base/kb.pkl',
    'feedback_path': './data/feedback/feedback.pkl',
    'log_level': 'INFO',
    'log_dir': 'logs',
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'use_ml_if_available': True
}

def load_config(config_file: str = None) -> Dict[str, Any]:
    """
    Load configuration from file and merge with defaults.
    
    Args:
        config_file: Path to a JSON configuration file
        
    Returns:
        Dict containing configuration settings
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load from file if specified
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            config.update(file_config)
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    return config

def save_config(config: Dict[str, Any], config_file: str):
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to save the configuration
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config file: {e}")
