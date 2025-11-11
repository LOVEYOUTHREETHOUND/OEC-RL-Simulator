# -*- coding: utf-8 -*-
"""
This module provides a utility function to load YAML configuration files.
"""

import yaml
import os

def load_config(file_path: str):
    """
    Loads a YAML configuration file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        dict: The configuration loaded from the YAML file.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found at: {os.path.abspath(file_path)}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {file_path}")
            raise e
