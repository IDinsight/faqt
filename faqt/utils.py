"""
General utility functions
"""
import os
from collections import UserDict
from pathlib import Path

import yaml


def load_yaml_data(folder, filename):
    """
    Load generic yaml files from data and return dictionary
    """
    full_path = Path(__file__).parents[1] / "data/{}/{}".format(folder, filename)

    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    return yaml_dict
