"""
General utility functions
"""
import os
from collections import UserDict
from pathlib import Path

import pandas as pd
import yaml


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_yaml_data(folder, filename):
    """
    Load generic yaml files from data and return dictionary
    """
    full_path = Path(__file__).parents[1] / "data/{}/{}".format(folder, filename)

    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    return yaml_dict


def load_yaml_config(filename, config_subfolder=None):
    """
    Load generic yaml files from config and return dictionary
    """
    if config_subfolder:
        full_path = Path(__file__).parents[1] / "config/{}/{}".format(
            config_subfolder, filename
        )
    else:
        full_path = Path(__file__).parents[1] / "config/{}".format(filename)

    with open(full_path) as file:
        yaml_dict = yaml.full_load(file)

    return yaml_dict


def load_data_sources(key=None):
    """
    Load the yaml file containing all data sources
    """
    params = load_yaml_config("data_sources.yml")

    if key is not None:
        params = params[key]

    return params


def load_parameters(key=None):
    """
    Load parameters
    """
    params = load_yaml_config("parameters.yml")

    if key is not None:
        params = params[key]

    return params


def load_databases(env):
    """
    Load databases
    """
    databases = load_yaml_config("databases.yml")[env]
    secrets = load_yaml_config("databases_secrets.yml", "secrets")[env]
    return databases, secrets


def load_pairwise_entities():
    """
    Load pairwise entities to conform to Google News model n-grams
    E.g., (african, union) => "African_Union"
    """
    entities_file = (
        Path(__file__).parents[1] / "contextualization/pairwise_triplewise_entities.yml"
    )

    with open(entities_file) as file:
        entities = yaml.full_load(file)

    return entities


def load_custom_wvs():
    """
    Load custom Word2vec embeddings
    """
    custom_wv_file = Path(__file__).parents[1] / "contextualization/custom_wvs.yml"

    with open(custom_wv_file) as file:
        custom_wv = yaml.full_load(file)

    return custom_wv


def load_tags_guiding_typos():
    """
    Load tags that guide Hunspell correction
    """
    tags_file = Path(__file__).parents[1] / "contextualization/tags_guiding_typos.yml"

    with open(tags_file) as file:
        tags_guiding_typos = yaml.full_load(file)

    return tags_guiding_typos


def load_generic_dataset(data_source_name):
    """
    Load any dataset using the data_sources.yml name
    """
    data_sources = load_data_sources()
    my_data_source_info = data_sources[data_source_name]

    file_name = my_data_source_info["filename"]
    folder = my_data_source_info["folder"]
    args = my_data_source_info.get("args")
    file_type = file_name.split(".")[1]

    if args is None:
        args = {}

    path = Path(__file__).parents[2] / "data/{}/{}".format(folder, file_name)

    if file_type == "csv":
        dataset = pd.read_csv(path, **args)
    elif file_type == "dta":
        dataset = pd.read_stata(path, **args)
    elif file_type in ["xlsx", "xls"]:
        dataset = pd.read_excel(path, **args)
    else:
        raise NotImplementedError(
            "Cannot load file {} with extention {}".format(file_name, file_type)
        )

    return dataset


def get_postgres_uri(
    endpoint,
    port,
    database,
    username,
    password,
):
    """
    Returns PostgreSQL database URI given info and secrets
    """

    connection_uri = "postgresql://%s:%s@%s:%s/%s" % (
        username,
        password,
        endpoint,
        port,
        database,
    )

    return connection_uri


class DefaultEnvDict(UserDict):
    """
    Dictionary but uses env variables as defaults
    """

    def __missing__(self, key):
        """
        If `key` is missing, look for env variable with the same name.
        """

        value = os.getenv(key)
        if value is None:
            raise KeyError(f"{key} not found in dict or environment variables")
        return os.getenv(key)
