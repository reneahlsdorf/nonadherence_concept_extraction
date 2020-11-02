"""This modules deals with importing the config YAML file and offers methods which allow the simple extraction of config items from the config."""

"""
Medical lexicon NLP extraction pipeline

File contains: Helper methods for using the global logger

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

from ruamel.yaml import YAML
import sys, os 
import collections

# Hide these variables from the outside world since users should use this module's get method for accessing config items
_YAML_DICT = dict()
_YAML_DICT_KEYS = list()

def init_config(config_file_path):
    """Init the config module by importing the YAML file.

    Args:
        config_file_path (str): The path to the YAML file.
    """

    global _YAML_DICT, _YAML_DICT_KEYS
    if not os.path.isfile(config_file_path):
        print('ERROR: There is no file under the provided file path: ' + str(config_file_path))
        sys.exit(1)
    
    yaml_file = open(config_file_path, 'rb')
    yaml_file_content = yaml_file.read()
    yaml_file.close()
    yaml = YAML()
    _YAML_DICT = yaml.load(yaml_file_content)

    new_dict = dict()
    for _key, _val in _YAML_DICT.items():
        new_dict[_key.lower()] = _val
        if isinstance(_YAML_DICT[_key], collections.Mapping):
            _YAML_DICT_KEYS.append(_key.lower())
    _YAML_DICT = new_dict

def override_in_pipeline_config(pipeline_mod_name, key, value):
    """Override item in loaded YAML config dict of pipeline stage.

    Args:
        pipeline_mod_name (str): Snake case name of pipeline stage module
        key (str): The key to be overriden
        value: The new value

    Returns:
        bool: True if operation was successful (both pipeline config and key existed and could be overriden)
    """
    subdict = _YAML_DICT.get(pipeline_mod_name.lower(), None)
    key = key.lower()
    if not subdict or not key in subdict:
        return False
    subdict[key] = value
    return True

def pipeline_config_has(pipeline_mod_name, key):
    """Check if key in pipeline stage config

    Args:
        pipeline_mod_name (str): Snake case name of pipeline stage module
        key (str): The key which should be checked

    Returns:
        bool: True if key in config dict
    """
    subdict = _YAML_DICT.get(pipeline_mod_name.lower(), None)
    key = key.lower()
    if not subdict or not key in subdict:
        return False
    return True

def get(key, default):
    """Get a config bloc from the YAML config file.

    Args:
        default (dict): The default bloc if the key is not available

    Returns:
        dict: The config bloc (or the default one)
    """
    if not key.lower() in _YAML_DICT or isinstance(_YAML_DICT[key.lower()], collections.Mapping):
        return default
    else:
        return _YAML_DICT[key.lower()]


def get_pipeline_config_item(pipeline_mod_name, key, default):
    """Get the config options for a pipeline module.

    Args:
        pipeline_mod_name (str): Name of the pipeline module
        key (str): The config item key you're interested in
        default: The default value if the config item is not present

    Returns:
        object: The config item value (or its default)
    """
    if not pipeline_mod_name.lower() in _YAML_DICT or not key.lower() in _YAML_DICT[pipeline_mod_name.lower()] or not isinstance(_YAML_DICT[pipeline_mod_name.lower()], collections.Mapping):
        return default
    else:
        return _YAML_DICT[pipeline_mod_name.lower()][key.lower()]
