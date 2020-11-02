"""The utils module provides useful helpers for e.g. Colab, dataname generation etc."""

"""
Medical lexicon NLP extraction pipeline

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py') and not basename(f)[0] == '_']

import importlib
import os 

for _ in __all__:
    importlib.import_module('na_pipeline_tool.utils.'+ _)

try:
    import google.colab
    from tqdm.notebook import tqdm
    PROGRESSBAR_TYPE = "tqdm_notebook"
except:
    from tqdm import tqdm
    PROGRESSBAR_TYPE = "tqdm"

def progressbar(*args, **kwargs):
    """Create a progress bar based on the current environment

    Yields:
        [type]: [description]
    """
    yield from tqdm(*args, **kwargs)

def default_dataframe_name(filename):
    """Generate the name of a result dataframe parquet file based on the current config options.

    Args:
        filename (str): The parquet base file name

    Returns:
        str: The full path to the parquet file
    """

    overall_project_name = config.get('overall_project_name', 'lexicon_labeled_notes')
    generated_df_dir = config.get('generated_df_dir', 'generated')
    if not os.path.isdir(generated_df_dir):
            os.mkdir(generated_df_dir)

    filename = os.path.join(generated_df_dir, overall_project_name + filename)
    return filename
