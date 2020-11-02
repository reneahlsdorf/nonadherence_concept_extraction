"""This module removes catch-all labels from data rows if at least one different label was found for it (e.g. general NA -> dietary NA)."""

"""
Medical lexicon NLP extraction pipeline

File contains: Contains the module for removing catchall class labels if there are other predicted class labels for a note 

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

from na_pipeline_tool import utils
from na_pipeline_tool.utils import logger
from na_pipeline_tool.utils import config
from na_pipeline_tool.utils import helper_classes

import re
import pandas as pd
import sys
import os


class CatchallLabelFiltering(helper_classes.Module):
    """This module removes catch-all labels from data rows if at least one different label was found for it (e.g. general NA -> dietary NA).
    """

    def __init__(self):
        super().__init__()

        self._output_note_file = config.get_pipeline_config_item(self.module_name(), 'output_note_file', '_negex_filtered_notes.parquet')
        self._df_notes_labeled_path = config.get_pipeline_config_item(self.module_name(), 'input_note_file', None)
        self._catchall = config.get_pipeline_config_item(self.module_name(), 'catchall', False)

        self._loaded_df = None
        self._filtered_cohort_df = None
        self._labeled_df = None

        logger.log_info('Loading note file')
        self._load_note_input_file()
        logger.log_info('DONE: Loading note file.')

        logger.log_info('Replacing catchall labels.')
        self._replace_catchall()
        logger.log_info('DONE: Replacing catchall labels.')

        logger.log_info('Dumping processed notes.')
        self._dump_processed_df()
        logger.log_info('DONE: Dumping processed notes.')


    def _load_note_input_file(self):
        if not self._df_notes_labeled_path:
            raise RuntimeError('Please specify a valid note input file.')
        
        filename = utils.default_dataframe_name(self._df_notes_labeled_path)

        assert os.path.isfile(filename), 'Could not find note parquet file: {}'.format(filename)
        self._loaded_df = pd.read_parquet(filename)
        self._loaded_df.columns = [_.upper() for _ in self._loaded_df.columns]
        assert 'PREDICTED_CATEGORIES' in self._loaded_df.columns and 'SUBJECT_ID' in self._loaded_df.columns and 'CHARTDATE' in self._loaded_df.columns and 'CATEGORY' in self._loaded_df.columns and 'TEXT' in self._loaded_df.columns and 'ROW_ID' in self._loaded_df.columns and 'HADM_ID' in self._loaded_df.columns, 'Notes file need to have columns: Row_id, Subject_id, Hadm_id, chartdate, category and text'

    def _replace_catchall(self):
        assert self._catchall, 'Please specify the "catchall" option in the YAML file'
        assert self._catchall.upper() in self._loaded_df.columns, 'Catchall label was not found in the given input file please re-check!'
        
        self._catchall = self._catchall.upper()
        for _i in self._loaded_df.index:
            predicted_cats = self._loaded_df.loc[_i,'PREDICTED_CATEGORIES'].split('|')
            if self._catchall in predicted_cats and len(predicted_cats) > 1:
                predicted_cats.remove(self._catchall)
                predicted_cats = '|'.join(predicted_cats)
                self._loaded_df.loc[_i,'PREDICTED_CATEGORIES'] = predicted_cats

    def _dump_processed_df(self):
        filename = utils.default_dataframe_name(self._output_note_file)
        self._loaded_df.to_parquet(filename)

    @classmethod
    def register_argparser_object(cls, subparser_instance):
        subparser_instance.add_parser(cls.module_name(), help="Removes the specified catchall class label if more labels than this label only were predicted for a note.")
    
