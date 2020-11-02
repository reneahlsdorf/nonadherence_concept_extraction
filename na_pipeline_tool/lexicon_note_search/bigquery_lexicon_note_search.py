"""
Medical lexicon NLP extraction pipeline

File contains: Contains the module for medical lexicon note parsing using bigquery 

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""


from na_pipeline_tool import utils
from na_pipeline_tool.utils import logger
from na_pipeline_tool.utils import config
from na_pipeline_tool.utils import google_tools
from na_pipeline_tool.utils import helper_classes
from na_pipeline_tool.utils import progressbar

import re
import pandas_gbq

import sys, os
import pandas as pd
import collections

class BigqueryMimicLexiconNoteSearchModule(helper_classes.Module):

    def __init__(self):
        super().__init__()
        
        if not google_tools.check_google_authenticated():
            logger.log_error('User not authenticated for Google cloud access.')
        
        self._output_note_file = config.get_pipeline_config_item(self.module_name(), 'output_note_file', '_labeled_lexicons.parquet')
        self._cohort_file = config.get_pipeline_config_item(self.module_name(), 'cohort_file', '')
        self._lexicon_dir = config.get_pipeline_config_item(self.module_name(), 'lexicon_dir', './used_lexicons')

        self._lexicon_map = { 'positive' : {}, 'negative' : {}}
        self._lexicon_weights = {'positive' : {}, 'negative' : {}}
        self._labeled_df = None
        self._ignore_cat_list = config.get_pipeline_config_item(self.module_name(), 'ignore_categories', [])
        self._dump_all = config.get_pipeline_config_item(self.module_name(), 'dump_all_notes', True)

        self._ignore_cat_list = [self._ignore_cat_list] if not isinstance(self._ignore_cat_list, collections.Sequence) else self._ignore_cat_list

        self._parse_lexicons()
        self._query_bigquery()
        self._dump_df()

    def _dump_df(self):
        logger.log_info('Dumping the extracted notes into a parquet file.')
        filename = utils.default_dataframe_name(self._output_note_file)
        self._labeled_df.to_parquet(filename)
        logger.log_info('DONE: Dumping the extracted notes into a parquet file.')

    def _parse_lexicons(self):
        assert os.path.isdir(self._lexicon_dir), 'Invalid lexicon dir. Does not exist.'
        assert len(os.listdir(self._lexicon_dir)) > 0, 'Lexicon dir is empty.'
        pos_dir = os.path.join(self._lexicon_dir, 'positive')
        neg_dir = os.path.join(self._lexicon_dir, 'negative')
        assert os.path.isdir(pos_dir), 'There needs to be a positive lexicon.'
        assert os.path.isdir(neg_dir), 'There needs to be a negative lexicon.'

        pos_files = os.listdir(pos_dir)
        neg_files = os.listdir(neg_dir)

        unknown_files = [_ for _ in neg_files if not _ in pos_files]
        assert len(unknown_files) == 0, 'The lexicon filenames in the positive and negative dirs need to match! Found: ' + str(unknown_files)

        def parse_dir(dirlist, prefix):
            for _lexi in dirlist:
                file = open(os.path.join(self._lexicon_dir, prefix, _lexi), 'r')
                filename = _lexi.strip()
                filename = re.sub(r'\..*', '', filename)
                filename = helper_classes.Module.camelcase_to_snakecase(filename)
                filename = filename.replace(' ', '_')
                lines = file.readlines()
                lines = [_.strip() for _ in lines if len(_) > 0]

                self._lexicon_map[prefix][filename] = []
                self._lexicon_weights[prefix][filename] = {}
                if lines:
                    for _ in lines:
                        term = _.split(';;')
                        assert len(term) > 0, 'Invalid line found in {} lexicon: {}'.format(prefix ,_)
                        if len(term[0]) < 1:
                            continue
                        if len(term) < 2:
                            self._lexicon_map[prefix][filename].append(term[0])
                            self._lexicon_weights[prefix][filename][term[0].lower()] = 2
                        else:
                            self._lexicon_map[prefix][filename].append(term[0])
                            self._lexicon_weights[prefix][filename][term[0].lower()] = (int(term[1]))
                file.close()


        logger.log_debug('Parsing the lexicons now..')
        parse_dir(pos_files, 'positive')
        parse_dir(neg_files, 'negative')

        for cat in ['positive', 'negative']:
            for _k, _v in self._lexicon_map[cat].items():
                logger.log_info('{} {} lexicon: {} entries'.format(cat, _k, len(_v)))
        
        logger.log_info('Parsed and stored all lexicons.')


    def _query_bigquery(self):

        sql_search = ""

        merged_lexicon_map = { _k : self._lexicon_map['positive'][_k] + self._lexicon_map['negative'][_k] for _k in self._lexicon_map['positive'].keys()}

        for _name, _terms  in merged_lexicon_map.items():
            if not _terms:
                sql_search = sql_search + "," + " FALSE AS " + _name
            else:
                lex = [r'\\b'+ x + r'\\b' for x in _terms]
                sql_search = sql_search + "," + " REGEXP_CONTAINS(text, '(?i)(" + '|'.join(lex) + ")') AS " + _name
            
        ignore_str = '\n'.join(['AND category NOT LIKE "%{}%"'.format(_) for _ in self._ignore_cat_list])

        use_bqstorage_api = config.get_pipeline_config_item(self.module_name(), "use_bqstorage_api", False)

        limitstr = ""
        if config.get_pipeline_config_item(self.module_name(), "debug_download", False):
            limitstr = 'LIMIT 10'

        cohort_ids = []
        if self._cohort_file and os.path.isfile(self._cohort_file):
            cohort_ids = pd.read_csv(self._cohort_file)
            cohort_ids.columns = [_.lower() for _ in cohort_ids.columns]
            cohort_ids = list(cohort_ids.loc[:,'hadm_id'])

        sql = """
        SELECT row_id, subject_id, hadm_id, chartdate, category, text{}
        FROM `physionet-data.mimiciii_notes.noteevents`
        WHERE hadm_id IS NOT NULL 
        AND hadm_id IN ({})
        {}
        {}
        """.format(sql_search, ','.join([str(_) for _ in cohort_ids]), ignore_str, limitstr)

        logger.log_info('Querying noteevents for lexicon occurences.')
        self._labeled_df = pandas_gbq.read_gbq(sql, project_id=google_tools.PROJECT_ID, dialect = 'standard', use_bqstorage_api=use_bqstorage_api)#, progress_bar_type=utils.PROGRESSBAR_TYPE)
        self._labeled_df.columns = [_.upper() for _ in self._labeled_df.columns]

        if not self._dump_all:
            mask = None
            for _ in self._labeled_df.columns:
                if _.lower() in ['subject_id', 'row_id', 'hadm_id', 'chartdate', 'category', 'text']:
                    continue
                if mask is None:
                    mask = self._labeled_df[_].astype(bool)
                else:
                    mask = mask | self._labeled_df[_].astype(bool)
            self._labeled_df = self._labeled_df[mask].copy()
            

        logger.log_info('DONE: Querying noteevents for lexicon occurences.')
        logger.log_debug('Number of admissions {}, number of notes {}.'.format(self._labeled_df['HADM_ID'].nunique(),len(self._labeled_df)))
        for _key in self._lexicon_map['positive'].keys():
            _key = _key.upper()
            logger.log_debug('Number of notes with {}: {}.'.format(_key.lower(), self._labeled_df[_key.upper()].sum()))


    @classmethod
    def register_argparser_object(cls, subparser_instance):
        subparser_instance.add_parser(cls.module_name(), help="Queries notes in MIMIC III based on the specified lexicon file dir.")
    
