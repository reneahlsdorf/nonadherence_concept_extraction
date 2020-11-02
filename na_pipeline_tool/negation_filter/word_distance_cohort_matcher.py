"""
Medical lexicon NLP extraction pipeline

File contains: Module which checks notes with found evidence for other lexicon terms and changes the cohort mapping if necessary
Based on the diet and drug matching in Joys original extraction notebook

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

import re
import nltk
import pandas as pd
from joblib import Parallel, delayed  
import sys, os
import collections
import numpy as np 

from na_pipeline_tool.utils import logger
from na_pipeline_tool.utils import config
from na_pipeline_tool.utils import helper_classes
from na_pipeline_tool import utils 
from na_pipeline_tool.utils import progressbar


class WordDistanceCohortMatcherModule(helper_classes.Module):

    def __init__(self):
        super().__init__()

        self._output_note_file = config.get_pipeline_config_item(self.module_name(), 'output_note_file', '_negex_filtered_notes.parquet')
        self._df_notes_labeled_path = config.get_pipeline_config_item(self.module_name(), 'input_note_file', None)
        self._word_distance = config.get_pipeline_config_item(self.module_name(), 'word_distance', 10)
        self._debug_check = config.get_pipeline_config_item(self.module_name(), 'debug_check', True)
        #self._has_negation_flag = False

        self._loaded_df = None

        self._lexicon_dir = config.get_pipeline_config_item(self.module_name(), 'word_filter_dir', './word_cohort_filters')
        self._lexicon_map = {}

        self._njobs = config.get('njobs', multiprocessing.cpu_count())

        logger.log_info('Loading note file')
        self._load_note_input_file()
        logger.log_info('DONE: Loading note file.')

        logger.log_info('Parsing word filter lexicons.')
        self._load_word_matching_lexicons()
        logger.log_info('DONE: Parsing word filter lexicons.')

        logger.log_info('Word filtering and cohort adaptions.')
        self._change_cohort_mappings()
        logger.log_info('DONE: Word filtering and cohort adaptions.')

        logger.log_info('Dumping changed notes.')
        self._dump_filtered_df()
        logger.log_info('DONE: Dumping changed notes.')


    def _dump_filtered_df(self):
        filename = utils.default_dataframe_name(self._output_note_file)
        if self._loaded_df.empty:
            logger.log_warn('There are no more entries left after filtering the dataframe.')
        else:
            self._loaded_df.to_parquet(filename)


    def _load_note_input_file(self):
        if not self._df_notes_labeled_path:
            raise RuntimeError('Please specify a valid note input file.')
        
        filename = utils.default_dataframe_name(self._df_notes_labeled_path)

        assert os.path.isfile(filename), 'Could not find note parquet file: {}'.format(filename)
        self._loaded_df = pd.read_parquet(filename)
        self._loaded_df.columns = [_.upper() for _ in self._loaded_df.columns]
        assert 'PREDICTED_CATEGORIES' in self._loaded_df.columns and 'SUBJECT_ID' in self._loaded_df.columns and 'CHARTDATE' in self._loaded_df.columns and 'CATEGORY' in self._loaded_df.columns and 'TEXT' in self._loaded_df.columns and 'ROW_ID' in self._loaded_df.columns and 'HADM_ID' in self._loaded_df.columns, 'Notes file need to have columns: Row_id, Subject_id, Hadm_id, predicted_categories, chartdate, category and text'
        #if 'FOUND_EVIDENCE' in self._loaded_df.columns:
        #    self._has_negation_flag = True
            

    def _load_word_matching_lexicons(self):
        def read_file(path):
            with open(path, 'r') as file:
                terms = file.readlines()
                terms = [_.strip().lower() for _ in terms]
            return terms
        
        assert self._lexicon_dir and os.path.isdir(self._lexicon_dir), "Lexicon dir needs to be valid and exist."
        files = os.listdir(self._lexicon_dir)
        nonallowed_categories = [_ for _ in files if not _.upper() in self._loaded_df.columns]
        assert len(nonallowed_categories) == 0, 'Following files do not match existing cohorts: {}'.format(nonallowed_categories)

        for _ in files:
            search_term_dir = os.path.join(self._lexicon_dir, _)
            if not os.path.isdir(search_term_dir):
                continue
            search_type = _.upper()
            assert search_type in self._loaded_df.columns, 'Search cohort type {} does not exist in dataframe.'.format(search_type)
            _files = os.listdir(search_term_dir)
            files_path = [os.path.join(self._lexicon_dir,_, __) for __ in _files]
            _files = [re.sub(r'\..*', '', _) for _ in _files]
            unknown_types = [_ for _ in _files if not _.upper() in self._loaded_df.columns]
            assert search_type in self._loaded_df.columns, 'Lexicon files {} do not match cohort in dataframe.'.format(unknown_types)
            self._lexicon_map[search_type] = {}
            for _filepath, _file in zip(files_path, _files):
                _file = _file.upper()
                terms = read_file(_filepath)
                self._lexicon_map[search_type][_file] = terms


    def _process_note(self, notes, _search_type, _lexicons):
        ready_notes = []
        _i = 0

        for _, note in notes.iterrows():
            _i += 1
            if _i % 1000 == 0:
                print('Processed 1000 notes.')

            # if self._has_negation_flag and not note['FOUND_EVIDENCE']:
            #     ready_notes.append(list(note))
            #     continue

            def reSpotter(text, lexicons):
                for term in lexicons:
                    mre = r'\b' + re.escape(term) + r'\b'
                    if (re.search(mre, text, flags=re.IGNORECASE) != None):
                        return term
                return False

            preds = note['PREDICTED_CATEGORIES'].upper()
            if not _search_type in preds:
                ready_notes.append(list(note))
                continue

            pos_sentences = note[_search_type + '_POSITIVE_LEXICON_SENTENCES'].split('\n')
            negations = note[_search_type + '_POSITIVE_LEXICON_NEGATED_PHRASES'].split('\n')
            affirms = note[_search_type + '_POSITIVE_LEXICON_AFFIRMED_PHRASES'].split('\n')
            
            for _sent, _negs, _affirms in zip(pos_sentences, negations, affirms):
                if _negs != '[]' and _negs != '':
                    continue
                affirms = re.findall(r"'(.*?)'", _affirms)
                for _aff in affirms:
                    lh, _, rh = _sent.lower().partition(_aff.lower())
                    n = self._word_distance
                    span = ' '.join(lh.split()[-n:]+[_aff]+rh.split()[:n]) 
                    for _coh, _lex in _lexicons.items():
                        _coh = _coh.upper()
                        matched = reSpotter(span,_lex)
                        if matched != False:
                            if not _coh in preds:
                                if _search_type in preds:
                                    preds = preds.replace(_search_type, _coh)
                                else:
                                    preds = '|'.join(preds.split('|') + [_coh]) 
                                note['PREDICTED_CATEGORIES'] = preds
                                break
                            elif _search_type in preds:
                                preds = preds.replace(_search_type + '|', '')
                                preds = preds.replace('|' + _search_type, '')
                                preds = preds.replace(_search_type, '')
                                note['PREDICTED_CATEGORIES'] = preds
                                break

            ready_notes.append(list(note))
        return ready_notes


    def _change_cohort_mappings(self):
        for _search_type, _lexicons in self._lexicon_map.items():
            assert _search_type + '_POSITIVE_LEXICON_SENTENCES' in self._loaded_df.columns, "Missing column in dataframe: {}. This module only suports inputs from the negex_negation_filter module.".format(_search_type + '_POSITIVE_LEXICON_SENTENCES')
            assert _search_type + '_POSITIVE_LEXICON_NEGATED_PHRASES' in self._loaded_df.columns, "Missing column in dataframe: {}. This module only suports inputs from the negex_negation_filter module.".format(_search_type + '_POSITIVE_LEXICON_NEGATED_PHRASES')
            assert _search_type + '_POSITIVE_LEXICON_AFFIRMED_PHRASES' in self._loaded_df.columns, "Missing column in dataframe: {}. This module only suports inputs from the negex_negation_filter module.".format(_search_type + '_POSITIVE_LEXICON_AFFIRMED_PHRASES')

            if self._debug_check:
                self._loaded_df = self._loaded_df.iloc[:100]

            logger.log_info('Long dist. matching for: ' + str(_lexicons.keys()))

            notes_parsed = Parallel(n_jobs=self._njobs)(delayed(self._process_note)(note, _search_type, _lexicons) for note in ([self._loaded_df.iloc[_:min(_+7000, len(self._loaded_df)), :] for _ in range(0, len(self._loaded_df), 7000)]))
            notes_parsed = [__ for _ in notes_parsed for __ in _]
            self._loaded_df = pd.DataFrame(notes_parsed, columns=self._loaded_df.columns)


    @classmethod
    def register_argparser_object(cls, subparser_instance):
        subparser_instance.add_parser(cls.module_name(), help="Tries to find special words around affirmed phrased extracted during the negation phase. If found, changes the cohort to the one belonging to that special word (e.g. finding drug words in the NOS Nonadherence cohort.)")
    
