"""
Medical lexicon NLP extraction pipeline

File contains: Contains the module for filtering negated sentences by applying the Negex utility 

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
from joblib import Parallel, delayed  
import multiprocessing
import sys, os
import collections
import numpy as np 

from na_pipeline_tool.utils import logger
from na_pipeline_tool.utils import config
from na_pipeline_tool.utils import helper_classes
from na_pipeline_tool import utils 
from na_pipeline_tool.utils import progressbar
from na_pipeline_tool.libs import custom_negex

class NegexNegationFilterModule(helper_classes.Module):

    def __init__(self):
        super().__init__()
        nltk.download('punkt')

        self._output_note_file = config.get_pipeline_config_item(self.module_name(), 'output_note_file', '_negex_filtered_notes.parquet')
        self._df_notes_labeled_path = config.get_pipeline_config_item(self.module_name(), 'input_note_file', None)
        self._negex_triggers = config.get_pipeline_config_item(self.module_name(), 'negex_triggers', 'negex_trigger.txt')
        self._use_old_negation_scheme = config.get_pipeline_config_item(self.module_name(), 'use_old_negation_scheme', False)
        self._use_only_positive_lexicons = config.get_pipeline_config_item(self.module_name(), 'use_only_positive_lexicons', False)
        self._dump_all_notes = config.get_pipeline_config_item(self.module_name(), 'dump_all_matched_notes', False)
        self._add_fullstop_after_newline_uppercase = config.get_pipeline_config_item(self.module_name(), 'add_fullstop_after_newline_uppercase', False)
        self._keep_categories = config.get_pipeline_config_item(self.module_name(), 'keep_categories', [])
        self._dont_include_predicted_categories = config.get_pipeline_config_item(self.module_name(), 'dont_include_predicted_categories', False)
        self._debug_row_id = config.get_pipeline_config_item(self.module_name(), 'debug_row_id', None)

        self._loaded_df = None
        self._pre_filtered_df = None
        self._filtered_cohort_df = None
        self._labeled_df = None

        self._lexicon_dir = config.get_pipeline_config_item(self.module_name(), 'lexicon_dir', './used_lexicons')
        self._lexicon_map = { 'positive' : {}, 'negative' : {}}
        self._lexicon_weights = {'positive' : {}, 'negative' : {}}
        self._debug_check = config.get_pipeline_config_item(self.module_name(), 'debug_check', False)

        self._njobs = config.get('njobs', multiprocessing.cpu_count())

        logger.log_info('Loading note file')
        self._load_note_input_file()
        logger.log_info('DONE: Loading note file.')

        logger.log_info('Parsing Negex triggers.')
        self._load_negex_triggers()
        logger.log_info('DONE: Parsing Negex triggers.')

        logger.log_info('Negex note filtering.')
        self._parse_lexicons()
        self._check_note_negations()
        logger.log_info('DONE: Negex note filtering.')

        logger.log_info('New cohort labeling.')
        self._label_improve_cohort()
        logger.log_info('DONE: New cohort labeling.')

        logger.log_info('Dumping filtered notes.')
        self._dump_filtered_df()
        logger.log_info('DONE: Dumping filtered notes.')

    def _load_negex_triggers(self):
        assert os.path.isfile(self._negex_triggers), 'Cannot find negex trigger file'
        with open(self._negex_triggers, 'r') as file:
            triggers = file.readlines()
            self._negex_triggers = custom_negex.sortRules(triggers)

    def _parse_lexicons(self):
        assert os.path.isdir(self._lexicon_dir), 'Invalid lexicon dir. Does not exist.'
        assert len(os.listdir(self._lexicon_dir)) > 0, 'Lexicon dir is empty.'
        pos_dir = os.path.join(self._lexicon_dir, 'positive')
        neg_dir = os.path.join(self._lexicon_dir, 'negative')
        assert os.path.isdir(pos_dir), 'There needs to be a positive lexicon. If you just want to use one variant of the lexicons, create a positive lexicon folders with empty text files.'
        assert os.path.isdir(neg_dir), 'There needs to be a negative lexicon. If you just want to use one variant of the lexicons, create a negative lexicon folders with empty text files.'

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
                        if len(term[0]) < 1:
                            continue
                        assert len(term) > 0, 'Invalid line found in {} lexicon: {}'.format(prefix ,_)
                        if len(term) < 2:
                            self._lexicon_map[prefix][filename].append(term[0])
                            self._lexicon_weights[prefix][filename][term[0].lower()] = 2
                        else:
                            self._lexicon_map[prefix][filename].append(term[0])
                            self._lexicon_weights[prefix][filename][term[0].lower()] = (int(term[1]))
                file.close()
        
        logger.log_debug('Parsing the lexicons..')
        parse_dir(pos_files, 'positive')
        parse_dir(neg_files, 'negative')

        for cat in ['positive', 'negative']:
            for _k, _v in self._lexicon_map[cat].items():
                logger.log_info('{} {} lexicon: {} entries'.format(cat, _k, len(_v)))
        
        logger.log_info('Parsed and stored all lexicons.')

    def _process_note(self, note):
        def reSpotter(text, lexicons, oldscheme=False):
            term_list = '|'.join([r'\b' + re.escape(_) + r'\b' for _ in lexicons if len(_) > 0])
            all_findings = re.findall(term_list, text, flags=re.IGNORECASE)
            if all_findings:
                if oldscheme:
                    return [all_findings[0]]
                return all_findings
            return False

        categories = [_ for _ in note.index if not _.lower() in ['subject_id', 'row_id', 'hadm_id', 'chartdate', 'category', 'text', 'section_id', 'section_group', 'section_group_new', 'section_name']]
        id = note['ROW_ID']
        note = note['TEXT']

        negation_info_dict = {
            _.upper() : { 'positive' : [[], [], []], 'negative' : [[],[], []]} for _ in categories
        }
        negation_info_dict['ROW_ID'] = id


        if self._add_fullstop_after_newline_uppercase:
            _note = re.sub(r'(.(?!\.))\s*\n\s*\b([A-Z0-9].*?)', r'\1 .\n\n\2', note)
            sents = sent_tokenize(_note.replace('\n',' '))
        else:
            sents = sent_tokenize(note.replace('\n',' '))

        param_list = [['positive', 1], ['negative', -1]]
        total_score_sum = collections.defaultdict(int)
        score_cat = {'positive' : collections.defaultdict(int), 'negative': collections.defaultdict(int)}

        new_sents = sents
        for _params in param_list:
            cat, sign = _params
            sents = list(set(new_sents))
            new_sents = []

            if not sents:
                for _k, _v in self._lexicon_map[cat].items():
                    negation_info_dict['SCORE_SUM_' + cat.upper() + '_' + _k.upper()] = score_cat[cat][_k.upper()] 
                    negation_info_dict['TOTAL_SCORE_SUM_' + _k.upper()] = total_score_sum[_k.upper()]
                break
            
            for _k, _v in self._lexicon_map[cat].items():
                if not _v:
                    negation_info_dict['SCORE_SUM_' + cat.upper() + '_' + _k.upper()] = score_cat[cat][_k.upper()] 
                    negation_info_dict['TOTAL_SCORE_SUM_' + _k.upper()] = total_score_sum[_k.upper()]
                    continue

                score_cat[cat][_k.upper()] = 0

                num_neg, num_affirm = 0, 0
                for sent in sents:
                    
                    findings = reSpotter(sent, _v, oldscheme=self._use_old_negation_scheme)
                    
                    if not findings:
                        new_sents.append(sent)
                        continue
                
                    findings = list(set([_.lower() for _ in findings]))
                    findings = sorted(findings, key=lambda x: -1 * len(x.split(' ')))
                    tagger = custom_negex.negTagger(sentence = sent, phrases = findings, rules = self._negex_triggers, negP=False)
                    tag = tagger.getNegTaggedSentence()
                    
                    tag = tag.replace('\\', r'\\')

                    #negated_tags = re.sub(r'\[!NEGATED\]((?:(?!\[NEGATED\]).)*?\[!NEGATED\])', r'\1', tag)
                    #negated_tags = re.sub(r'(\[NEGATED\](?:(?!\[!NEGATED\]).)*?)\[NEGATED\]', r'\1', negated_tags)
                    negated_tags = re.findall(r'\[NEGATED\](.*?)\[!NEGATED\]', tag)
                    
                    negated_tags = [_.strip().lower() for _ in negated_tags]
                    affirmed_tags = [_.lower() for _ in findings if _.lower() not in negated_tags]
                    
                    num_neg += len(negated_tags)
                    num_affirm += len(affirmed_tags)

                    scores_negated = [self._lexicon_weights[cat][_k][_] for _ in negated_tags]
                    scores_affirmed = [self._lexicon_weights[cat][_k][_] + 1 for _ in affirmed_tags]

                    sentence_score = sum(scores_affirmed) - sum(scores_negated)
                    total_score_sum[_k.upper()] += sign * sentence_score
                    score_cat[cat][_k.upper()]  += sentence_score
                    negation_info_dict[_k.upper()][cat][0].append(sent)
                    negation_info_dict[_k.upper()][cat][1].append(affirmed_tags)
                    negation_info_dict[_k.upper()][cat][2].append(negated_tags)

                negation_info_dict['SCORE_SUM_' + cat.upper() + '_' + _k.upper()] = score_cat[cat][_k.upper()] 
                negation_info_dict['TOTAL_SCORE_SUM_' + _k.upper()] = total_score_sum[_k.upper()]

        return negation_info_dict

    def _load_note_input_file(self):
        if not self._df_notes_labeled_path:
            raise RuntimeError('Please specify a valid note input file.')
        
        filename = utils.default_dataframe_name(self._df_notes_labeled_path)

        assert os.path.isfile(filename), 'Could not find note parquet file: {}'.format(filename)
        self._loaded_df = pd.read_parquet(filename)

        #self._loaded_df = self._loaded_df[self._loaded_df.ROW_ID == 23191] # 3083

        self._loaded_df.columns = [_.upper() for _ in self._loaded_df.columns]
        assert 'SUBJECT_ID' in self._loaded_df.columns and 'CHARTDATE' in self._loaded_df.columns and 'CATEGORY' in self._loaded_df.columns and 'TEXT' in self._loaded_df.columns and 'ROW_ID' in self._loaded_df.columns and 'HADM_ID' in self._loaded_df.columns, 'Notes file needs to have columns: Row_id, Subject_id, Hadm_id, chartdate, category and text'
        logger.log_info('Notes before category removal: {}'.format(len(self._loaded_df))) 
        self._loaded_df['CATEGORY'] = self._loaded_df['CATEGORY'].str.lower()
        self._keep_categories = [_.lower() for _ in self._keep_categories]
        filtered_df = self._loaded_df[self._loaded_df['CATEGORY'].isin(self._keep_categories)].copy()
        del self._loaded_df
        self._loaded_df = filtered_df
        logger.log_info('Notes after category removal: {}'.format(len(self._loaded_df)))

        if not self._debug_row_id is None:
            self._loaded_df = self._loaded_df[self._loaded_df.ROW_ID == self._debug_row_id]
            if self._loaded_df.empty:
                logger.log_error('Could not find requested debugging row id.')
    
    def _check_note_negations(self):
        mask = None
        for _ in self._loaded_df.columns:
            if _.lower() in ['subject_id', 'row_id', 'hadm_id', 'chartdate', 'category', 'text', 'section_id', 'section_group', 'section_group_new', 'section_name']:
                continue
            if mask is None:
                mask = self._loaded_df[_].astype(bool)
            else:
                mask = mask | self._loaded_df[_].astype(bool)

        logger.log_info('Starting negation checking loop')
        logger.log_debug(str(self._njobs) + ' processes used for check routine.')
        note_pos_df = self._loaded_df[mask].copy()

        logger.log_info('Total patients (before negex filtering): {} / Total admissions: {}'.format(note_pos_df['SUBJECT_ID'].nunique(), note_pos_df['HADM_ID'].nunique()))
        del self._loaded_df
        
        if self._debug_check:
            #note_pos_df = note_pos_df[note_pos_df['ROW_ID'] == 106]
            note_pos_df = note_pos_df.iloc[0:10]

        import gc
        gc.collect()

        note_infos = Parallel(n_jobs=self._njobs)(delayed(self._process_note)(note) for _, note in progressbar(note_pos_df.iterrows(), total=len(note_pos_df)))
        logger.log_debug('Found {} note infos.'.format(len(note_infos)))
        if note_infos:
            example_note = note_infos[0]
            logger.log_debug(str(example_note))        

        note_infos_df = []
        cols = ['ROW_ID']

        for _entry in note_infos:
            lis = [_entry['ROW_ID']]
            for _key in self._lexicon_map['positive'].keys():
                _key = _key.upper()
                positive_terms = 0
                negative_terms = 0
                positive_negations = 0
                negative_negations = 0

                for _pos_terms in _entry[_key]['positive'][1]:
                    positive_terms += len(_pos_terms)
                for _pos_terms in _entry[_key]['positive'][2]:
                    positive_terms += len(_pos_terms)
                    positive_negations += len(_pos_terms)
                for _neg_terms in _entry[_key]['negative'][1]:
                    negative_terms += len(_neg_terms)
                for _neg_terms in _entry[_key]['negative'][2]:
                    negative_terms += len(_neg_terms)
                    negative_negations += len(_neg_terms)
                
                lis.append(_entry['TOTAL_SCORE_SUM_' + _key.upper()])
                lis.append(_entry['SCORE_SUM_POSITIVE_' + _key.upper()])
                lis.append(_entry['SCORE_SUM_NEGATIVE_' + _key.upper()])

                lis.append(positive_terms)
                lis.append(positive_negations)
                lis.append(negative_terms)
                lis.append(negative_negations)

                lis.append('\n'.join(_entry[_key]['positive'][0]))
                lis.append('\n'.join([str(_) for _ in _entry[_key]['positive'][1]]))
                lis.append('\n'.join([str(_) for _ in _entry[_key]['positive'][2]]))

                lis.append('\n'.join(_entry[_key]['negative'][0]))
                lis.append('\n'.join([str(_) for _ in _entry[_key]['negative'][1]]))
                lis.append('\n'.join([str(_) for _ in _entry[_key]['negative'][2]]))

            note_infos_df.append(lis)
        
        cols_suffix = ['TOTAL_SCORE_SUM', 'SCORE_SUM_POSITIVE', 'SCORE_SUM_NEGATIVE', 'POSITIVE_LEXICON_OCCURENCES', 'POSITIVE_LEXICONS_NEGATED_OCCURENCES', 'NEGATIVE_LEXICON_OCCURENCES', 'NEGATIVE_LEXICONS_NEGATED_OCCURENCES', 'POSITIVE_LEXICON_SENTENCES', 'POSITIVE_LEXICON_AFFIRMED_PHRASES', 'POSITIVE_LEXICON_NEGATED_PHRASES', 'NEGATIVE_LEXICON_SENTENCES', 'NEGATIVE_LEXICON_AFFIRMED_PHRASES', 'NEGATIVE_LEXICON_NEGATED_PHRASES']
        for _key in self._lexicon_map['positive'].keys():
            for _suff in cols_suffix:
                cols.append(_key.upper() + '_' + _suff.upper())

        info_df = pd.DataFrame(note_infos_df, columns=cols)
        note_pos_df = note_pos_df.merge(info_df, how='left', on='ROW_ID')

        note_pos_df.loc[:, cols] = note_pos_df[cols].fillna('')
        self._pre_filtered_df = note_pos_df


    def _label_improve_cohort(self):
        self._pre_filtered_df['FOUND_EVIDENCE_NEGATED'] = 0
        if not self._dont_include_predicted_categories:
            self._pre_filtered_df['PREDICTED_CATEGORIES'] = ''

        self._pre_filtered_df['MAX_SCORE_CAT'] = ''
        self._pre_filtered_df['MAX_SCORE'] = -np.inf

        for _k in self._lexicon_map['positive'].keys():
            _k = _k.upper()  

            if self._use_only_positive_lexicons:
                self._pre_filtered_df[_k + '_TOTAL_SCORE_SUM'] = self._pre_filtered_df[_k + '_SCORE_SUM_POSITIVE']

            if self._use_old_negation_scheme:
                mask = (self._pre_filtered_df[_k + '_POSITIVE_LEXICONS_NEGATED_OCCURENCES'] == 1) & (self._pre_filtered_df[_k + '_POSITIVE_LEXICON_OCCURENCES'] == 1)
                self._pre_filtered_df[_k + '_TOTAL_SCORE_SUM'] = self._pre_filtered_df[_k + '_POSITIVE_LEXICON_OCCURENCES']
                self._pre_filtered_df.loc[mask,_k + '_TOTAL_SCORE_SUM'] = -1
                self._pre_filtered_df.loc[~mask,_k + '_TOTAL_SCORE_SUM'] = 1
                self._pre_filtered_df.loc[self._pre_filtered_df[_k + '_POSITIVE_LEXICON_OCCURENCES'] == 0, _k + '_TOTAL_SCORE_SUM'] = 0

            self._pre_filtered_df['MAX_SCORE_CAT'] = self._pre_filtered_df.apply(lambda x: _k if x[_k + '_TOTAL_SCORE_SUM'] > x['MAX_SCORE'] else x['MAX_SCORE_CAT'], axis=1)
            self._pre_filtered_df['MAX_SCORE'] = self._pre_filtered_df.apply(lambda x: x[_k + '_TOTAL_SCORE_SUM'] if x[_k + '_TOTAL_SCORE_SUM'] > x['MAX_SCORE'] else x['MAX_SCORE'], axis=1)
            if not self._dont_include_predicted_categories:
                self._pre_filtered_df['_PREDICTED_CATEGORIES'] = self._pre_filtered_df[_k + '_TOTAL_SCORE_SUM'].apply(lambda x: _k if x > 0 else '')                
                self._pre_filtered_df['PREDICTED_CATEGORIES'] = self._pre_filtered_df.apply(lambda x: x['PREDICTED_CATEGORIES'] + '|' + _k if len(x['_PREDICTED_CATEGORIES']) > 0 else x['PREDICTED_CATEGORIES'], axis=1)
                del self._pre_filtered_df['_PREDICTED_CATEGORIES']
        
        if not self._dont_include_predicted_categories:
            self._pre_filtered_df.loc[self._pre_filtered_df['MAX_SCORE'] > 0 , 'FOUND_EVIDENCE_NEGATED'] = 1
            if not self._dump_all_notes:
                self._filtered_cohort_df = self._pre_filtered_df[self._pre_filtered_df['FOUND_EVIDENCE_NEGATED'] == 1].copy()
            else:
                self._filtered_cohort_df = self._pre_filtered_df
        else:
            self._filtered_cohort_df = self._pre_filtered_df

        if not self._dont_include_predicted_categories:
            self._filtered_cohort_df['PREDICTED_CATEGORIES'] = self._filtered_cohort_df['PREDICTED_CATEGORIES'].apply(lambda x: x[1:] if len(x) > 1 else x)
        
            if 'FOUND_EVIDENCE' in self._filtered_cohort_df.columns:
                self._filtered_cohort_df['FOUND_EVIDENCE'] = (self._filtered_cohort_df['FOUND_EVIDENCE'] > 0) & (self._filtered_cohort_df['FOUND_EVIDENCE_NEGATED'] > 0)
            else:
                self._filtered_cohort_df['FOUND_EVIDENCE'] = self._filtered_cohort_df['FOUND_EVIDENCE_NEGATED']
            del self._filtered_cohort_df['FOUND_EVIDENCE_NEGATED']

        logger.log_info('Total patients (after negex filtering): {} / Total admissions: {}'.format(self._filtered_cohort_df['SUBJECT_ID'].nunique(), self._filtered_cohort_df['HADM_ID'].nunique()))

    def _dump_filtered_df(self):
        filename = utils.default_dataframe_name(self._output_note_file)
        if self._filtered_cohort_df.empty:
            logger.log_warn('There are no more entries left after filtering the dataframe.')
        else:
            self._filtered_cohort_df.to_parquet(filename)

    @classmethod
    def register_argparser_object(cls, subparser_instance):
        subparser_instance.add_parser(cls.module_name(), help="Filters the pre-labeled notes and removes negated instances from the final note dataset by using the Negex utility.")
    
