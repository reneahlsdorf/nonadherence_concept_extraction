"""
Medical lexicon NLP extraction pipeline

File contains: Compared two predicted files with the validation sets. Outputs example sentences identified correctly by one prediction file not classified correctly by the other one and vice versa.
Used for comparing the performance of different pipeline settings/methods.

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

from na_pipeline_tool.utils import logger
from na_pipeline_tool.utils import config
from na_pipeline_tool.utils import helper_classes
from na_pipeline_tool import utils 

from na_pipeline_tool.utils import progressbar

import re
import pandas as pd
from joblib import Parallel, delayed  

import sys, os
import collections
import numpy as np 
import openpyxl

import sklearn.metrics

class ValidationComparePipelineResults(helper_classes.Module):

    def __init__(self):
        super().__init__()

        self._validation_set = config.get_pipeline_config_item(self.module_name(), 'validation_set_file', None)
        self._df_path_a = config.get_pipeline_config_item(self.module_name(), 'file_prediction_a', None)
        self._df_path_b = config.get_pipeline_config_item(self.module_name(), 'file_prediction_b', None)
        self._ignore_b = config.get_pipeline_config_item(self.module_name(), 'ignore_b', False)
        self._section_info__file = config.get_pipeline_config_item(self.module_name(), 'section_info__file', None)

        self._df_path_a_negated = config.get_pipeline_config_item(self.module_name(), 'file_sentence_info_a', None)
        self._df_path_b_negated = config.get_pipeline_config_item(self.module_name(), 'file_sentence_info_b', None)
        self._get_examples_for_categories = config.get_pipeline_config_item(self.module_name(), 'get_examples_for_categories', None)
        self._compare_dumping_dir_name = 'dumped_validaton_sentences_between__{}_and_{}'

        self._required_tag_list = ['POSITIVE_LEXICON_SENTENCES', 'POSITIVE_LEXICON_AFFIRMED_PHRASES', 'POSITIVE_LEXICON_NEGATED_PHRASES', 'NEGATIVE_LEXICON_SENTENCES', 'NEGATIVE_LEXICON_AFFIRMED_PHRASES', 'NEGATIVE_LEXICON_NEGATED_PHRASES']

        self._df_a = None
        self._df_b = None
        self._df_sents_a = None
        self._df_sents_b = None

        self._loaded_validation = None
        self._loaded_validation_labels = None
        self._loaded_validation_label_map = None

        logger.log_info('Processing validation file')
        self._loading_validation_labeling_file()
        logger.log_info('DONE: Processing validation file')

        if self._section_info__file:
            logger.log_info('Loading section file to include section information')
            self._loading_section_info_file()
            logger.log_info('DONE: Loading section file to include section information')

        logger.log_info('Loading prediction and sentence info file for dataframes A and B to be compared.')
        self._loading_note_files()
        logger.log_info('DONE: Loading prediction and sentence info file for dataframes A and B to be compared.')

        logger.log_info('Dumping sentences into folder')
        self._dump_examples_for_comparison()
        logger.log_info('DONE: Dumping sentences into folder')


    def _loading_section_info_file(self):
        filename = utils.default_dataframe_name(self._section_info__file)
        assert os.path.isfile(filename), 'Could not find section parquet file: {}'.format(filename)
        df = pd.read_parquet(filename)
        df.columns = [_.upper() for _ in df.columns]

        assert 'SECTION_GROUP' in df.columns, 'Section file need to have columns: Section_id, section_group, row_id, text'
        assert 'ROW_ID' in df.columns, 'Section file need to have columns: Section_id, section_group, row_id, text'
        assert 'SECTION_ID' in df.columns, 'Section file need to have columns: Section_id, section_group, row_id, text'
        assert 'TEXT' in df.columns, 'Section file need to have columns: Section_id, section_group, row_id, text'
       
        self._section_info__file = df


    def _loading_validation_labeling_file(self):
        assert self._validation_set, 'Please specify a validation labeling file.'
        try:
            with open(self._validation_set, 'r') as file:
                self._loaded_validation = file.readlines()
                self._loaded_validation = self._loaded_validation[1:]
                self._loaded_validation = [_.strip() for _ in self._loaded_validation]
                self._loaded_validation = [_.split(',') for _ in self._loaded_validation]
                self._loaded_validation = [[int(_[0]), [_.upper().replace(' ', '_') for _ in str(_[1]).split('|')], (int(_[2]) > 0)] for _ in self._loaded_validation]
                self._loaded_validation = pd.DataFrame(self._loaded_validation, columns=['ROW_ID', 'NOTE_TYPES', 'VALID_INCLUDED'])
                self._loaded_validation.loc[~self._loaded_validation['VALID_INCLUDED'], 'NOTE_TYPES'] = pd.Series([['NONE']]*self._loaded_validation.shape[0])
        except:
            raise RuntimeError('Error while processing validation labeling file. Check file structure.')
        self._loaded_validation_labels = []
        for _i, _ in self._loaded_validation.iterrows():
            self._loaded_validation_labels.extend(_['NOTE_TYPES'])

        self._loaded_validation_labels = set(self._loaded_validation_labels)


    def _loading_note_files(self):
        if not self._df_path_a or not self._df_path_b:
            raise RuntimeError('Please specify a valid note input file.')
        
        def load_prediction_file(path):
            filename = utils.default_dataframe_name(path)
            assert os.path.isfile(filename), 'Could not find note parquet file: {}'.format(filename)
            df = pd.read_parquet(filename)
            df.columns = [_.upper() for _ in df.columns]

            assert 'ROW_ID' in df.columns, 'Notes file need to have columns: Row_id, predicted_categories'
            assert 'PREDICTED_CATEGORIES' in df.columns, "Processed note file needs to have the PREDICTED_CATEGORIES column generated by e.g. the negation module."
            df['PREDICTED_CATEGORIES'] = df.PREDICTED_CATEGORIES.str.upper()
            df['PREDICTED_CATEGORIES'] = df.PREDICTED_CATEGORIES.str.replace(' ', '_')
            df['PREDICTED_CATEGORIES'] = df.PREDICTED_CATEGORIES.str.split('|')
            if 'FOUND_EVIDENCE' in df.columns:
                df['FOUND_EVIDENCE'] = df['FOUND_EVIDENCE'].astype(bool)
                df = df[df['FOUND_EVIDENCE']]
            
            return df

        def load_sentence_info_file(path, allowed_class_labels):
            filename = utils.default_dataframe_name(path)
            assert os.path.isfile(filename), 'Could not find note parquet file: {}'.format(filename)
            df = pd.read_parquet(filename)
            df.columns = [_.upper() for _ in df.columns]

            assert 'ROW_ID' in df.columns, 'Notes file need to have columns: Row_id, predicted_categories'
            for _ in allowed_class_labels:
                assert _ in df.columns, "Processed note file has no {} column - class label not found!".format(_)

                for __ in self._required_tag_list:
                    assert _ + '_' + __ in df.columns, "Processed note file has no {} column - the file needs to be generated by the negex_negation_filter module!".format(_ + '_' + __)
            
            return df

        self._df_a = load_prediction_file(self._df_path_a)
        self._df_b = load_prediction_file(self._df_path_b)

        # Identify and map all class labels to integer numbers
        unique_labels = []
        for _ in [*self._df_a.PREDICTED_CATEGORIES, *self._df_b.PREDICTED_CATEGORIES, self._loaded_validation_labels]:
            unique_labels.extend(_)

        unique_labels = set(unique_labels)
        unique_labels = set([_.upper() for _ in unique_labels])
        unique_labels_unmatched = unique_labels - self._loaded_validation_labels

        logger.log_info('Found the following labels which are present in the predicted notes but not in the validation set: ' + str(unique_labels_unmatched))
        lbl_id = 1
        self._loaded_validation_label_map = { 'NONE' : 0 }
        self._inv_loaded_validation_label_map = { 0 : 'NONE' }

        for _lbl in unique_labels:
            self._loaded_validation_label_map[_lbl] = lbl_id
            self._inv_loaded_validation_label_map[lbl_id] = _lbl
            lbl_id += 1

        for _lbl in unique_labels_unmatched:
            self._loaded_validation_label_map[_lbl] = 0
            self._inv_loaded_validation_label_map[0] = _lbl

        logger.log_info('Label string to int map: {}'.format(str(self._loaded_validation_label_map)))

        class_labels = [_ for _ in self._loaded_validation_label_map.keys() if _ != 'NONE']

        self._df_sents_a = load_sentence_info_file(self._df_path_a_negated, class_labels)
        self._df_sents_b = load_sentence_info_file(self._df_path_b_negated, class_labels)

        self._df_a['PREDICTED_CATEGORIES'] = self._df_a.PREDICTED_CATEGORIES.apply(lambda x: [self._loaded_validation_label_map[_] for _ in x])
        self._df_b['PREDICTED_CATEGORIES'] = self._df_b.PREDICTED_CATEGORIES.apply(lambda x: [self._loaded_validation_label_map[_] for _ in x])
        
        self._loaded_validation['NOTE_TYPES'] = self._loaded_validation.NOTE_TYPES.apply(lambda x: [self._loaded_validation_label_map[_] for _ in x])

        if not self._get_examples_for_categories:
            self._get_examples_for_categories = [*class_labels, 'NONE']
        else:
            self._get_examples_for_categories = [_.upper() for _ in self._get_examples_for_categories]
        self._get_examples_for_categories = [_ for _ in self._get_examples_for_categories if _ != 'NONE']

        logger.log_info('Dumping the following class labels of interest: {}'.format(str(self._get_examples_for_categories)))



    def _dump_examples_for_comparison(self):
        def generate_validation_results(df, class_tags):

            validset = self._loaded_validation.sort_values('ROW_ID').reset_index(drop=True)[['ROW_ID', 'NOTE_TYPES']].copy()
            validset = validset.drop_duplicates(subset=['ROW_ID'])

            predicted = df[['ROW_ID', 'PREDICTED_CATEGORIES']].copy()
            predicted = predicted.rename(columns={'PREDICTED_CATEGORIES' : 'PREDICTED_CAT'})
            predicted = predicted.drop_duplicates(subset=['ROW_ID'])
            
            validset = validset.merge(predicted, how='left', on='ROW_ID')
            validset.loc[validset['PREDICTED_CAT'].isnull(), 'PREDICTED_CAT'] = pd.Series([[0]]*validset.shape[0])
            validset.loc[validset['NOTE_TYPES'].isnull(), 'NOTE_TYPES'] = pd.Series([[0]]*validset.shape[0])
            
            validset['MATCHED'] = validset.apply(lambda x: [_ for _ in x.NOTE_TYPES if _ in x.PREDICTED_CAT], axis=1)
            validset['UNMATCHED_VALID'] = validset.apply(lambda x: [_ for _ in x.NOTE_TYPES if _ not in x.PREDICTED_CAT], axis=1)
            validset['UNMATCHED_PREDICTED']= validset.apply(lambda x: [_ for _ in x.PREDICTED_CAT if _ not in x.NOTE_TYPES], axis=1)

            
            for _class_tag in class_tags:
                classid = self._loaded_validation_label_map[_class_tag]
                validset['CORRECT_NOTE_' + _class_tag.upper()] = ~(validset['UNMATCHED_VALID'].apply(lambda x: classid in x) | validset['UNMATCHED_PREDICTED'].apply(lambda x: classid in x))
            
            validset['CORRECT_NOTE'] = False

            validset.loc[(validset.UNMATCHED_VALID.str.len() == 0) & (validset.UNMATCHED_PREDICTED.str.len() == 0), 'CORRECT_NOTE'] = True

            return validset

        matrix_A = generate_validation_results(self._df_a, self._get_examples_for_categories)
        matrix_B = generate_validation_results(self._df_b, self._get_examples_for_categories)

        self._compare_dumping_dir_name = self._compare_dumping_dir_name.format(self._df_path_a, self._df_path_b)
        if os.path.isdir(self._compare_dumping_dir_name):
            import shutil
            shutil.rmtree(self._compare_dumping_dir_name)
        os.mkdir(self._compare_dumping_dir_name)


        def export_df(path, a_right, b_right,ids, class_cat):
            if not self._ignore_b:
                filename = 'A_{}_B_{}.xlsx'.format('true' if a_right else 'false', 'true' if b_right else 'false')
            else:
                filename = 'A_{}.xlsx'.format('true' if a_right else 'false')

            filename = os.path.join(path, filename)
            
            predicted_entries = self._loaded_validation[['ROW_ID', 'NOTE_TYPES']]
            a_pred = self._df_a[['ROW_ID', 'PREDICTED_CATEGORIES']]
            b_pred = self._df_b[['ROW_ID', 'PREDICTED_CATEGORIES']]

            entries_a = self._df_sents_a[self._df_sents_a.ROW_ID.isin(ids)].copy()
            entries_b = self._df_sents_b[self._df_sents_b.ROW_ID.isin(ids)].copy()

            class_cols = [class_cat.upper() + '_' + _ for _ in self._required_tag_list]
            sentence_cols = [_ for _ in class_cols if 'SENTENCES' in _]

            #if 'SECTION_GROUP_NEW' in entries_a.columns:
            #    entries_a = entries_a[['ROW_ID', 'HADM_ID', 'SECTION_GROUP', 'SECTION_GROUP_NEW', *class_cols]]
            #else:

            entries_a = entries_a[['ROW_ID', 'HADM_ID', *class_cols]]

            #if 'SECTION_GROUP_NEW' in entries_b.columns:
            #    entries_b = entries_b[['ROW_ID', 'HADM_ID', 'SECTION_GROUP', 'SECTION_GROUP_NEW', *class_cols]]
            #else:
            entries_b = entries_b[['ROW_ID', 'HADM_ID', *class_cols]]


            if not self._section_info__file is None:

                for _ in (sentence_cols):
                    sections = []
                    sections_gr = []
                    for _idx, _ent in entries_a.iterrows():
                        sentences = str(_ent[_]).split('\n')
                        subsects = []
                        subsects_gr = []
                        sub_df = self._section_info__file[self._section_info__file.ROW_ID == _ent.ROW_ID].copy()
                        sub_df.loc[:, 'TEXT'] = sub_df.TEXT.str.replace('\n', ' ')

                        for _sent in sentences:
                            if not _sent:
                                continue
                            _sent = re.match(r'^(?:\[\'){0,1}(.*?)(?:\'\]){0,1}$', _sent).group(1)

                            try:
                                subsel = sub_df[sub_df.TEXT.str.find(_sent.replace('\n',' ')) > -1].iloc[0]
                            except:
                                continue
                            subsects.append(str(subsel.SECTION_NAME))
                            subsects_gr.append(str(subsel.SECTION_GROUP))
                        sections.append(subsects)
                        sections_gr.append(subsects_gr)

                    entries_a['SECTIONS_GROUP_' + _] = [str('\n'.join(_)) for _ in sections_gr]
                    entries_a['SECTION_TITLE_' + _] = [str('\n'.join(_)) for _ in sections]

           
            if not self._ignore_b:
                entries_a = entries_a.merge(entries_b, on=['ROW_ID', 'HADM_ID'], how='outer', suffixes=('__A', '__B'))
            entries_a = entries_a.merge(predicted_entries, on=['ROW_ID'], how='left')
            entries_a = entries_a.merge(a_pred, on=['ROW_ID'], how='left')
            if not self._ignore_b:
                entries_a = entries_a.merge(b_pred, on=['ROW_ID'], how='left', suffixes=('__A', '__B'))

            entries_a = entries_a.rename(columns={'NOTE_TYPES' : 'REAL_LABEL'})
            
            if self._ignore_b:
                entries_a.loc[~entries_a['PREDICTED_CATEGORIES'].isnull(), 'PREDICTED_CATEGORIES'] = entries_a.loc[~entries_a['PREDICTED_CATEGORIES'].isnull(), 'PREDICTED_CATEGORIES'].apply(lambda x: [self._inv_loaded_validation_label_map[_] for _ in x])
            else:
                entries_a.loc[~entries_a['PREDICTED_CATEGORIES__A'].isnull(), 'PREDICTED_CATEGORIES__A'] = entries_a.loc[~entries_a['PREDICTED_CATEGORIES__A'].isnull(), 'PREDICTED_CATEGORIES__A'].apply(lambda x: [self._inv_loaded_validation_label_map[_] for _ in x])
                entries_a.loc[~entries_a['PREDICTED_CATEGORIES__B'].isnull(), 'PREDICTED_CATEGORIES__B'] = entries_a.loc[~entries_a['PREDICTED_CATEGORIES__B'].isnull(), 'PREDICTED_CATEGORIES__B'].apply(lambda x: [self._inv_loaded_validation_label_map[_] for _ in x])

            entries_a.loc[~entries_a['REAL_LABEL'].isnull(), 'REAL_LABEL'] = entries_a.loc[~entries_a['REAL_LABEL'].isnull(), 'REAL_LABEL'].apply(lambda x: [self._inv_loaded_validation_label_map[_] for _ in x])

            entries_a.to_excel(filename)


        for _ in self._get_examples_for_categories:
            entries = []
            os.mkdir(os.path.join(self._compare_dumping_dir_name, _.lower()))

            if not self._ignore_b:
                # Examples for A right, B false
                rows_a_right = matrix_A[matrix_A['CORRECT_NOTE_' + str(_)]].ROW_ID.unique()
                rows_b_false = matrix_B[~matrix_B['CORRECT_NOTE_' + str(_)]].ROW_ID.unique()
                a_right_b_false = set(rows_a_right).intersection(set(rows_b_false))

            if not self._ignore_b:
                # Examples for A false, B right
                rows_a_false = matrix_A[~matrix_A['CORRECT_NOTE_' + str(_)]].ROW_ID.unique()
                rows_b_right = matrix_B[matrix_B['CORRECT_NOTE_' + str(_)]].ROW_ID.unique()
                a_false_b_right = set(rows_b_right).intersection(set(rows_a_false))

            # Examples for A right, B right
            rows_a_right = matrix_A[matrix_A['CORRECT_NOTE_' + str(_)]].ROW_ID.unique()
            if not self._ignore_b:
                rows_b_right = matrix_B[matrix_B['CORRECT_NOTE_' + str(_)]].ROW_ID.unique()
                a_right_b_right = set(rows_b_right).intersection(set(rows_a_right))
            else:
                a_right_b_right = set(rows_a_right)

            # Examples for A false, B false
            rows_a_false = matrix_A[~matrix_A['CORRECT_NOTE_' + str(_)]].ROW_ID.unique()
            if not self._ignore_b:
                rows_b_false = matrix_B[~matrix_B['CORRECT_NOTE_' + str(_)]].ROW_ID.unique()
                a_false_b_false = set(rows_b_false).intersection(set(rows_a_false))
            else:
                a_false_b_false = set(rows_a_false)
            

            ## Exporting the sentences
            if not self._ignore_b:
                export_df(os.path.join(self._compare_dumping_dir_name, _.lower()),True,False,a_right_b_false,_.upper())
                export_df(os.path.join(self._compare_dumping_dir_name, _.lower()),False,True,a_false_b_right,_.upper())

            export_df(os.path.join(self._compare_dumping_dir_name, _.lower()),True,True,a_right_b_right,_.upper())
            export_df(os.path.join(self._compare_dumping_dir_name, _.lower()),False,False,a_false_b_false,_.upper())

            
    @classmethod
    def register_argparser_object(cls, subparser_instance):
        subparser_instance.add_parser(cls.module_name(), help="Compares to predicted not sources with a validation set and export their differences (sentences) in different CSV files.")
    
