"""This module contains some simple helpers for the Google tool interaction, e.g. authenticating the user and querying BigQuery."""

"""
Medical lexicon NLP extraction pipeline

File contains: Helper methods for using google services 

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

import sys, os
import pandas as pd
import google.auth

from . import logger
from . import config
from .helper_classes import Module

GLOBAL_CREDS = None
PROJECT_ID = None
_BIGQUERY_CLIENT = None
    
def check_google_authenticated():
    """Checks if the user is authenticated.

    Returns:
        bool: Is user authenticated?
    """
    global GLOBAL_CREDS
    global PROJECT_ID
    try:
        GLOBAL_CREDS, PROJECT_ID = google.auth.default()
        assert GLOBAL_CREDS
        assert PROJECT_ID
    except:
        return False
    return True
    
def get_big_query_client():
    """Get the BigQuery client object (or create one if there is none)

    Returns:
        google.cloud.bigquery.Client: The BigQuery client
    """

    global _BIGQUERY_CLIENT
    if _BIGQUERY_CLIENT:
        return _BIGQUERY_CLIENT
    assert GLOBAL_CREDS, 'Authentification required for Google services.'
    from google.cloud import bigquery

    _BIGQUERY_CLIENT = bigquery.Client(project=PROJECT_ID, credentials=GLOBAL_CREDS)
    return _BIGQUERY_CLIENT


def query_bigquery_raw(sql_query):
    """Query Google BigQuery using the provided SQL query. Logs errors but does not throw any exceptions while accessing BigQuery.

    Args:
        sql_query (str): The SQL query string.

    Returns:
        google.cloud.bigquery.table.Row: The result rows
    """

    import google.cloud.exceptions
    client = get_big_query_client()
    assert client, 'Global bigquery client is not configured yet.'
    query_job = _BIGQUERY_CLIENT.query(sql_query)
    try: 
        rows = query_job.result()
    except google.cloud.exceptions.GoogleCloudError as e:
        logger.log_error('Exception during query: ' + str(e))
    return rows

def query_bigquery_df(sql_query):
    """Query Google BigQuery using the provided SQL query and returns pandas DataFrames. Logs errors but does not throw any exceptions while accessing BigQuery.

    Args:
        sql_query (str): The SQL query string.

    Returns:
        pandas.DataFrame: The resulting DataFrame
    """
    assert _BIGQUERY_CLIENT, 'Global bigquery client is not configured yet.'
    try:
        df = pd.read_gbq(sql_query, credentials=GLOBAL_CREDS)
    except Exception as e:
        logger.log_error('Exception during query: {}'.format(e))
    return df

class GoogleAuthModule(Module):
    """Module which allows to sign in users via the command line interface.
    """

    def __init__(self):
        super().__init__()
        self._colab = False

        if check_google_authenticated():
            logger.log_info('Already authenticated.')
            return

        colab = self._auth_colab()
        if colab:
            logger.log_info('Google Colab authenticated.')
            return
        else:
            logger.log_info('Google Colab auth not present.')

        auth = self._auth_normal()
        if not auth:
            logger.log_warn('No auth method worked! Printing help instructions for using this tool on private machines.')
            self._print_instructions()
        else:
            logger.log_info('Authenticated.')

    def _print_instructions(self):
        """
        Print login instructions."""

        text = """


        For using Google Cloud services (and of course this tool), you either need to authenticate via gcloud or use this tool in regular Colab notebooks.

        How to authenticate on your local machine:

        1) Follow the instructions on https://cloud.google.com/storage/docs/gsutil_install to install gsutil
        2) If not already done, run 'gcloud init' in you bash
        3) Run 'gcloud auth login' and follow the instructions
        4) Set your billing project: gcloud config set project
        5) Rerun this tool with the 'authenticate_at_google' option
        6) Finished!


        """
        print(text)

    def _auth_colab(self):
        """Authenticate via Colab backend.
        WARNING: Not tested yet.
        """
        try:
            from google.colab import auth
            from google.colab import drive
        except ImportError:
            return False
    
        try: 
            logger.log_debug('Google Colab environment detected - using this auth flow.')
            auth.authenticate_user()
            drive.mount('/content/drive', force_remount=True)
        except Exception as e:
            logger.log_debug('Google Colab auth not present. Exception: ' + str(e))
            return False

        return True

    def _auth_normal(self):
        """Authenticate user via normal sign in flow.

        Returns:
            bool: Login succeeded?
        """
        try:
            global GLOBAL_CREDS
            global PROJECT_ID
            key_file_path = config.get_pipeline_config_item('google_setup', 'keystore_path', None)

            if not key_file_path is None and not os.path.isfile(key_file_path):
                raise Exception('Could not find kdystore file for Google services: '+ str(key_file_path))
            elif os.path.isfile(key_file_path):
                from google.oauth2 import service_account
                GLOBAL_CREDS = service_account.Credentials.from_service_account_file(key_file_path)
                PROJECT_ID = config.get_pipeline_config_item('google_setup', 'google_project_id', None)
            else:
                GLOBAL_CREDS, PROJECT_ID = google.auth.default()
            
            assert PROJECT_ID, 'Project id required for Google services.'
            assert GLOBAL_CREDS, 'Global credentials  required for Google services.'

            return True

        except Exception as e:
            logger.log_debug('Exception during normal auth: ' + str(e))
            return False


    @classmethod
    def register_argparser_object(cls, subparser):
        """Register this module to the pipeline tool CLI

        Args:
            subparser (argparse.ArgumentParser): The argparse subparser to add this module to.
        """
        subparser.add_parser(cls.module_name(), help='Authenticate at Google Cloud.')

