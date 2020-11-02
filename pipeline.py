"""
Medical lexicon NLP extraction pipeline

File contains: Main file

-- (c) Rene Ahlsdorf 2019 - Team  D in the HST 953 class
"""

from na_pipeline_tool import utils
from na_pipeline_tool.utils import logger         as logger
from na_pipeline_tool.utils import google_tools   as google_tools
from na_pipeline_tool.utils import config         as config
from na_pipeline_tool.utils import helper_classes as helper

from na_pipeline_tool.lexicon_note_search import *
from na_pipeline_tool.negation_filter import *
from na_pipeline_tool.validation import *
from na_pipeline_tool.comprehension import *

import argparse, sys


VALID_MODULES = helper.ModuleMetaclass.get_modules()


def do_argparser(optstr = None):
    argparser = argparse.ArgumentParser(description="The NLP medical term extraction pipeline queries the MIMIC database for medical terms (e.g. non adherence) by utilizing medial lexicons and removes negated occurences of these terms from the final note dataset. It also provides further tools for analyzing the resulting cohort etc.")
    argparser.add_argument('-c', '--config', default="config.yaml", help='The YAML config file containing all options of the pipeline, Google cloud options etc.', required=True)
    
    subparser = argparser.add_subparsers(dest='module', help='The different submodules of this tool.')
    subparser.add_parser('all', help="Run the full pipeline. Please add the 'full_pipeline' key to your config file which specifies the pipeline stage order.")

    for _k, _v in VALID_MODULES.items():
        _v.register_argparser_object(subparser)
    
    if not optstr:
        parsed = argparser.parse_args()
    else:
        parsed = argparser.parse_args(optstr.strip().split(' '))

    return parsed, argparser


def run_module(optstr=None):
    parsed, parser = do_argparser(optstr=optstr)
    config.init_config(parsed.config)
    logger.init_logger()

    if parsed.module is None:
        parser.print_help()
        sys.exit(0)

    if parsed.module.lower() != 'all' and not parsed.module.lower() in VALID_MODULES:
        parser.error('Unknown module: {}'.format(parsed.module))
    if parsed.module.lower() == 'all':
        config_stages = config.get('full_pipeline', [])
        if not config_stages:
            parser.error('The full_pipeline key is empty or does not exist in config.')
        else:
            submodules_valid = [_.lower() for _ in config_stages if _.lower() in VALID_MODULES.keys()]
            submodules_invalid = [_.lower() for _ in config_stages if not _.lower() in VALID_MODULES.keys()]
            if submodules_invalid:
                parser.error('Invalid/Unknown pipeline stages in pipeline list: {}'.format(submodules_invalid))
            submodules = [VALID_MODULES[_] for _ in submodules_valid]
            last_piper_name = ""
            for _modname, _mod in zip(submodules_valid, submodules):
                inpname = config.pipeline_config_has(_modname, 'input_note_file')
                outname = config.pipeline_config_has(_modname, 'output_note_file')

                if inpname and last_piper_name:
                    assert config.override_in_pipeline_config(_modname, 'input_note_file', last_piper_name), "Could not override input key in piper {}".format(_modname)

                if outname:
                    gen_outname = '_all_out_{}.parquet'.format(_modname)
                    assert config.override_in_pipeline_config(_modname, 'output_note_file', gen_outname), "Could not override output key in piper {}".format(_modname)
                    last_piper_name = gen_outname
    else:
        submodules = [VALID_MODULES[parsed.module.lower()]]

    try:
        for _mod in submodules:
            _mod()
    except RuntimeError as e:
        if config.get('debugging', False):
            raise e
        logger.log_error('Runtime error during runtime: ' + str(e),1)
    except AssertionError as e:
        if config.get('debugging', False):
            raise e
        logger.log_error('Assertion error during runtime: ' + str(e),2)
    except Exception as e:
        if config.get('debugging', False):
            raise e
        logger.log_error('Common error during runtime: ' + str(e), 3)


if __name__ == '__main__':
    run_module()
    
