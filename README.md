# Nonadherence Concept Extraction 
[![Python 3.4](https://img.shields.io/badge/Requires-Python%203.4-blue.svg)](https://www.python.org/downloads/release/python-340/)

This repository contains the NLP code for the publication  *Automated Detection of Patient Non-Adherence from Clinical Notes* by *Rene Ahlsdorf, Shrey Lakhotia, M. Abdulhadi Alagha, Mattie Wasiak, Won Young Cheong, Adrian Wong, Joseph Terdiman, Daniel Gruhl, Leo A. Celi and Joy T. Wu*.  It does not contain the data science evaluation part of the paper and the actual MIMIC data files.
For accessing [MIMIC-III](https://mimic.physionet.org/) via Google BigQuery, please have a look at [the official guide](https://mimic.physionet.org/gettingstarted/cloud/). Once you have access to the BigQuery MIMIC dataset. you can start using this tool and create your own concept extraction pipeline.

## Table of Content

 1. [Structure of this repository](#1-structure-of-this-repository)
 2. [How to run the tool](#2-how-to-run-the-tool)
 3. [Citation](#3-citation)
 4. [License](#4-license)

## 1. Structure of this repository

> Please note that the code in this repository is only compatible to Python >=3.4


| File/Folder | Purpose |
|--|--|
| na_pipeline_tool/ | This folder contains the source of the pipeline tool |
| generated_df_dir | This folder is empty but will contain the resutling pipeline stage `.parquet` files. | 
| non_adherence_lexicons/ | This folder contains the different non-adherence lexicons (`positive`: Non-adherence terms; `negative`: Adherence terms)|
| nonadherence_diet_drugs_correction/ | This folder contains the lexicons for special words which can change the label assignment from one NA category (e.g. NOS) to another one (e.g. dietary NA) if they match these terms in the note texts. |
| cohort_na.csv | The MIMIC-III cohort we analyzed in our publication |
| validationset_nonadherence_2019.csv | The non-adherence validation dataset which we contributed to Physionet. |
| config.yaml | The config file which the pipeline uses to configure the different stages. Please have a look at the file to see the available options. |
| requirements.txt | The Python3 requirements. |
| pipeline.py | The main pipeline script. Please have a look at [Section 2](#2-how-to-run-the-tool) how to run it. |

## 2. How to run the tool
If you want to run the tool, we recommend you to create a new `anaconda / virtualenv` environment before you proceed. While doing so, you should pick a Python version `>=3.4.0,<4.0`. The example below demonstrates how you re-run our non-adherence analysis given that you already have access to MIMIC-III on Google BigQuery.

 1. Install all pip requirements by running `pip3 install -r requirements.txt` in the root of this repository in your `anaconda / virtualenv` environment.
 2. Configure the `config.yaml` file according to your needs. First, you should insert your Google project id in the `google_setup` config block by setting `google_project_id: <YOUR ID>`.
 3. Run the pipeline. For this, you can run individual pipeline stages by calling `python3 pipeline.py -c config.yaml <Name of the pipeline stage>`. You can list all available pipeline stages and their features by running `python3 pipeline.py --help`. If you want to run the full pipeline (each stage normally just outputs one `.parquet` file), define the `full_pipeline: ["bigquery_mimic_lexicon_note_search",..]` item in the root of the YAML file and run `python pipeline.py -c config.yaml all`.
 4. If you want to compute the validation metrics, run the module `validation`. You'll find the outputs of each stage in the folder you specified in the `config.yaml` file under the `generated_df_dir` key.

## 3. Citation
If you're building on top of our work, please cite us as follows:

> Blockquote TODO

    Bibtex Source TODO

## 4. License
[MIT License](./LICENSE)
