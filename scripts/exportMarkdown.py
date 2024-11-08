#!/usr/bin/env python

"""
Converts the following Python notebooks into the same format used for the Wallaroo Documentation site.

This uses the jupyter nbconvert command.  For now this will always assume we're exporting to markdown:

    jupyter nbconvert {file} --to markdown --output {output}

"""

import os
import nbformat
from traitlets.config import Config
import re
import shutil
import glob
#import argparse

c = Config()

c.NbConvertApp.export_format = "markdown"

docs_directory = "docs/markdown"

fileList = [
    ## Classification - Cybersecurity
    {
        "inputFile": "Classification/Cybersecurity/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/cybersecurity",
        "outputFile": "N1_deploy_a_model-with-code-reference.md"
    },
    {
        "inputFile": "Classification/Cybersecurity/Notebooks-with-code/N2_production_experiments-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/cybersecurity",
        "outputFile": "N2_production_experiments-with-code-reference.md"
    },
    {
        "inputFile": "Classification/Cybersecurity/Notebooks-with-code/N3_validation_rules-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/cybersecurity",
        "outputFile": "N3_validation_rules-with-code-reference.md"
    },
    {
        "inputFile": "Classification/Cybersecurity/Notebooks-with-code/N4_drift_detection-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/cybersecurity",
        "outputFile": "N4_drift_detection-with-code-reference.md"
    },
    {
        "inputFile": "Classification/Cybersecurity/Notebooks-with-code/N5_automate-data-connections-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/cybersecurity",
        "outputFile": "N5_automate-data-connections-with-code-reference.md"
    },
    {
        "inputFile": "Classification/Cybersecurity/Notebooks-with-code/N6_automate-ml-orchestration-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/cybersecurity",
        "outputFile": "N6_automate-ml-orchestration-with-code-reference.md"
    },
    {
        "inputFile": "Classification/Cybersecurity/Notebooks-with-code/N7_publish_pipeline_for_edge-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/cybersecurity",
        "outputFile": "N7_publish_pipeline_for_edge-with-code-reference.md"
    },
    ## Classification - FinServ
    {
        "inputFile": "Classification/FinServ/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
        "outputFile": "N1_deploy_a_model-with-code-reference.md"
    },
    {
        "inputFile": "Classification/FinServ/Notebooks-with-code/N2_production_experiments-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
        "outputFile": "N2_production_experiments-with-code-reference.md"
    },
    {
        "inputFile": "Classification/FinServ/Notebooks-with-code/N3_validation_rules-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
        "outputFile": "N3_validation_rules-with-code-reference.md"
    },
    {
        "inputFile": "Classification/FinServ/Notebooks-with-code/N4_drift_detection-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
        "outputFile": "N4_drift_detection-with-code-reference.md"
    },
    {
        "inputFile": "Classification/FinServ/Notebooks-with-code/N5_automate-data-connections-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
        "outputFile": "N5_automate-data-connections-with-code-reference.md"
    },
    {
        "inputFile": "Classification/FinServ/Notebooks-with-code/N6_publish_pipeline_for_edge-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
        "outputFile": "N6_publish_pipeline_for_edge-with-code-reference.md"
    },
    ## Computer-Vision - Healthcare
    {
        "inputFile": "Computer-Vision/Healthcare/Notebooks-with-code/N0-environment-prep-model-conversion.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/healthcare",
        "outputFile": "N0-environment-prep-model-conversion-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Healthcare/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/healthcare",
        "outputFile": "N1_deploy_a_model-with-code-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Healthcare/Notebooks-with-code/N2_automate-data-connections-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/healthcare",
        "outputFile": "N2_automate-data-connections-with-code-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Healthcare/Notebooks-with-code/N3_publish_pipeline_for_edge-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/healthcare",
        "outputFile": "N3_publish_pipeline_for_edge-with-code-reference.md"
    },
    ## Computer-Vision - Retail
    {
        "inputFile": "Computer-Vision/Retail/N0-environment-prep-model-conversion.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/retail",
        "outputFile": "N0-environment-prep-model-conversion-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Retail/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/retail",
        "outputFile": "N1_deploy_a_model-with-code-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Retail/Notebooks-with-code/N2_production_experiments-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/retail",
        "outputFile": "N2_production_experiments-with-code-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Retail/Notebooks-with-code/N3_validation_rules-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/retail",
        "outputFile": "N3_validation_rules-with-code-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Retail/Notebooks-with-code/N4_drift_detection-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/retail",
        "outputFile": "N4_drift_detection-with-code-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Retail/Notebooks-with-code/N5_publsh_pipeline_for_edge-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/retail",
        "outputFile": "N5_publsh_pipeline_for_edge-with-code-reference.md"
    },
    ## Computer-Vision - Yolov8
    {
        "inputFile": "Computer-Vision/Yolov8/Notebooks-with-code/N0-environment-prep-model-conversion.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/yolov8",
        "outputFile": "N0-environment-prep-model-conversion-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Yolov8/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/yolov8",
        "outputFile": "N1_deploy_a_model-with-code-reference.md"
    },
    {
        "inputFile": "Computer-Vision/Yolov8/Notebooks-with-code/N2_publish_pipeline_for_edge-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/computer-vision/yolov8",
        "outputFile": "N2_publish_pipeline_for_edge-with-code-reference.md"
    },
    ## Forecasting - Retail CPG
    {
        "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/forecast/retail",
        "outputFile": "N1_deploy_a_model-with-code-reference.md"
    },
    {
        "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N2_production_experiments-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/forecast/retail",
        "outputFile": "N2_production_experiments-with-code-reference.md"
    },
    {
        "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N3_validation_rules-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/forecast/retail",
        "outputFile": "N3_validation_rules-with-code-reference.md"
    },
    {
        "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N4_drift_detection-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/forecast/retail",
        "outputFile": "N4_drift_detection-with-code-reference.md"
    },
    {
        "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N5_automate-data-connections-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/forecast/retail",
        "outputFile": "N5_automate-data-connections-with-code-reference.md"
    },
    {
        "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N6_automate-ml-orchestration-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/forecast/retail",
        "outputFile": "N6_automate-ml-orchestration-with-code-reference.md"
    },
    {
        "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N7_publish_pipeline_for_edge-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/forecast/retail",
        "outputFile": "N7_publish_pipeline_for_edge-with-code-reference.md"
    },
    ## Linear Regression - Real Estate
    {
        "inputFile": "Linear-Regression/Real-Estate/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
        "outputFile": "N1_deploy_a_model-with-code-reference.md"
    },
    {
        "inputFile": "Linear-Regression/Real-Estate/Notebooks-with-code/N2_production_experiments-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
        "outputFile": "N2_production_experiments-with-code-reference.md"
    },
    {
        "inputFile": "Linear-Regression/Real-Estate/Notebooks-with-code/N3_validation_rules-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
        "outputFile": "N3_validation_rules-with-code-reference.md"
    },
    {
        "inputFile": "Linear-Regression/Real-Estate/Notebooks-with-code/N3_validation_rules-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
        "outputFile": "N3_validation_rules-with-code-reference.md"
    },
    {
        "inputFile": "Linear-Regression/Real-Estate/Notebooks-with-code/N4_drift_detection-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
        "outputFile": "N4_drift_detection-with-code-reference.md"
    },
    {
        "inputFile": "Linear-Regression/Real-Estate/Notebooks-with-code/N5_automate-data-connections-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
        "outputFile": "N5_automate-data-connections-with-code-reference.md"
    },
    {
        "inputFile": "Linear-Regression/Real-Estate/Notebooks-with-code/N6_automate-ml-orchestration-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
        "outputFile": "N6_automate-ml-orchestration-with-code-reference.md"
    },
    {
        "inputFile": "Linear-Regression/Real-Estate/Notebooks-with-code/N7_publish_pipeline_for_edge-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
        "outputFile": "N7_publish_pipeline_for_edge-with-code-reference.md"
    },
    ## LLM Summarization
    {
        "inputFile": "LLM/Summarization/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/llm/summarization",
        "outputFile": "N1_deploy_a_model-with-code-reference.md"
    },
    {
        "inputFile": "LLM/Summarization/Notebooks-with-code/N2_automate-data-connections-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/llm/summarization",
        "outputFile": "N2_automate-data-connections-with-code-reference.md"
    },
    {
        "inputFile": "LLM/Summarization/Notebooks-with-code/N3_publsh_pipeline_for_edge-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/llm/summarization",
        "outputFile": "N3_publsh_pipeline_for_edge-with-code-reference.md"
    },
    ## NLP Classification - Sentiment Analysis
    {
        "inputFile": "NLP_Classification/Sentiment-Analysis/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/nlp-classification/sentiment-analysis",
        "outputFile": "N1_deploy_a_model-with-code-reference.md"
    },
    {
        "inputFile": "NLP_Classification/Sentiment-Analysis/Notebooks-with-code/N2_validation_rules-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/nlp-classification/sentiment-analysis",
        "outputFile": "N2_validation_rules-with-code-reference.md"
    },
    {
        "inputFile": "NLP_Classification/Sentiment-Analysis/Notebooks-with-code/N3_drift_detection-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/nlp-classification/sentiment-analysis",
        "outputFile": "N3_drift_detection-with-code-reference.md"
    },
    {
        "inputFile": "NLP_Classification/Sentiment-Analysis/Notebooks-with-code/N4_publish_pipeline_for_edge-with-code.ipynb",
        "outputDir": "/wallaroo-use-case-tutorials/nlp-classification/sentiment-analysis",
        "outputFile": "N4_publish_pipeline_for_edge-with-code-reference.md"
    }
]

def format(outputdir, document_file):
    # Take the markdown file, remove the extra spaces
    document = open(f'{docs_directory}{outputdir}/{document_file}', "r").read()
    result = re.sub
    
    # fix tables for publication
    # document = re.sub(r'<table.*?>', r'{{<table "table table-striped table-bordered" >}}\n<table>', document)
    # document = re.sub('</table>', r'</table>\n{{</table>}}', document)
    # remove any div table sections
    document = re.sub('<div.*?>', '', document)
    document = re.sub(r'<style.*?>.*?</style>', '', document, flags=re.S)
    document = re.sub('</div>', '', document)

    # replace workshop with tutorial
    document = re.sub('Workshop', 'Tutorial', document)
    document = re.sub('workshop', 'tutorial', document)

    # remove non-public domains
    document = re.sub('wallaroocommunity.ninja', 'wallarooexample.ai', document)

    # fix image directories
    # ](01_notebooks_in_prod_explore_and_train-reference_files
    # image_replace = f'![png]({outputdir}'
    document = re.sub('!\[png\]\(', f'![png](/images/2024.2{outputdir}/', document)
    document = re.sub('\(./images', f'(/images/2024.2{outputdir}', document)
    # move them all to Docsy figures
    document = re.sub(r'!\[(.*?)\]\((.*?)\)', r'{{<figure src="\2" width="800" label="\1">}}', document)

    # strip the excess newlines - match any pattern of newline plus another one or more empty newlines
    document = re.sub(r'\n[\n]+', r'\n\n', document)

    # save the file for publishing
    newdocument = open(f'{docs_directory}{outputdir}/{document_file}', "w")
    newdocument.write(document)
    newdocument.close()

def move_images(image_directory):
    source_directory = f"{docs_directory}{image_directory}"
    target_directory = f"./images{image_directory}"
    # check the current directory for reference files
    # reference_directories = os.listdir(image_directory)
    print(source_directory)
    reference_directories = [ name for name in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, name)) ]
    # copy only the directories to their image location
    for reference in reference_directories:
        print(f"cp -rf ./{source_directory}/{reference} {target_directory}")
        # print(f"To: {target_directory}/{reference}")
        os.system(f"cp -rf ./{source_directory}/{reference} {target_directory}")

def main():
    for currentFile in fileList:
        convert_cmd = f'jupyter nbconvert --to markdown --output-dir {docs_directory}{currentFile["outputDir"]} --output {currentFile["outputFile"]} {currentFile["inputFile"]}'
        print(convert_cmd)
        os.system(convert_cmd)
        # format(f'{docs_directory}{currentFile["outputDir"]}/{currentFile["outputFile"]}')
        format(currentFile["outputDir"],currentFile["outputFile"])
        #move_images(currentFile["outputDir"])
    # get rid of any extra markdown files
    #os.system("find ./images -name '*.md' -type f -delete")

if __name__ == '__main__':
    main()