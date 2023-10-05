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
    # {
    #     "inputFile": "Classification/FinServ/Notebooks\ with\ Code/N1_deploy_a_model-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
    #     "outputFile": "01-use-case-classification-finserv-reference.md"
    # },
    # {
    #     "inputFile": "Classification/FinServ/Notebooks\ with\ Code/N2_production_experiments-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
    #     "outputFile": "02-use-case-classification-finserv-reference.md"
    # },
    # {
    #     "inputFile": "Classification/FinServ/Notebooks\ with\ Code/N3_validation_rules-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
    #     "outputFile": "03-use-case-classification-finserv-reference.md"
    # },
    # {
    #     "inputFile": "Classification/FinServ/Notebooks\ with\ Code/N4_drift_detection-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
    #     "outputFile": "04-use-case-classification-finserv-reference.md"
    # },
    # {
    #     "inputFile": "Computer\ Vision/Retail/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/cv/retail",
    #     "outputFile": "01-use-case-cv-retail-reference.md"
    # },
    # {
    #     "inputFile": "Computer\ Vision/Retail/Notebooks-with-code/N2_production_experiments-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/cv/retail",
    #     "outputFile": "02-use-case-cv-retail-reference.md"
    # },
    # {
    #     "inputFile": "Computer\ Vision/Retail/Notebooks-with-code/N3_drift_detection-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/cv/retail",
    #     "outputFile": "03-use-case-cv-retail-reference.md"
    # },
    # {
    #     "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/forecasting/retail-cpg",
    #     "outputFile": "01-use-case-forecasting-retail-cpg-reference.md"
    # },
    # {
    #     "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N2_production_experiments-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/forecasting/retail-cpg",
    #     "outputFile": "02-use-case-forecasting-retail-cpg-reference.md"
    # },
    # {
    #     "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N3_drift_detection-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/forecasting/retail-cpg",
    #     "outputFile": "03-use-case-forecasting-retail-cpg-reference.md"
    # },
    # {
    #     "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N4_automate-data-connections-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/forecasting/retail-cpg",
    #     "outputFile": "04-use-case-forecasting-retail-cpg-reference.md"
    # },
    # {
    #     "inputFile": "Forecasting/Retail-CPG/Notebooks-with-code/N5_automate-ml-orchestration-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/forecasting/retail-cpg",
    #     "outputFile": "05-use-case-forecasting-retail-cpg-reference.md"
    # },
    # {
    #     "inputFile": "Linear\ Regression/Real\ Estate/Notebooks-with-code/N1_deploy_a_model-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
    #     "outputFile": "01-use-case-linear-regression-real-estate-reference.md"
    # },
    # {
    #     "inputFile": "Linear\ Regression/Real\ Estate/Notebooks-with-code/N2_production_experiments-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
    #     "outputFile": "02-use-case-linear-regression-real-estate-reference.md"
    # },
    # {
    #     "inputFile": "Linear\ Regression/Real\ Estate/Notebooks-with-code/N3_validation_rules-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
    #     "outputFile": "03-use-case-linear-regression-real-estate-reference.md"
    # },
    # {
    #     "inputFile": "Linear\ Regression/Real\ Estate/Notebooks-with-code/N4_drift_detection-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
    #     "outputFile": "04-use-case-linear-regression-real-estate-reference.md"
    # },
    # {
    #     "inputFile": "Linear\ Regression/Real\ Estate/Notebooks-with-code/N5_automate-data-connections-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
    #     "outputFile": "05-use-case-linear-regression-real-estate-reference.md"
    # },
    # {
    #     "inputFile": "Linear\ Regression/Real\ Estate/Notebooks-with-code/N6_automate-ml-orchestration-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/linear-regression/real-estate",
    #     "outputFile": "06-use-case-linear-regression-real-estate-reference.md"
    # },
    # {
    #     "inputFile": "NLP_Classification/Sentiment\ Analysis/Notebooks\ with\ code/N1_deploy_a_model-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/nlp-classification/sentiment-analysis",
    #     "outputFile": "01-use-case-nlp-classification-sentiment-analysis-reference.md"
    # },
    # {
    #     "inputFile": "NLP_Classification/Sentiment\ Analysis/Notebooks\ with\ code/N2_validation_rules-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/nlp-classification/sentiment-analysis",
    #     "outputFile": "02-use-case-nlp-classification-sentiment-analysis-reference.md"
    # },
    # {
    #     "inputFile": "NLP_Classification/Sentiment\ Analysis/Notebooks\ with\ code/N3_drift_detection-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/nlp-classification/sentiment-analysis",
    #     "outputFile": "03-use-case-nlp-classification-sentiment-analysis-reference.md"
    # },
    # {
    #     "inputFile": "Classification/FinServ/Notebooks\ with\ Code/N1_deploy_a_model-with-code.ipynb",
    #     "outputDir": "/wallaroo-use-case-tutorials/classification/finserv",
    #     "outputFile": "01-use-case-classification-finserv-reference.md"
    # },

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
    document = re.sub('!\[png\]\(', f'![png](/images/2023.2.1{outputdir}/', document)
    document = re.sub('\(./images', f'(/images/2023.2.1{outputdir}', document)
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
        move_images(currentFile["outputDir"])
    # get rid of any extra markdown files
    os.system("find ./images -name '*.md' -type f -delete")

if __name__ == '__main__':
    main()