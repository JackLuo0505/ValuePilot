# README

This project involves generating, processing, training, and testing a dataset using several scripts. The following steps outline the entire process:

## 1. Generate Dataset using `dataset_preparing`

First, generate the dataset using the scripts available in the `dataset_preparing` directory. Each script is designed to generate a dataset with a specific number of values.

## 2. Filter Data using `find_error_data.py`

Next, filter the generated dataset using the `find_error_data.py` script. This script will process the JSON files to identify and filter out entries influenced by specific value dimensions.

## 3. Preprocess Data using `preprocess.py`

Once the data is filtered, preprocess the data using the `preprocess.py` script. This step prepares the data for training by performing necessary transformations and formatting.

## 4. Train Models using `CL_labeling_v2.py` and `ACCL_CJRN.py`

With the preprocessed data, train the models using the `CL_labeling_v2.py` and `ACCL_CJRN.py` scripts. These scripts implement the training process for the Sequential Evaluation Network (SEN) with the Adaptive Complexity Curriculum Learning (ACCL) framework.

## 5. Test Value-Module using `inference_combine_results.py`

After training the models, test the value-module using the `inference_combine_results.py` script. This script evaluates the model's performance and combines the results from different networks.

## 6. Test Action Selection in Scenarios using `action_chosing.py`

Then, test the action selection in different scenarios using the `action_chosing.py` script. This script evaluates and ranks actions based on user preferences and the results from the trained models.

## 7. Compare final results using `human_study/compare.py`
Finally, test model performance in action choosing with real human and other benchmark LLM models using `human_study/compare.py`. This script list out the similarity between real human data and models, including our ValuePilot and other large language models.