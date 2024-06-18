# Kaggle Challenge: Data driven text mining

## Introduction
This is the code for the 2024-UCSD-CSE291-data-driven-text-mining competition. Several approaches have been implemented, with the best one achieving a 0.82820 Micro-F1 on the final leaderboard. For more details, check the competition [link](https://www.kaggle.com/competitions/2024-ucsd-cse291-data-driven-text-mining).

## Dataset

Access the training dataset (with `label` column) and the test dataset (without `label` column) at [Kaggle](https://www.kaggle.com/competitions/2024-ucsd-cse291-data-driven-text-mining/data). The objective is to predict the `label` column using the F1-micro as the performance metric.

## Approaches
### Data Processing
- Initially, convert all data into tensor float type: transform booleans to 0 and 1, apply regularization on numerical data, convert categorical data into one-hot encoding, and flatten JSON data into multiple columns.
- For missing data, perform an analysis to determine the significance of different data columns. Based on significance and missing rates, decide whether to keep or remove a column. Then, impute missing values for the retained columns.

### Models
- Multiple models are used to predict the probability distribution, training a probability weight vector to sum all probabilities, which improves the F1-micro score on both validation and test datasets.
- The `name` column shows a strong indication of the restaurant type. By merging the `name` and `review` columns and training on BERT and BERTopic ([source](https://maartengr.github.io/BERTopic/index.html)), we leverage textual data effectively. TFIDF is used to generate vector representations of each text, which are then trained on a linear classifier.
- For boolean, numeric, and categorical columns, apply Random Forest and XGBoost to predict the probabilities.
- These columns are also used as supporting data for LLMS. Using dense layers, they are converted into linear representations and merged with the output of LLMS. We then further train a classifier on this augmented data.

## How to Use

1) All code used in this challenge is provided in the provided Jupyter notebook:
   - `Data Processing`: For data analysis and preparation.
   - `Hybrid Model - Data Preparation`: Converts data to model inputs.
   - `Hybrid Model - Model Final`: Contains the final model configuration and has yielded the best results.
2) Running on Google Colab is recommended
3) Check [Result](predicted.csv) for best result on `test.csv`