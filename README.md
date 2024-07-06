
# Credit Score Classification 

The credit score classification was performed using five different datasets.


## Installation



```bash
pip install -r requirements.txt
```
    
## Usage/Examples

```
usage: main.py [-h] [--run_mode RUN_MODE] [--dataset DATASET] [--model_type MODEL_TYPE] [--imputing_type_numerical IMPUTING_TYPE_NUMERICAL]
               [--imputing_type_categorical IMPUTING_TYPE_CATEGORICAL] [--split_ratio SPLIT_RATIO] [--filepath FILEPATH]

Train a model on a specified dataset.

options:
  -h, --help            show this help message and exit
  --run_mode RUN_MODE   Mode to run the script in, e.g., training or testing
  --dataset DATASET     Dataset to use for training
  --model_type MODEL_TYPE
                        Type of model to use, e.g., 'logistic regression' or 'NN'
  --imputing_type_numerical IMPUTING_TYPE_NUMERICAL
                        Imputing strategy for numerical features
  --imputing_type_categorical IMPUTING_TYPE_CATEGORICAL
                        Imputing strategy for categorical features
  --split_ratio SPLIT_RATIO
                        Train-test split ratio
  --filepath FILEPATH   Path to the dataset file
None
usage: main.py [-h] [--run_mode RUN_MODE] [--dataset DATASET] [--model_type MODEL_TYPE] [--imputing_type_numerical IMPUTING_TYPE_NUMERICAL]
               [--imputing_type_categorical IMPUTING_TYPE_CATEGORICAL] [--split_ratio SPLIT_RATIO] [--filepath FILEPATH]

Train a model on a specified dataset.

options:
  -h, --help            show this help message and exit
  --run_mode RUN_MODE   Mode to run the script in, e.g., training or testing
  --dataset DATASET     Dataset to use for training
  --model_type MODEL_TYPE
                        Type of model to use, e.g., 'logistic regression' or 'NN'
  --imputing_type_numerical IMPUTING_TYPE_NUMERICAL
                        Imputing strategy for numerical features
  --imputing_type_categorical IMPUTING_TYPE_CATEGORICAL
                        Imputing strategy for categorical features
  --split_ratio SPLIT_RATIO
                        Train-test split ratio
  --filepath FILEPATH   Path to the dataset file


```


## Deployment

To deploy this project run

```bash
   docker build -t ml_proje . 
   docker run ml_proje  --run_mode training --dataset statlog_german_credit_data  --model_type NN  --imputing_type_numerical median --imputing_type_categorical mode --split_ratio 0.7 --filepath data/statlog_german_credit_data.csv

```

