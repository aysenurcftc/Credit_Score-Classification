
#Constants
RANDOM_SEED = 42


run_mode = "training"
dataset = "uci_credit_approval" # datasets -> "uci_credit_approval" , "statlog_german_credit_data", "loan", "statlog_australian_credit_approval", "default_of_credit_card_clients"


model_type = "NN" # types -> "logistic regression", "NN"
imputing_type_numerical = "median" # types -> median, mean
imputing_type_categorical = "mode" # types -> mode
split_ratio = 0.7

#data path
filepath = "data/uci_credit_approval.csv" 
