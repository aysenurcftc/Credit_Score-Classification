
#Constants
RANDOM_SEED = 6


run_mode = "training"
dataset = "default_of_credit_card_clients" # datasets -> "uci_credit_approval" , "statlog_german_credit_data", "loan", "statlog_australian_credit_approval", "default_of_credit_card_clients"


model_type = "logistic regression" # types -> "logistic regression", "NN"
imputing_type_numerical = "median" # types -> median, mean
imputing_type_categorical = "mode" # types -> mode
split_ratio = 0.8

#data path
filepath = "data/default_of_credit_card_clients.csv" 
