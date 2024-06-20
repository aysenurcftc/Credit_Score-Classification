

RANDOM_SEED = 6


run_mode = "training"
dataset = "loan" # datasets -> "uci_credit_approval" , "statlog_german_credit_data", "loan"

model_type = "logistic regression" # types -> "logistic regression", "NN"
imputing_type_numerical = "median" # types -> median, mean
imputing_type_categorical = "mode" # types -> mode
split_ratio = 0.7

#data path
filepath = "data/loan/loan.csv" 
