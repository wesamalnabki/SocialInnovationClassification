import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# where to save the models
model_path = 'models'
prediction_path = 'prediction'
log_path = "logs"
learning_curve_path = "learning_curve"

SI_Criteria = ['CriterionActors', 'CriterionInnovativeness', 'CriterionObjectives', 'CriterionOutputs']

# model to use
model_name = "bert-base-cased"

# model family
model_type = "bert"

os.makedirs(prediction_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
