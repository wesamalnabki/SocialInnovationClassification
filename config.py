import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# where to save the models
model_path = 'models'
prediction_path = 'prediction'
log_path = "logs"
CRITERIONS = ['CriterionActors', 'CriterionInnovativeness', 'CriterionObjectives', 'CriterionOutputs']

os.makedirs(prediction_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
