import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from simpletransformers.classification import ClassificationModel
from tqdm import tqdm

from config import *
from data_collector.data_collector_funs import DataCollector
from text_processing.text_processing_unit import TextProcessingUnit
from utils import print_result, return_df, get_logger


class SocialInnovationClassifierTraining:
    # use GPU for training or not
    USE_CUDA = False
    if torch.cuda.is_available():
        USE_CUDA = True

    def __init__(self):

        self.model_name = "bert-base-cased"
        self.model_type = "bert"

        # format it to a string
        timestamp_format = datetime.now().strftime('%Y-%m-%d_%H-%M')

        self.run_name = f'{self.model_name}_{timestamp_format}'
        self.run_path = f'{model_path}/{self.run_name}'

        self.text_processing_unit = TextProcessingUnit()
        self.data_collector = DataCollector()

        self.logger = get_logger(run_name=self.run_name, run_type="Training", log_path=log_path)

    def load_training_set(self, dataset_path, top_x=None, save_finally=True):

        if dataset_path is not None and os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
        else:
            projects_data = self.data_collector.load_training_project_ids(top_x=top_x)
            for p in tqdm(projects_data, "Getting and cleaning projects text"):
                p_id = p['Project_id']
                p['text'] = self.text_processing_unit.clean_text(
                    self.text_processing_unit.shorten_text(
                        self.data_collector.get_project_text(p_id, mongodb_max_pages=10)
                    )
                )

            df = pd.DataFrame(projects_data)
            if save_finally:
                df.to_csv(dataset_path, index=False, encoding='utf-8')

        # shuffle the samples
        df = df.sample(n=len(df), random_state=42)

        # Remove any sample has empty text:
        df.dropna(subset=['text'], how='all', inplace=True)

        return df

    def training_classifiers(self, training_set):

        # Dictionary to hold the exp information
        run_dict = {
            'MODEL_NAME': self.model_name,
            'MODEL_TYPE': self.model_type,
            'TRAINING_ARGS': {},
            'TRAINING_PROJECT_IDS': {},
        }

        run_config_path = self.run_path + '/run_config.json'
        os.makedirs(self.run_path, exist_ok=True)

        # Start building a classifier for each criterion
        for CRITERION in CRITERIONS:

            self.logger.info(f'Training a classifier for {CRITERION}')
            # read three columns from the cvs file:
            sub_dataset = training_set[['Project_id', 'text', CRITERION]]

            # split dataframe to train and test 80/20.
            train_df, test_df = return_df(sub_dataset)
            self.logger.info(
                f'Split the dataset 80/20. Training length is {len(train_df)} and Testing length is {len(test_df)}')

            # setup the training params (Do NOT change them)
            train_args = {
                # path where to save the model
                "output_dir": f'{self.run_path}/{CRITERION}/',
                # path where to save the best model
                "best_model_dir": f'{self.run_path}/{CRITERION}/best_model/',
                # longest text processed (max bert)
                'max_seq_length': 256 + 128,
                'num_train_epochs': 3,
                'train_batch_size': 8,
                'eval_batch_size': 16,
                'gradient_accumulation_steps': 1,
                'learning_rate': 5e-5,
                'save_steps': 5000,

                'reprocess_input_data': True,
                "save_model_every_epoch": False,
                'overwrite_output_dir': True,
                'no_cache': True,

                'use_early_stopping': True,
                'early_stopping_patience': 3,
                'manual_seed': 42,
            }

            if self.USE_CUDA:
                train_args['n_gpu'] = 1
                train_args['fp16'] = True
            else:
                train_args['n_gpu'] = 0
                train_args['fp16'] = False

            train_args['use_multiprocessing'] = False
            train_args['use_multiprocessing_for_evaluation'] = False

            run_dict['TRAINING_ARGS'] = train_args
            run_dict['TRAINING_PROJECT_IDS'] = train_df.Project_id.tolist()

            # Create a ClassificationModel
            model = ClassificationModel(self.model_type,
                                        self.model_name,
                                        args=train_args,
                                        num_labels=len(set(train_df.labels.values.tolist())),  # --> 3
                                        use_cuda=self.USE_CUDA,
                                        weight=[1, 0.5, 4]
                                        )

            self.logger.info(f'Start training a model for {CRITERION}')
            model.train_model(train_df, test_df)
            self.logger.info(f'Finish training a model for {CRITERION}')

            # Evaluate the model
            self.logger.info(f'Start evaluating the {CRITERION} model')
            result, model_outputs, wrong_predictions = model.eval_model(test_df)

            self.logger.info(f'Writing the result to dist')
            # evaluated the performance on the test set
            predictions = np.argmax(model_outputs, axis=1)
            metric_result = print_result(predictions, test_df.labels.values.tolist())
            run_dict[f'model_metric_result_{CRITERION}'] = metric_result

            self.logger.info(f'Moving to the next classification model')

        # Saving the exp config to disk (applied only after training 4 classifiers)
        with open(run_config_path, 'w') as outfile:
            json.dump(run_dict, outfile, sort_keys=True, indent=4)

        self.logger.info('Finish training!!')
