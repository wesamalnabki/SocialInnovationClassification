import math
import shutil

import pandas as pd
import torch
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import *
from data_collector.data_collector_funs import DataCollector
from text_processing.text_processing_unit import TextProcessingUnit
from utils import print_result


class SocialInnovationLearningCurve:
    def __init__(self):

        self.USE_CUDA = False
        if torch.cuda.is_available():
            self.USE_CUDA = True

        self.text_processing_unit = TextProcessingUnit()
        self.data_collector = DataCollector()

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

    def build_learning_curve(self, dataset_path, step_size):

        dataset = self.load_training_set(dataset_path, save_finally=True)
        training_set, testing_set = train_test_split(dataset, test_size=0.2)

        for CRITERION in SI_Criteria:
            print(f'Start building the LR for {CRITERION}')

            lc_tmp_path = f'learning_curve/lc_tmp_path/{CRITERION}/'
            os.makedirs(lc_tmp_path, exist_ok=True)

            train_args = {
                # path where to save the model
                "output_dir": lc_tmp_path,
                # path where to save the best model
                "best_model_dir": lc_tmp_path,
                # longest text processed (max bert)
                'max_seq_length': 512,  # 512
                'num_train_epochs': 3,  # 5
                'train_batch_size': 16,
                'eval_batch_size': 32,
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
            else:
                train_args['n_gpu'] = 0

            train_args['use_multiprocessing'] = False
            train_args['use_multiprocessing_for_evaluation'] = False
            train_args['fp16'] = False

            training_set_ctr = training_set[['Project_id', 'text', CRITERION]]
            testing_set_ctr = testing_set[['Project_id', 'text', CRITERION]]

            training_set_ctr.columns = ['Project_id', "text", "labels"]
            testing_set_ctr.columns = ['Project_id', "text", "labels"]

            training_set_ctr['text'] = training_set_ctr['text'].replace(r"\n", " ", regex=True)
            testing_set_ctr['text'] = testing_set_ctr['text'].replace(r"\n", " ", regex=True)

            training_set_ctr['labels'] = training_set_ctr['labels'].astype(int)
            testing_set_ctr['labels'] = testing_set_ctr['labels'].astype(int)

            learning_curve_data = {}
            # Iterate over the dataset
            for subset_size in range(1, int(math.ceil(len(training_set_ctr) / step_size) + 1)):
                training_subset = training_set_ctr.iloc[0:subset_size * step_size]
                print('training_subset: ', len(training_subset))
                if len(set(training_subset.labels.values.tolist())) < 3:
                    print('Escape, the subset has to have samples from all the classes')
                    continue

                # Create a ClassificationModel
                clf_model = ClassificationModel(model_type,
                                                model_name,
                                                num_labels=len(set(training_subset.labels.values.tolist())),
                                                args=train_args,
                                                use_cuda=self.USE_CUDA
                                                )

                clf_model.train_model(training_subset, show_running_loss=False, verbose=False)

                # Evaluate the model
                predictions, _ = clf_model.predict(testing_set_ctr.text.tolist())

                metric_result = print_result(predictions, testing_set_ctr.labels.values.tolist())
                learning_curve_data[str(len(training_subset))] = metric_result

            # Saving the result
            with open(f'{learning_curve_path}/lr_{CRITERION}.csv', 'w') as wrt:
                line = f"Training Size\tMacro F1\tMacro Recall\tMacro Precision\n"
                wrt.write(line)
                for size, res in learning_curve_data.items():
                    line = f"{size}\t{res['macro F1']}\t{res['macro recall']}\t{res['macro precision']}\n"
                    wrt.write(line)

            if os.path.exists(lc_tmp_path):
                shutil.rmtree(lc_tmp_path)

            print(f'Finish building the LR for {CRITERION}')
