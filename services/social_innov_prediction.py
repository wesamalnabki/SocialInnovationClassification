import sys

import pandas as pd
from simpletransformers.classification import ClassificationModel
from tqdm import tqdm

from config import *
from data_collector.data_collector_funs import DataCollector
from text_processing.text_processing_unit import TextProcessingUnit


class SocialInnovationClassifierPrediction:
    """
    Class to load the four classifiers and predict the projects
    """

    def __init__(self, run_name):
        model_type = "bert"
        model_name = model_type

        use_cuda = False
        # set the model's path
        actor_model_path = f'{model_path}/{run_name}/CriterionActors/'
        output_model_path = f'{model_path}/{run_name}/CriterionOutputs/'
        innov_model_path = f'{model_path}/{run_name}/CriterionInnovativeness/'
        obj_model_path = f'{model_path}/{run_name}/CriterionObjectives/'
        if not os.path.exists(actor_model_path) or \
                not os.path.exists(output_model_path) or \
                not os.path.exists(innov_model_path) or \
                not os.path.exists(obj_model_path):
            print(f'Invalid run name "{run_name}"')
            sys.exit()

        n_gpu = 0 if use_cuda else 1
        eval_batch_size = 16
        use_multiprocessing = False

        self.run_name = run_name
        self.model_name = model_name

        # Loading BERT classification models. ( we will use ONLY CPU for prediction).
        # GPU is faster but it's not available in your server
        self.actor_clf_model = ClassificationModel(model_type, actor_model_path, use_cuda=use_cuda)
        self.actor_clf_model.args.n_gpu = n_gpu
        self.actor_clf_model.args.eval_batch_size = eval_batch_size
        self.actor_clf_model.args.use_multiprocessing = use_multiprocessing
        self.actor_clf_model.args.use_multiprocessing_for_evaluation = False

        self.output_clf_model = ClassificationModel(model_type, output_model_path, use_cuda=use_cuda)
        self.output_clf_model.args.n_gpu = n_gpu
        self.output_clf_model.args.eval_batch_size = eval_batch_size
        self.output_clf_model.args.use_multiprocessing = use_multiprocessing
        self.output_clf_model.args.use_multiprocessing_for_evaluation = False

        self.innov_clf_model = ClassificationModel(model_type, innov_model_path, use_cuda=use_cuda)
        self.innov_clf_model.args.n_gpu = n_gpu
        self.innov_clf_model.args.eval_batch_size = eval_batch_size
        self.innov_clf_model.args.use_multiprocessing = use_multiprocessing
        self.innov_clf_model.args.use_multiprocessing_for_evaluation = False

        self.obj_clf_model = ClassificationModel(model_type, obj_model_path, use_cuda=use_cuda)
        self.obj_clf_model.args.n_gpu = n_gpu
        self.obj_clf_model.args.eval_batch_size = eval_batch_size
        self.obj_clf_model.args.use_multiprocessing = use_multiprocessing
        self.obj_clf_model.args.use_multiprocessing_for_evaluation = False

        self.data_collector = DataCollector()
        self.text_processing_unit = TextProcessingUnit()

    def predict_samples(self, project_ids_path=None, testing_dataset_path= None, insert_finally=True,
                        save_finally=True):
        """
        function to predict the 4 criteria for each project on the testing dataframe
        """

        if testing_dataset_path:
            df = pd.read_csv(testing_dataset_path)
            projects_text = df.text.tolist()
            project_ids = df.Project_id.tolist()
        else:

            if project_ids_path:
                with open(project_ids_path) as rdr:
                    project_ids = [x.strip() for x in rdr.readlines()]
            else:
                project_ids = []

            if not project_ids:
                project_ids = self.data_collector.load_project_ids(self.run_name)

            if not project_ids:
                return 0

            projects_text = []
            for p_id in tqdm(project_ids, "Getting and cleaning projects text"):
                projects_text.append(
                    self.text_processing_unit.clean_text(
                        self.text_processing_unit.shorten_text(
                            self.data_collector.get_project_text(p_id, mongodb_max_pages=10)
                        )
                    )
                )

        all_actor_pred, _ = self.actor_clf_model.predict(projects_text)
        all_output_pred, _ = self.output_clf_model.predict(projects_text)
        all_innov_pred, _ = self.innov_clf_model.predict(projects_text)
        all_obj_pred, _ = self.obj_clf_model.predict(projects_text)
        for idx, sample_text in enumerate(projects_text):
            if not sample_text:
                all_actor_pred[idx] = -1
                all_output_pred[idx] = -1
                all_innov_pred[idx] = -1
                all_obj_pred[idx] = -1

        # saving the predictions into a list of dictionaries. Each dict has 5 keys, as shown below.
        result = [{
            "Project_id": i,
            "CriterionActors": x,
            "CriterionInnovativeness": z,
            "CriterionObjectives": w,
            "CriterionOutputs": y,
            'Social_Innovation_overall': '-1',
            'AnnSource': 'MACHINE',
            'ModelName': self.model_name,
            'expName': self.run_name
        }
            for i, x, y, z, w in zip(
                project_ids,
                all_actor_pred,
                all_output_pred,
                all_innov_pred,
                all_obj_pred
            )
        ]

        df_prediction = pd.DataFrame(result)
        if save_finally:
            df_prediction.to_csv(os.path.join(prediction_path, f'prediction_{self.run_name}.csv'),
                                 index=False, encoding='utf-8')

        if insert_finally:
            self.data_collector.insert_predictions(df_prediction)
