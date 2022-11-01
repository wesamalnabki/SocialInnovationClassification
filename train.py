import os
from argparse import ArgumentParser

from services.social_innov_trainer import SocialInnovationClassifierTraining

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--topx", dest="topx", default=None,
                        help="top X files to select from the unlabeled projects")

    parser.add_argument("-d", "--dataset", dest="dataset", default=None,
                        help="A CSV to be used for training and testing")

    args = parser.parse_args()
    dataset_path = args.dataset

    # dataset_path = 'dataset/training_bert-base-cased_2022-10-11_23-25_df.csv'
    trainer = SocialInnovationClassifierTraining()
    ds = trainer.load_training_set(dataset_path, top_x=None, save_finally=True)
    trainer.training_classifiers(ds)
