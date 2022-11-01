from argparse import ArgumentParser

from services.social_innov_prediction import SocialInnovationClassifierPrediction

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-r", "--run_name", dest="run_name",
                        help="Classification run name")
    parser.add_argument("-p", "--projects", dest="project_ids_path", type=str,
                        help="A path to a text file of project IDs, one ID per line.", default=None)
    parser.add_argument("-d", "--dataset", dest="dataset_path", default=None, type=str,
                        help="A CSV file of two columns text and Project_id to be predicted by the classifier.")

    args = parser.parse_args()

    run_name = args.run_name
    project_ids_path = args.project_ids_path
    testing_dataset_path = args.dataset_path

    if not run_name:
        raise ValueError("Missing experiment configuration file")

    # run_name = 'bert-base-cased_2022-10-12_07-36'

    si_classifier = SocialInnovationClassifierPrediction(run_name)

    si_classifier.predict_samples(project_ids_path=project_ids_path,
                                  testing_dataset_path=testing_dataset_path)
