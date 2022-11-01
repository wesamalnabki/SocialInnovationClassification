from argparse import ArgumentParser

from services.social_innov_learning_curve import SocialInnovationLearningCurve

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--step_size", dest="step_size", default=50, help="Training dataset step size")

    parser.add_argument("-d", "--dataset", dest="dataset", default=None,
                        help="A CSV to be used for building the learning curve")

    args = parser.parse_args()
    step_size = int(args.step_size)
    dataset_path = args.dataset

    lc = SocialInnovationLearningCurve()
    lc.build_learning_curve(dataset_path, step_size=step_size)
