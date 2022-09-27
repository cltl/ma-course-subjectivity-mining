from ml_pipeline import experiment
import argparse

parser = argparse.ArgumentParser(description='run classifier on data')
parser.add_argument('--task', dest='task', default="vua_format")
parser.add_argument('--data_dir', dest='data_dir', default="./data/gibert/")
parser.add_argument('--pipeline', dest='pipeline', default='naive_bayes_counts')
parser.add_argument('--print_predictions', dest='print_predictions', default=False)
parser.add_argument('--add_extra_data', dest='add_extra', default=False)
args = parser.parse_args()

# experiment.run(args.task, args.data_dir, args.pipeline, args.print_predictions, add_extra_training=args.add_extra)
experiment.my_run('vua_format', './data/trac2018_VUA/', 'svm_libsvc_counts', False)