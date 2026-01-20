import os
import argparse
parser = argparse.ArgumentParser(description='TensorBoard log directory')
parser.add_argument('--logdir', type=str, required=True, help='Path to the log directory')
args = parser.parse_args()

logdir = args.logdir

abs_path = os.path.abspath('.')

log_path = os.path.join(abs_path, logdir)

os.system(f'tensorboard --logdir={log_path}')
