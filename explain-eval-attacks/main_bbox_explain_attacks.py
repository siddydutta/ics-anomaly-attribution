"""

   Copyright 2023 Lujo Bauer, Clement Fung

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import numpy as np
import pdb
import pickle

import os
import sys
sys.path.append('..')

# Ignore ugly futurewarnings from np vs tf.
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf

from sklearn.model_selection import train_test_split
from data_loader import load_train_data, load_test_data
from main_train import load_saved_model

from live_bbox_explainer.score_generator import lime_score_generator, shap_score_generator, lemna_score_generator

from utils import attack_utils, utils

np.set_printoptions(suppress=True)

HOUR = 2000

def explain_true_position(event_detector, lookup_name, attack_idx, attacks, Xtest, method='MSE', expl=None, num_samples=1):

	history = event_detector.params['history']
	nsensors = Xtest.shape[1]

	if method in ['LIME', 'SHAP', 'LEMNA']:
		full_scores = np.zeros((num_samples, history, nsensors))
	else:
		full_scores = np.zeros((num_samples, nsensors))

	# TODO: modified to start from history for now here
	# att_start = attacks[attack_idx][0] + history
	att_start = attacks[attack_idx][0]

	print('============================')

	for i in range(num_samples):

		capture_idx = att_start + i
		capture_start = capture_idx - history - 1

		Xinput = np.expand_dims(Xtest[capture_start:capture_start+history], axis=0)
		Yinput = np.expand_dims(Xtest[capture_idx], axis=0)

		print(f'For attack {attack_idx}: capture {att_start} + {i}')

		if method == 'LEMNA':
			exp_output = lemna_score_generator(event_detector, Xinput, Yinput)
		elif method == 'SHAP':
			exp_output = shap_score_generator(event_detector, expl, Xinput, Yinput)
		elif method == 'LIME':
			exp_output = lime_score_generator(event_detector, expl, Xinput, Yinput)
		else:
			exp_output = (event_detector.predict(Xinput) - Yinput)**2

		full_scores[i] = exp_output

	pickle.dump(full_scores, open(f'explanations-dir/explain23-pkl/explanations-{method}-{lookup_name}-{attack_idx}-true{num_samples}.pkl', 'wb'))

	return

def explain_detect(event_detector, lookup_name, attack_idx, attacks, Xtest, detection_points, method='MSE', expl=None, num_samples=1):

	history = event_detector.params['history']
	nsensors = Xtest.shape[1]

	if attack_idx in detection_points:

		if method in ['LIME', 'SHAP', 'LEMNA']:
			full_scores = np.zeros((num_samples, history, nsensors))
		else:
			full_scores = np.zeros((num_samples, nsensors))

		# TODO: modified to start from beginning, since detect_idx accounts for history
		att_start = attacks[attack_idx][0]
		detect_idx = detection_points[attack_idx]

		print('============================')

		for i in range(num_samples):

			capture_idx = att_start + detect_idx + i
			capture_start = capture_idx - history - 1

			Xinput = np.expand_dims(Xtest[capture_start:capture_start+history], axis=0)
			Yinput = np.expand_dims(Xtest[capture_idx], axis=0)

			print(f'For attack {attack_idx}: capture {att_start} + {detect_idx + i}')

			if method == 'LEMNA':
				exp_output = lemna_score_generator(event_detector, Xinput, Yinput)
			elif method == 'SHAP':
				exp_output = shap_score_generator(event_detector, expl, Xinput, Yinput)
			elif method == 'LIME':
				exp_output = lime_score_generator(event_detector, expl, Xinput, Yinput)
			else:
				exp_output = (event_detector.predict(Xinput) - Yinput)**2

			full_scores[i] = exp_output

		pickle.dump(full_scores, open(f'explanations-dir/explain23-detect-pkl/explanations-{method}-{lookup_name}-{attack_idx}-detect{num_samples}.pkl', 'wb'))
	
	else:

		print(f'Attack {attack_idx} was missed')

	return

def parse_arguments():

	parser = utils.get_argparser()

	parser.add_argument("attack",
		help="Which attack to explore?",
		type=int)

	parser.add_argument("--explain_params_methods",
        choices=['MSE', 'LIME', 'SHAP', 'LEMNA'],
        default='MSE')

	parser.add_argument("--num_samples",
		default=5,
		type=int,
		help="Number of samples")

	return parser.parse_args()

if __name__ == "__main__":

	import os
	os.chdir('..')
	args = parse_arguments()
	model_type = args.model
	dataset_name = args.dataset
	exp_method = args.explain_params_methods

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
	os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

	run_name = args.run_name
	config = {} # type: Dict[str, str]
	utils.update_config_model(args, config, model_type, dataset_name)
	model_name = config['name']
	event_detector = load_saved_model(model_type, f'models/{run_name}/{model_name}.json', f'models/{run_name}/{model_name}.h5')
	history = event_detector.params['history']
	attacks, _ = attack_utils.get_attack_indices(dataset_name)
	attack_idx = args.attack

	Xtest, _, _ = load_test_data(dataset_name)
	lookup_name = f'{model_name}-{run_name}'
	num_samples = args.num_samples

	if exp_method == 'SHAP':
		import shap
		Xfull, sensor_cols = load_train_data(dataset_name)
		baseline = utils.build_baseline(Xfull, history, method=exp_method)
		expl = shap.DeepExplainer(event_detector.inner, baseline)
	elif exp_method == 'LIME':
		import lime
		Xfull, sensor_cols = load_train_data(dataset_name)
		baseline = utils.build_baseline(Xfull, history, method=exp_method)
		expl = lime.lime_tabular.RecurrentTabularExplainer(baseline,
								feature_names=np.arange(baseline.shape[2]),
								verbose=False,
								mode='regression')
	else:
		expl = None

	# Practical detection
	# detection_points = pickle.load(open(f'meta-storage/{lookup_name}-detection-points.pkl', 'rb'))
	# model_detection_points = detection_points[lookup_name]
	# explain_detect(event_detector, lookup_name, attack_idx, attacks, Xtest,
	# 		model_detection_points,
	# 		method=exp_method,
	# 		expl=expl,
	# 		num_samples=num_samples)

	# Ideal detection
	explain_true_position(event_detector, lookup_name, attack_idx, attacks, Xtest,
			method=exp_method,
			expl=expl,
			num_samples=num_samples)

	print('All done!')
