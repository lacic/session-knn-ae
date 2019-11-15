import numpy as np
import pandas as pd
import time
import pickle
import argparse
# set seed to reproduce results with tf and keras
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
# other imports
from keras.layers import Input, Dense, multiply, Lambda
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from keras import backend as K
from keras import optimizers
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--att", 
		type=str, 
		help="Attention (False|True)", 
		nargs='?', 
		default="False", 
		const="False")

	parser.add_argument(
		"--algo", 
		type=str, 
		help="Algorithm to use for inference (ae|dae|vae)", 
		nargs='?', 
		default="ae", 
		const="ae")


	parser.add_argument(
		"--train", 
        type=str, 
        help="Infer from training or test data (False|True)", 
        nargs='?', 
        default="False", 
        const="False")
    
	args = parser.parse_args()




	path =  "../../../data/"
	dataset = "cb12/"

	raw_path = path + dataset + "raw/" 
	interim_path = path + dataset + "interim/"
	processed_path = path + dataset + "processed/"

	model_path = interim_path + "models/"

	# load vocabulary
	unqiue_train_items_df = pd.read_csv(interim_path + 'vocabulary_min_5_app.csv', index_col=0)
	print(unqiue_train_items_df.head(2))
	# create dictionary out of it
	unqiue_train_items_dict = unqiue_train_items_df.to_dict('dict')["item_id"]
	# inverse that item_id is key and index is value
	unqiue_train_items_dict_inv = {v: k for k, v in unqiue_train_items_dict.items()}
	print("Vocabulary size: " + str(len(unqiue_train_items_dict_inv)))

	vocab_size = len(unqiue_train_items_dict_inv)  # size of vocab

	algo = args.algo
	attention = args.att

	data_type = "test"
	if args.train == "True":
		data_type = "train"

	test = pd.read_csv(processed_path + data_type + "_14d.csv", header=0, sep='\t')
	print("Loaded " + processed_path + data_type + "_14d.csv")
	test_session_groups = test.groupby("session_id")
	print(str(len(test_session_groups)) + " sessions to test.")
		
	model_to_load = algo + "_encoder.model"
	if attention == "True":
		model_to_load = "att_" + model_to_load
		
	print("Loading " + model_to_load)
	encoder = pickle.load(open(model_path + model_to_load, 'rb'))

		
	infer_columns = ["session_id", "latent_session_vector", "items"]
	if args.train == "False":
		infer_columns = ["session_id", "latent_session_vector", "input_items", "remaining_items"]
	infer_df = pd.DataFrame(columns = infer_columns)
		
	s_counter = 0
	for session_id, session_group in test_session_groups:
		# vector length = len(unqiue_train_items)
		session_vector = np.zeros((vocab_size,), dtype=int)
		# fill 1s for session items
		max_index = len(session_group) - 1
		item_counter = 0

		for index, row in session_group.iterrows():
			# go from item to item 
			# leave last one for testing
			if item_counter == max_index:
				break
			# make prediction
			item = row["item_id"]
			if item in unqiue_train_items_dict_inv:
				item_index = unqiue_train_items_dict_inv[item]
				session_vector[item_index] = 1
		        
			if args.train == "False":
				# predict for every subsession
				test_in = session_vector.reshape((1, vocab_size))
				latent_session_vec = encoder.predict(test_in)
				# get input and remaining (expected) items
				current_input_set = session_group["item_id"].values[:item_counter+1]
				remaining_test_set = session_group["item_id"].values[item_counter+1:]
				item_counter += 1

				infer_df = infer_df.append({
				    "session_id": session_id,
				    "latent_session_vector": ','.join(map(str, latent_session_vec[0])),
				    "input_items":  ','.join(map(str, current_input_set)),
				    "remaining_items":  ','.join(map(str, remaining_test_set))
				}, ignore_index=True)


		if args.train == "True":
				test_in = session_vector.reshape((1, vocab_size))
				latent_session_vec = encoder.predict(test_in)
				infer_df = infer_df.append({
					"session_id": session_id,
					"latent_session_vector": ','.join(map(str, latent_session_vec[0])),
					"items":  ','.join(map(str, session_group["item_id"].values)),
				}, ignore_index=True)


		s_counter += 1
		if (s_counter % 1000 == 0):
		    print(str(len(test_session_groups) - s_counter) + " test sessions remaining to infer latent vector.")

	infer_path = interim_path + "infer/"
	infer_file_name = algo + "_encoder_" + data_type + ".csv"

	if attention == "True":
		infer_file_name = "att_" + infer_file_name
	infer_df.to_csv(infer_path + infer_file_name, sep='\t', header=False, index=False)
	K.clear_session()
