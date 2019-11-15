import numpy as np
import pandas as pd
import Algorithms.baselines as base
import Algorithms.gru4rec as gru4rec
import Algorithms.context_knn as cknn
import Algorithms.SeqContextKNN as scknn
import Algorithms.svmknn as svmknn
import time
import pickle
import argparse

def train_algo(algo, algo_name, train_data):
    print("Training algorithm: " + algo_name)
    
    #train algorithm
    algo.fit( train_data )
    
    # save the model to disk
    filename = "models/recsys17_"  + algo_name + ".model"
    print("Finished training. Storing model to: " + filename)

    pickle.dump(algo, open(filename, 'wb'), protocol=4)
    pass

if __name__ == '__main__':
    train_path = '../../data/recsys17/processed/train_14d.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", 
        type=str, 
        help="Path to train data", 
        nargs='?', 
        default=train_path, 
        const=train_path)
    
    parser.add_argument(
        "--algo", 
        type=str, 
        help="Algorithm to train (pop|spop|iknn|bpr|gru|cknn|scknn|vcknn)", 
        nargs='?', 
        default="all", 
        const="all")
    
    args = parser.parse_args()
    
    train = pd.read_csv(args.train, sep='\t')[['session_id','item_id','created_at']]
    
    train.columns = ['SessionId', 'ItemId', 'Time']
    
    #Reproducing results from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" 
    # on RSC15 (http://arxiv.org/abs/1706.03847)
    
    algs = {}
    
    algs['pop'] = base.Pop()
    algs['iknn'] = base.ItemKNN(lmbd=50, alpha=0.75)
    algs['bpr'] = base.BPR(lambda_session = 0, lambda_item = 0)
    # gru-100-bpr-max-0.5
    algs['gru'] = gru4rec.GRU4Rec(
        loss='bpr-max-0.5',
        final_act='linear', 
        hidden_act='tanh', 
        layers=[100], 
        batch_size=32, 
        dropout_p_hidden=0.0, 
        learning_rate=0.2, 
        momentum=0.5, 
        n_sample=2048, 
        sample_alpha=0, 
        time_sort=True
    )

    algs['cknn'] = cknn.ContextKNN(500, 1000, sampling="random", similarity="cosine", pop_boost = 0)
    algs['scknn'] = scknn.SeqContextKNN(500, 1000, sampling="random", similarity="jaccard", pop_boost = 0)
    algs['vknn'] = svmknn.VMContextKNN(500, 1000, sampling="random", similarity="cosine", pop_boost = 0)
    
    
    if args.algo is "all":
        for key, algo in algs.items():
            train_algo(algo, key, train)
    else:
        algo = algs[args.algo]
        train_algo(algo, args.algo, train)
    # to load again
    # loaded_model = pickle.load(open(filename, 'rb'))
    #
