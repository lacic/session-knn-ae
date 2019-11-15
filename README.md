
## Using Autoencoders for Session-based Job Recommendations

This repository accompanies the paper **Using Autoencoders for Session-based Job Recommendations** for **UMUAI**: Special Issue on Session-based and Sequential Recommender Systems.

| Folder | Description |
| ------ | ------ |
| [Notebooks](ipython) | Contains python code to train and predict jobs with baseline approaches from related work.  |
| [Plots](plots) | Contains .pdf plots that are used for the paper. |
| [Main code](main_code) | Implementation of the three autoencoder approaches as well the Bayes baseline. Additionally contains the logic to evaluate the predictions made by all the approaches wrt. accuracy and non-accuracy metrics. |

## Notebooks

### Setup

To run the notebooks, you first need to setup the conda environment:

```
conda create -n py3 python=3.6
source activate py3
conda install matplotlib
conda install pandas
conda install nltk
conda install theano
conda install keras
```
### Init dataset folders

For every dataset, use the `generate_folders.py` script to initialize the data set structure that is referenced in the notebooks:

```
python generate_folders.py recsys17
python generate_folders.py cb12
``` 

This repository contains the code to reproduce the experiments on the Recsys Challenge 2017 dataset. Download the [RecSys17](http://www.recsyschallenge.com/2017/#dataset) and [CareerBuilder12](https://www.kaggle.com/c/job-recommendation/data) datasets and put them in the `data/\<dataset\>/raws/` folder. Use the `ipython/1_Preprocessing/` notebooks to create the train and test files. The code to create the dataset plots in the paper can be found in the `ipython/2_Data_Analysis/` folder.

##### Explanation of the data folder structure

- raw: contains the raw data, put the dataset there (items.csv and interactions.csv)
- interim: contains processed data (interactions.csv, train_session_interaction_vector.csv and vocabulary.csv)
- interim/infer: contains the inferred session representations by autoencoders as csv
- interim/models: contains the trained autoencoder models
- interim/predict: contains the prediction csv files for autoencoders
- interim/predict/base: contains the prediction csv files for baseline algorithms
- interim/predict/hyperparam: contains the prediction csv files for hyperparameter optimization
- processed: contains training, validation and test sets (train_14d.csv, valid_train_14d.csv, valid_test_14d.csv, test_14d.csv)
- processed/eval: evaluation results for autoencoders
- processed/eval/base: evaluation results for baseline algorithms
- processed/eval/hyperparam: evaluation results for hyperparameter optimization

All eval folders contain the subfolders `all` and `next` for the two prediction scenarios.

### Compile Main Code
The main code to (i) evaluate the generated predictions, (ii) create predictions for the Bayes baseline and (iii) generate predictions for the latent session vectors inferred from the autoencoders is written in Java 8 and is contained in the `main_code` folder.  

```
cd main_code
mvn clean install
```

##### Running modes

| Mode | Description |
| ------ | ------ |
| **predict-bayes** | Generates predictions using the Bayes baseline |
| **predict** | Predicts for each test session the next jobs to interact with |
| **eval** | Evaluates the generated prediction file with respect to accuracy and non-accuracy metrics |

### Baseline Hyperparameter Optimization
 
 **Train:**
 Use the `ipython/3_Training_Predicting/*_hyperparameter_opt_train.ipynb` notebooks to train the models with different hyperparameter combinations. 
 
 Training these models expects a folder structure `ipython/3_Training_Predicting/models/valid` to exist.
 
 **Predict:**
Use the `ipython/3_Training_Predicting/*_predictor.ipynb` notebooks to generate predictions for the models with different hyperparameter combinations. 

**Evaluate:**
Evaluate the **eval** flag and theprediction files by using the main code:
```
cd main_code/test-predictor
java -jar target/test-predictor-1.0-SNAPSHOT-bin.jar conf/cb12/eval_hyp_opt_cb12.properties eval
```
Depending on the dataset, it is possible that the **-Xms** and **-Xmx** flags need to be set. The properties file may also be need to be adjusted depending on the used data structure. 

**Plot:**
Use the `ipython/5_Eval/hyp_opt_eval.ipynb` to analyze and plot the hyperparameter optimization results.
 
 ### Baselines
 
  **Train:**
 Use the `*_trainer.ipynb` or the adjust the corresponding python scripts for the dataset to train the baseline models.  Training these models expects a folder structure `ipython/3_Training_Predicting/models/` to exist.
 
 **Predict:**
Use the `ipython/3_Training_Predicting/*_predictor.ipynb` notebooks to generate predictions for the models with different hyperparameter combinations. 

**Evaluate:**
Use the **eval** flag and the configuration files for the baselines:
```
cd main_code/test-predictor
java -jar target/test-predictor-1.0-SNAPSHOT-bin.jar conf/cb12/eval_base_cb12.properties eval
```
Again, depending on the dataset, it is possible that the **-Xms** and **-Xmx** flags need to be set. The properties file may also be need to be adjusted depending on the used data structure. 

 ### Bayes baseline
Use the Java code with the **predict-bayes** flag to generate the predictions for the Bayes baseline
```
java -jar target/test-predictor-1.0-SNAPSHOT-bin.jar conf/cb12/eval_base_cb12.properties predict-bayes
```

### Autoencoder variants

To train the autoencoder variants and infer session vectors use the `interactions_aes_cb12` and `item_aes_cb12` notebooks (i.e., or the respective recsys17 ones for the RecSys Challange 2017 dataset).

 **Predict:**
To create predictions, use the Java code with the **predict** flag:

```
java -jar target/test-predictor-1.0-SNAPSHOT-bin.jar conf/cb12/eval_ae_cb12_int.properties predict
```
**Evaluate:**
Use the **eval** flag and the configuration files for the autoencoder approaches:
```
cd main_code/test-predictor
java -jar target/test-predictor-1.0-SNAPSHOT-bin.jar conf/cb12/eval_ae_cb12_int.properties eval
```

**Plot:**
Use the `ipython/5_Eval/cb12_eval.ipynb` or `ipython/5_Eval/recsys17_eval.ipynb` notebook to analyze and plot all evaluation results.

### Embedding Analysis

Use the `ipython/4_Embedding_Analysis/` notebooks to create the t-SNE plots for the embedding analysis.

