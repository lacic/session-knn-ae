Train models by calling `recsys17_trainer.py`


 
| **Param** | Description | Example |
|----------------|----------------|-----------|
|`--train` |Path to the train data| ` --train ../../data/recsys17/processed/valid_train_14d.csv` |
|`--algo` |Name of the algorithm to train| `--algo pop` |


```
# use the validation set
$ python recsys17_trainer.py --train ../../data/recsys17/processed/valid_train_14d.csv --algo pop

# use the validation set
$ python recsys17_trainer.py --train ../../data/recsys17/processed/train_14d.csv --algo pop
```


