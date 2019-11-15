# Hyperparameter optimization

Hyperparameters are optimized for each dataset by evaluating the **nDCG** metric using the **validation test set**. 

## sKNN

Default: sampling = recent, similarity = jaccard, pop_boost = false

| |sampling | similarity | pop_boost | RecSys17 | Studo |
|----------------|----------------|-----------|-----------|------|------|
|1|recent| jaccard | false |||
|2|recent| jaccard | true |||
|3|recent| cosine | false |||
|4|recent| cosine | true |||
|5|random| jaccard | false |||
|6|random| jaccard | true |||
|7|random| cosine | false |||
|8|random| cosine | true |||


## S-sKNN

Default: sampling = recent, similarity = jaccard, pop_boost = false

| |sampling | similarity | pop_boost | RecSys17 | Studo |
|----------------|----------------|-----------|-----------|------|------|
|1|recent| jaccard | false |||
|2|recent| jaccard | true |||
|3|recent| cosine | false |||
|4|recent| cosine | true |||
|5|random| jaccard | false |||
|6|random| jaccard | true |||
|7|random| cosine | false |||
|8|random| cosine | true |||

## ItemKNN

Defualt: lmbd = 20, alpha = 0.5

| |lmbd | alpha | RecSys17 | Studo |
|----------------|----------------|-----------|-------|-------|
|1|20| 0.25 | | |
|2|20| 0.5 || |
|3|20| 0.75 || |
|4|50| 0.25 || |
|5|50| 0.5 || |
|6|50| 0.75 || |
|7|80| 0.25 | | |
|8|80| 0.5 | | |
|9|80| 0.75 | | |

## BPR

Default: lambda_session = 0.0, lambda_item = 0.0

| |lambda_session | lambda_item  | RecSys17 | Studo |
|----------------|----------------|-----------|-------|-------|
|1|0.0| 0.0 | | |
|2|0.0| 0.25 || |
|3|0.0| 0.5 || |
|4|0.25| 0.0 || |
|5|0.25| 0.25 || |
|6|0.25| 0.5 || |
|7|0.5| 0.0 || |
|8|0.5| 0.25 || |
|9|0.5| 0.5 || |

## GRU4Rec

Default: loss = bpr-max-0.5, layers = [1000]

| |loss | layers  | RecSys17 | Studo |
|----------------|----------------|-----------|-------|-------|
|1|top1-max| [100] | | |
|2|top1-max| [100,100] || |
|3|top1-max| [1000] || |
|4|top1-max| [1000,1000] || |
|5|bpr-max-0.5| [100]  || |
|6|bpr-max-0.5| [100,100]  || |
|7|bpr-max-0.5| [1000]  || |
|8|bpr-max-0.5| [1000,1000]  || |