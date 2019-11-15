# 1. Preprocessing

## Input

You need an interactions.csv and a items.csv file with tabs as a seperator in the raw folder. For the RecSys dataset 

## Output

In interim folder:
- interactions.csv
- train_session_interaction_vector.csv
- vocabulary.csv

In processed folder:
- test-14d.csv
- train-14d.csv
- valid-test-14d.csv
- valid-train-14d.csv


## Datasets

A detailed description for the RecSys17 dataset can be found on the [RecSys Challenge Webseite](http://www.recsyschallenge.com/2017/).

The RecSys interaction.csv file has the following fields:
- user_id [int]
- item_id [int]
- interaction_type [int]
- created_at [int] (timestamp since epoch in seconds)

The RecSys17 items.csv file has the following fields:
- item_id
- title # unused
- career_level
- discipline_id
- industry_id
- country
- is_payed
- region
- latitude # unused
- longitude # unused
- employment
- tags # unused
- created_at # unused

A detailed description for the CareerBuilder12 dataset can be found on the [Kaggle Website](https://www.kaggle.com/c/job-recommendation/data)

The CareerBuilder12 interaction.csv file has the following fields (after transformation):
- user_id [int]
- created_at [int] (timestamp since epoch in seconds)
- item_id [int]
- interaction_type [int] (always has the value 0)
- session_id [int] (present in interim interactions, but not raw)

The CareerBuilder12 items.csv has the following fields (after transformation):
city	state	country	zip5	ReqTopic	DescTopic	TitTopic
- item_id
- city
- state
- country
- zip5
- ReqTopic
- DescTopic
- TitTopic
