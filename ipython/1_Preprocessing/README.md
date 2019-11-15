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

A detailed description for the recsys dataset can be found on the [RecSys Challenge Webseite](http://www.recsyschallenge.com/2017/).

The RecSys interaction.csv file has the following fields:
- user_id [int]
- item_id [int]
- interaction_type [int]
- created_at [int] (timestamp since epoch in seconds)

The RecSys items.csv file has the following fields:
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

The Studo interaction.csv file has the following fields:
- \<empty\> [int] (id of interaction)
- item_id [string]
- session_id [string]
- interaction_type [int]
- user_id [string]
- created_at [int] (timestamp since epoch in nanoseconds)

The Studo items.csv has the following fields:
- item_id
- job_begins_now
- job_country
- job_effort
- job_language
- job_state
- labels
- tags
