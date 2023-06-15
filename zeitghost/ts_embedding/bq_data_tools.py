from google.cloud import bigquery
import pandas as pd

PROJECT_ID = 'cpg-cdp'
TABLE_ID = 'makeupcosmetics_10054_unitedstates_2840'
DATASET = 'trends_data'

bqclient = bigquery.Client(
    project=PROJECT_ID,
    # location=LOCATION
)

def get_term_set(project_id=PROJECT_ID,
                dataset=DATASET,
                table_id=TABLE_ID):
    '''
    Simple function to get the unique, sorted terms in the table
    '''
    query = f"""
    SELECT distinct
    term
    FROM `{project_id}.{dataset}.{table_id}`
    order by 1
    """

    df = bqclient.query(query = query).to_dataframe()
    return df["term"].to_list()


def pull_term_data_from_bq(term: tuple = ('mascara', 'makeup'), 
                           project_id=PROJECT_ID,
                           dataset=DATASET,
                           table_id=TABLE_ID):
    '''
    pull terms based on `in` sql clause from term
    takes a tuple of terms (str) and produces pandas dataset
    '''
    query = f"""
    SELECT
    cast(date AS DATE FORMAT 'YYYY-MM-DD') as date_field,
    term,
    score
    FROM `{project_id}.{dataset}.{table_id}`
    WHERE
    term in {term}
    order by term, 1
    """

    df = bqclient.query(query = query).to_dataframe()
    return df