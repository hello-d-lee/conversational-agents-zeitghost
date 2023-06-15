from sklearn.preprocessing import MinMaxScaler
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures
import os
import pandas as pd
import numpy as np
from .bq_data_tools import pull_term_data_from_bq
from decimal import Decimal



# https://stackoverflow.com/questions/434287/how-to-iterate-over-a-list-in-chunks

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

SLIDING_WINDOW_SIZE = 30 #n months for chunking - complete examples only used
STEP = 1 #step - default to 1


def write_embeddings_to_disk(term_chunk, filename='data/ts_embeddings.jsonl'):
    '''
    this funciton takes a chunk of n_terms (see chunker for input)
    and writes to `filename` a jsonl file compliant with 
    matching engine
    '''
    term_data = pull_term_data_from_bq(tuple(term_chunk))
    #run through by term
    for term in term_chunk:
        # emb_pair = get_feature_embedding_for_window(term_data[term_data.term == term], term)
        # ts_emedding_pairs.append(emb_pair)
        wdf = windows(term_data[term_data.term == term], SLIDING_WINDOW_SIZE, STEP)
        for window, new_df in wdf.groupby(level=0):
            # print(window, new_df)
            if new_df.shape[0] == SLIDING_WINDOW_SIZE: #full examples only
                emb_pair = get_feature_embedding_for_window(new_df, term)
                label, emb = emb_pair
                formatted_emb = '{"id":"' + str(label) + '","embedding":[' + ",".join(str(x) for x in list(emb)) + ']}'
                with open(filename, 'a') as f:
                    f.write(formatted_emb)
                    f.write("\n")
                f.close()
                
def windows(data, window_size, step):
    '''
    creates slices of the time series used for 
    creating embeddings
    '''
    r = np.arange(len(data))
    s = r[::step]
    z = list(zip(s, s + window_size))
    f = '{0[0]}:{0[1]}'.format
    g = lambda t: data.iloc[t[0]:t[1]]
    return pd.concat(map(g, z), keys=map(f, z))

def get_feature_embedding_for_window(df, term):
    '''
    this takes a df with schema of type `date_field` and `score` to create an embeddding
    takes 30 weeks of historical timeseries data
    '''
    ts_name = f"{term}_{str(df.date_field.min())}_{str(df.date_field.max())}"
    scaler=MinMaxScaler()
    df[['score']] = scaler.fit_transform(df[['score']])
    scores = df[['score']].values.tolist()
    flat_values = [item for sublist in scores for item in sublist]
    df = df.rename(columns={"date_field":"time"}) 
    ts_df = pd.DataFrame({'time':df.time, 
                          'score':flat_values})
    ts_df.drop_duplicates(keep='first', inplace=True)  

    # Use Kats to extract features for the time window
    try:
        if not (len(np.unique(ts_df.score.tolist())) == 1 \
            or len(np.unique(ts_df.score.tolist())) == 0):
            timeseries = TimeSeriesData(ts_df)
            features = TsFeatures().transform(timeseries)
            feature_list = [float(v) if not pd.isnull(v) else float(0) for _, v in features.items()]
            if Decimal('Infinity') in feature_list or Decimal('-Infinity') in feature_list:
                return None
            return (ts_name, feature_list)
    except np.linalg.LinAlgError as e:
        print(f"Can't process {ts_name}:{e}")
    return None

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))