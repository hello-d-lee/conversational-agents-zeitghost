from urllib.parse import urlparse
from collections import defaultdict
from newspaper import news_pool, Article, Source
import nltk
from typing import Dict, Any, List
import logging
from google.cloud import storage
from google.cloud import bigquery as bq
from google.cloud.bigquery.table import RowIterator
import pandas as pd
from zeitghost.gdelt.Helpers import gdelt_processed_record


#TODO:
#  Optimizations:
#    Generate embeddings during parsing process, then save to to bq
#    Build in "checker" to see if we have already pulled and processed articles in bq table
class GdeltData:
    """
    Gdelt query and parser class
    """
    def __init__(
        self
        , gdelt_data
        , destination_table: str = 'gdelt_actors'
        , project: str = 'cpg-cdp'
        , destination_dataset: str = 'genai_cap_v1'
    ):
        """
        :param gdelt_data: pandas.DataFrame|google.cloud.bigquery.table.RowIterator
            Input data for GDelt processing
        """
        logging.debug('Downloading nltk["punkt"]')
        nltk.download('punkt', "./")
        # BigQuery prepping
        self.__project = project
        self.__bq_client = bq.Client(project=self.__project)
        self.__location = 'us-central1'
        self.__destination_table = destination_table
        self.__destination_dataset = destination_dataset
        self.destination_table_id = f'{self.__destination_dataset}.{self.__destination_table}'
        # Prep for particulars of gdelt dataset

        # Builds self.gdelt_df based on incoming dataset type
        # TODO:
        if type(gdelt_data) is RowIterator:
            logging.debug("gdelt data came in as RowIterator")
            self.gdelt_df = self._row_iterator_loader(gdelt_data)
        elif type(gdelt_data) is pd.DataFrame:
            logging.debug("gdelt data came in as DataFrame")
            self.gdelt_df = self._dataframe_loader(gdelt_data)
        else:
            logging.error("Unrecognized datatype for input dataset")

        self.urllist = self.gdelt_df['url'].map(str).to_list()
        self.domains = [
            {urlparse(url).scheme + "://" + urlparse(url).hostname: url}
            for url in self.urllist
        ]
        self.news_sources = self._prepare_news_sources()
        self.full_source_data = self._parallel_parse_nlp_transform()
        self.chunk_df = pd.DataFrame.from_records(self.full_source_data)
        self.index_data = self._prepare_for_indexing()

    def _dataframe_loader(self, gdelt_df: pd.DataFrame) -> pd.DataFrame:
        logging.debug(f"DataFrame came in with columns: [{','.join(gdelt_df.columns)}]")
        #gdelt_df.fillna(0.0)

        return gdelt_df

    def _row_iterator_loader(self, row_iterator: RowIterator) -> pd.DataFrame:
        """
        This takes a bq iterator and loads data back into a bq table
        """
        # iterate over the bq result page - page size is default 100k rows or 10mb
        holder_df = []
        for df in row_iterator.to_dataframe_iterable():
            logging.debug(f"RowIterator came in with columns: [{','.join(df.columns)}]")
            tmp_df = df
            #tmp_df = tmp_df.fillna(0.0)
            holder_df.append(tmp_df)

        return pd.concat(holder_df)

    def pull_article_text(self, source_url) -> dict[str, Any]:
        """
        Process individual article for extended usage

        :param source_url: str
            url for article to download and process

        :return: dict
        """
        article = Article(source_url)
        article.parse()
        article.nlp()
        return {
            "title": article.title
            , "text": article.text
            , "authors": article.authors
            # , "keywords": article.keywords
            # , "tags" : article.tags
            , "summary": article.summary
            , "publish_date": article.publish_date
            , "url": article.url
            , "language": article.meta_lang
        }

    def _prepare_news_sources(self):
        """
        Given a Gdelt record: group articles by domain, download domain level information.
        For each article: download articles, parse downloaded information, and do simple nlp summarization

        :return: List[Source]
        """
        domain_article = defaultdict(list)
        tmp_list = list()

        # Build {<domain>: [<Article>]} dictionary in preparation
        # for newspaper activity
        for entry in self.domains:
            for domain, article in entry.items():
                domain_article[domain].append(
                    Article(article, fetch_images=False)
                )
        logging.debug("Attempting to fetch domain and article information")
        for domain, articles in domain_article.items():
            # Create Article Source
            tmp_domain = Source(
                url=domain
                , request_timeout=5
                , number_threads=2
            )
            # Download and parse top-level domain
            tmp_domain.download()
            tmp_domain.parse()

            # Build category information
            #tmp_domain.set_categories()
            #tmp_domain.download_categories()
            #tmp_domain.parse_categories()

            # Set articles to Articles built from urllist parameter
            tmp_domain.articles = articles
            tmp_list.append(tmp_domain)
            # Parallelize and download articles, with throttling
            news_pool.set(tmp_list, override_threads=1, threads_per_source=1)
            news_pool.join()

        # Handle articles in each domain
        logging.debug("Parsing and running simple nlp on articles")
        for domain in tmp_list:
            domain.parse_articles()
            for article in domain.articles:
                article.parse()
                article.nlp()

        return tmp_list

    def _parallel_parse_nlp_transform(self) -> List[Dict[str, Any]]:
        """
        Given a list of GDelt records, parse and process the site information.
        Actual data structure for dictionary is a
            list(zeitghost.gdelt.Helpers.gdelt_processed_records)
        :return: List[Dict[str, Any]]
        """
        # Prepare for final return list[dict(<domain/articles>)]
        logging.debug("Preparing full domain and article payloads")
        tmp_list = list()
        for src in self.news_sources:
            tmp = {
                "domain": src.domain
                , "url": src.url
                , "brand": src.brand
                , "description": src.description
                #, "categories": [category.url for category in src.categories]
                , "article_count": len(src.articles)
                , "articles": [
                    {
                        "title": article.title
                        , "text": article.text
                        , "authors": article.authors
                        # , "keywords": article.keywords
                        # , "tags" : article.tags
                        , "summary": article.summary
                        , "publish_date": article.publish_date
                        , "url": article.url
                        , "language": article.meta_lang
                        , "date": self.gdelt_df[self.gdelt_df['url'] == article.url]['SQLDATE'].item() if 'SQLDATE' in self.gdelt_df.columns else ''# self.gdelt_df[self.gdelt_df['url'] == article.url]['date'].item()
                        , "Actor1Name": self.gdelt_df[self.gdelt_df['url'] == article.url]['Actor1Name'].item() if 'Actor1Name' in self.gdelt_df.columns else ''
                        , "Actor2Name": self.gdelt_df[self.gdelt_df['url'] == article.url]['Actor2Name'].item() if 'Actor2Name' in self.gdelt_df.columns else ''
                        , "GoldsteinScale": self.gdelt_df[self.gdelt_df['url'] == article.url]['GoldsteinScale'].item() if 'GoldsteinScale' in self.gdelt_df.columns else ''
                        , "NumMentions": [self.gdelt_df[self.gdelt_df['url'] == article.url]['NumMentions'].item()] if 'NumMentions' in self.gdelt_df.columns else []#if self.gdelt_df[self.gdelt_df['url'] == article.url]['entities'].map(lambda x: [int(e['numMentions']) for e in x]).values else []
                        , "NumSources": self.gdelt_df[self.gdelt_df['url'] == article.url]['NumSources'].item() if 'NumSources' in self.gdelt_df.columns else 0
                        , "NumArticles": self.gdelt_df[self.gdelt_df['url'] == article.url]['NumArticles'].item() if 'NumArticles' in self.gdelt_df.columns else 0
                        , "AvgTone": self.gdelt_df[self.gdelt_df['url'] == article.url]['AvgTone'].item() if 'AvgTone' in self.gdelt_df.columns else 0.0
                        #, "entities_name": self.gdelt_df[self.gdelt_df['url'] == article.url]['entities'].map(lambda x: [str(e['name']) for e in x]).values if 'entities' in self.gdelt_df.columns else []
                        #, "entities_type": self.gdelt_df[self.gdelt_df['url'] == article.url]['entities'].map(lambda x: [str(e['type']) for e in x]).values if 'entities' in self.gdelt_df.columns else []
                        #, "entities_avgSalience": self.gdelt_df[self.gdelt_df['url'] == article.url]['entities'].map(lambda x: [float(e['avgSalience']) for e in x]).values if 'entities' in self.gdelt_df.columns else []
                    } for article in src.articles
                ]
            }
            tmp_list.append(tmp)

        return tmp_list

    def _reduced_articles(self) -> List[Dict[str, Any]]:
        """
        Given a list of GDelt records, parse and process the site information.
        Actual data structure for dictionary is a
            list(zeitghost.gdelt.Helpers.gdelt_reduced_articles)
        :return: List[Dict[str, Any]]
        """
        # Prepare for final return list[dict(<domain/articles>)]
        logging.debug("Preparing full domain and article payloads")
        tmp_list = list()
        for src in self.news_sources:
            for article in src.articles:
                row = self.gdelt_df[self.gdelt_df['url'] == article.url]
                tmp = {
                    "title": article.title
                    , "text": article.text
                    , "article_url": article.url
                    , "summary": article.summary
                    , "date": str(row['date'].values)
                    , "entities_name": row['entities'].map(lambda x: [str(e['name']) for e in x]).values
                    , "entities_type": row['entities'].map(lambda x: [str(e['type']) for e in x]).values
                    , "entities_numMentions": row['entities'].map(lambda x: [int(e['numMentions']) for e in x]).values
                    , "entities_avgSalience": row['entities'].map(lambda x: [float(e['avgSalience']) for e in x]).values
                }
                tmp_list.append(tmp)

        return tmp_list

    def _prepare_for_indexing(self):
        """
        Reduces the larger Gdelt and newspaper download into a more compact payload tuned for indexing

        :return: pandas.DataFrame
        """
        logging.debug("Reducing full payload into what Chroma expects for indexing")
        final_return_df = pd.DataFrame.from_dict(self.full_source_data)
        pre_vector_df = final_return_df[['articles', 'url']].copy()

        pre_vector_df.columns = ['text', 'url']

        pre_vector_df['text'] = str(pre_vector_df['text'])

        pre_vector_df['text'].astype("string")
        pre_vector_df['url'].astype("string")

        return pre_vector_df

    def write_to_gcs(self, output_df: pd.DataFrame, bucket_name: str):
        """
        Output article information to a cloud storage bucket

        :param output_df: pandas.DataFrame
            Input dataframe to write out to GCS
        :param bucket_name: str
            Bucket name for writing to

        :return: str
        """
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        if not bucket.exists():
            bucket.create()
        blob_name = "articles/data.json"

        bucket.blob(blob_name).upload_from_string(
            output_df.to_json(index=False)
            , 'text/json'
        )

        return f"gs://{bucket_name}/{blob_name}"

    def write_to_bq(self) -> str:
        self.chunk_df.to_gbq(self.destination_table_id
                             , project_id=self.__project
                             , if_exists='append'
                             , table_schema=gdelt_processed_record
                             )

        return f"{self.__project}:{self.destination_table_id}"