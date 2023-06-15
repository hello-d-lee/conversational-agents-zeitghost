from google.cloud import bigquery
from google.cloud.bigquery import QueryJob
from google.cloud.bigquery.table import RowIterator
import pandas as pd


class BigQueryAccessor:
    """
    Interface for querying BigQuery.
    """
    def __init__(self
                 , project_id
                 , gdelt_project_id='gdelt-bq'
                 , gdelt_dataset_id='gdeltv2'
                 , gdelt_table_name='events'):
        """
        :param gdelt_project_id: str
            Project ID for building BigQuery client
        :param gdelt_dataset_id: str
            Dataset ID for building BigQuery Client
        :param gdelt_table_name: str
            Table name for building BigQuery Client
        """
        self.project_id = project_id
        self.gdelt_project_id = gdelt_project_id
        self.gdelt_dataset_id = gdelt_dataset_id
        self.gdelt_table_name = gdelt_table_name
        self.client = bigquery.Client(project=self.project_id)

    def _query_bq(self, query_string: str) -> QueryJob:
        """

        :param query_string: str
            Full SQL query string to execute against BigQuery

        :return: google.cloud.bigquery.job.QueryJob
        """
        return self.client.query(query_string)

    def get_records_from_sourceurl(self, source_url) -> RowIterator:
        """
        Retrieve article record from Gdelt dataset given a source_url

        :param source_url: str

        :return: google.cloud.bigquery.table.RowIterator
        """
        query = f"""
        SELECT
          max(SQLDATE) as SQLDATE,
          max(Actor1Name) as Actor1Name,
          max(Actor2Name) as Actor2Name,
          avg(GoldsteinScale) as GoldsteinScale,
          max(NumMentions) as NumMentions,
          max(NumSources) as NumSources,
          max(NumArticles) as NumArticles,
          avg(AvgTone) as AvgTone,
          SOURCEURL as SOURCEURL,
          SOURCEURL as url
        FROM `{self.gdelt_project_id}.{self.gdelt_dataset_id}.{self.gdelt_table_name}`
        WHERE lower(SOURCEURL) like '%{source_url}%'
        GROUP BY SOURCEURL
        """

        return self._query_bq(query_string=query).result()

    def get_records_from_sourceurl_df(self, source_url):
        """
        Retrieve article record from Gdelt dataset given a source_url

        :param source_url: str

        :return: pandas.DataFrame
        """
        response = self.get_records_from_sourceurl(source_url)

        return response.to_dataframe()

    def get_records_from_actor_keyword(self
                                       , keyword: str
                                       , min_date: str = "2023-01-01"
                                       , max_date: str = "2023-05-30"
                                      ) -> RowIterator:
        """
        Retrieve BQ records given input keyword

        :param keyword: str
            Keyword used for filtering actor names

        :return: google.cloud.bigquery.table.RowIterator
        """

        query = f"""
        SELECT
          max(SQLDATE) as SQLDATE,
          PARSE_DATE('%Y%m%d', CAST(max(SQLDATE) AS STRING)) as new_date,
          max(Actor1Name) as Actor1Name,
          max(Actor2Name) as Actor2Name,
          avg(GoldsteinScale) as GoldsteinScale,
          max(NumMentions) as NumMentions,
          max(NumSources) as NumSources,
          max(NumArticles) as NumArticles,
          avg(AvgTone) as AvgTone,
          SOURCEURL as SOURCEURL,
          SOURCEURL as url
        FROM `{self.gdelt_project_id}.{self.gdelt_dataset_id}.{self.gdelt_table_name}`
        WHERE lower(SOURCEURL) != 'unspecified' 
        AND
            (
                REGEXP_CONTAINS(LOWER(Actor1Name),'{keyword.lower()}')
                OR REGEXP_CONTAINS(LOWER(Actor2Name), '{keyword.lower()}')
            )
        AND PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) >= "{min_date}"
        AND PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) <= "{max_date}"
        GROUP BY url
        """

        return self._query_bq(query_string=query).result()

    def get_records_from_actor_keyword_df(self
                                          , keyword: str
                                          , min_date: str = "2023-01-01"
                                          , max_date: str = "2023-05-30"
                                         ) -> pd.DataFrame:
        """
        Retrieves BQ records given input actor info

        :param keyword: str

        :return: pandas.DataFrame
        """
        response = self.get_records_from_actor_keyword(keyword, min_date, max_date)

        return response.to_dataframe()

    def get_term_set(self
                     , project_id='cpg-cdp'
                     , dataset='bigquery-public-data'
                     , table_id='top_terms'
                     ) -> RowIterator:
        """
        Simple function to get the unique, sorted terms in the table

        :param project_id: str
            project_id that holds the dataset.
        :param dataset: str
            dataset name that holds the table.
        :param table_id: str
            table name

        :return: google.cloud.bigquery.table.RowIterator
        """

        query = f"""
        SELECT distinct
        term
        FROM `{project_id}.{dataset}.{table_id}`
        order by 1
        """

        return self._query_bq(query_string=query).result()

    def get_term_set_df(self
                        , project_id='cpg-cdp'
                        , dataset='trends_data'
                        , table_id='makeupcosmetics_10054_unitedstates_2840'
                        ) -> list:
        """
        Simple function to get the unique, sorted terms in the table

        :param project_id: str
            project_id that holds the dataset.
        :param dataset: str
            dataset name that holds the table.
        :param table_id: str
            table name

        :return: pandas.DataFrame
        """
        df = self.get_term_set(project_id, dataset, table_id).to_dataframe()

        return df["term"].to_list()

    def pull_term_data_from_bq(self 
                               , term: tuple = ('mascara', 'makeup')
                               , project_id='bigquery-public-data'
                               , dataset='google_trends'
                               , table_id='top_rising_terms'
                               ) -> RowIterator:
        """
        Pull terms based on `in` sql clause from term
        takes a tuple of terms (str) and produces pandas dataset

        :param term: tuple(str)
            A tuple of terms to query for
        :param project_id: str
            project_id that holds the dataset.
        :param dataset: str
            dataset name that holds the table.
        :param table_id: str
            table name

        :return: google.cloud.bigguqery.table.RowIterator
        """
        query = f"""
        SELECT
        week,
        term,
        rank
        FROM `{project_id}.{dataset}.{table_id}`
        WHERE
        lower(term) in {term}
        order by term, 1
        """

        return self._query_bq(query_string=query).result()

    def pull_term_data_from_bq_df(self
                                  , term: tuple = ('mascara', 'makeup')
                                  , project_id='bigquery-public-data'
                                  , dataset='google_trends'
                                  , table_id='top_rising_terms'
                                  ) -> pd.DataFrame:
        """
        Pull terms based on `in` sql clause from term
        takes a tuple of terms (str) and produces pandas dataset

        :param term: tuple(str)
            A tuple of terms to query for
        :param project_id: str
            project_id that holds the dataset.
        :param dataset: str
            dataset name that holds the table.
        :param table_id: str
            table name

        :return: pandas.DataFrame
        """
        result = self.pull_term_data_from_bq(term, project_id, dataset, table_id)

        return result.to_dataframe()
    
    def pull_regexp_term_data_from_bq(self
                               , term: str
                               , project_id='bigquery-public-data'
                               , dataset='google_trends'
                               , table_id='top_rising_terms'
                               ) -> RowIterator:
        """
        Pull terms based on `in` sql clause from term
        takes a tuple of terms (str) and produces pandas dataset

        :param term: tuple(str)
            A tuple of terms to query for
        :param project_id: str
            project_id that holds the dataset.
        :param dataset: str
            dataset name that holds the table.
        :param table_id: str
            table name

        :return: google.cloud.bigguqery.table.RowIterator
        """
        query = f"""
        SELECT
        week,
        term,
        rank
        FROM `{project_id}.{dataset}.{table_id}`
        WHERE (
                  REGEXP_CONTAINS(LOWER(term), r'{term}')
              )
        order by term
        """

        return self._query_bq(query_string=query).result()
    
    def get_entity_from_geg_full(self
                                 , entity: str
                                 , min_date: str = "2023-01-01"
                                 ) -> RowIterator:
        entity_lower = entity.lower()
    
        query = f"""
                    WITH
                  entities AS (
                  SELECT
                    b.*,
                    url
                  FROM
                    `{self.gdelt_project_id}.{self.gdelt_dataset_id}.geg_gcnlapi` AS a,
                    UNNEST(a.entities) AS b
                  WHERE
                    LOWER(b.name) LIKE '%{entity_lower}%'
                    AND DATE(date) >= '{min_date}' )
                SELECT
                  *
                FROM
                  `gdelt-bq.gdeltv2.geg_gcnlapi` a
                INNER JOIN
                  entities AS b
                ON
                  a.url = b.url
                WHERE
                  DATE(date) >= '{min_date}'
              """

        return self._query_bq(query_string=query).result()

    def get_entity_from_geg_full_df(self
                                    , entity: str
                                    , min_date: str = "2023-01-01"):
        result = self.get_entity_from_geg_full(entity, min_date)

        return result.to_dataframe()
    
    
    def get_geg_entities_data(
        self
        , entity: str
        , min_date: str = "2023-01-01"
        , max_date: str = "2023-05-17"
    ) -> RowIterator:
        
        query = f"""
                WITH geg_data AS ((
                SELECT
                    groupId,
                    entity_type,
                    a.entity as entity_name,
                    a.numMentions,
                    a.avgSalience,
                    eventTime,
                    polarity,
                    magnitude,
                    score, 
                    mid,
                    wikipediaUrl
                FROM (
                    SELECT 
                        polarity,
                        magnitude,
                        score,
                        FARM_FINGERPRINT(url) groupId,
                        entity.type AS entity_type,
                        FORMAT_TIMESTAMP("%Y-%m-%d", date, "UTC") eventTime,
                        entity.mid AS mid,
                        entity.wikipediaUrl AS wikipediaUrl
                    FROM `gdelt-bq.gdeltv2.geg_gcnlapi`, 
                    UNNEST(entities) entity 
                    WHERE entity.mid is not null
                    AND LOWER(name) LIKE '%{entity}%'
                    AND lang='en' 
                    AND DATE(date) >= "{min_date}" 
                    AND DATE(date) <= "{max_date}"
              ) b JOIN (
                  # grab the entities from the nested json in the graph
                 SELECT APPROX_TOP_COUNT(entities.name, 1)[OFFSET(0)].value entity,
                    entities.mid mid, 
                    sum(entities.numMentions) as numMentions,
                    avg(entities.avgSalience) as avgSalience
                  FROM `gdelt-bq.gdeltv2.geg_gcnlapi`,
                  UNNEST(entities) entities where entities.mid is not null 
                  AND lang='en'
                  AND DATE(date) >= "{min_date}" 
                  AND DATE(date) <= "{max_date}" 
                  GROUP BY entities.mid
              ) a USING(mid)))
            SELECT * 
            FROM ( SELECT *, RANK() OVER (PARTITION BY eventTime ORDER BY numMentions desc) as rank # get ranks
            FROM (
                SELECT
                    entity_name,
                    max(entity_type) AS entity_type,
                    DATE(eventTime) AS eventTime,
                    sum(numMentions) as numMentions,
                    avg(magnitude) as avgMagnitude,
                    max(mid) AS mid,
                    max(wikipediaUrl) AS wikipediaUrl,
                    FROM geg_data 
                    GROUP BY 1,3
                    ) grouped_all
                    ) 
                    WHERE rank < 300 
            """
    
        return self._query_bq(query_string=query).result()
    
    def get_geg_entities_data_full_df(
        self
        , entity: str
        , min_date: str = "2023-01-01"
        , max_date: str = "2023-05-17"
    ):
        result = self.get_geg_entities_data(entity, min_date, max_date)

        return result.to_dataframe()
    
    def get_geg_article_data(
        self
        , entity: str
        , min_date: str = "2023-01-01"
        , max_date: str = "2023-05-17"
    ) -> RowIterator:
        
        # here
        
        query = f"""
                WITH geg_data AS ((
                    SELECT
                        groupId,
                        url,
                        name,
                        -- a.entity AS entity_name,
                        wikipediaUrl,
                        a.numMentions AS numMentions,
                        a.avgSalience AS avgSalience,
                        DATE(eventTime) AS eventTime,
                        polarity,
                        magnitude,
                        score
                    FROM (
                        SELECT 
                            name,
                            polarity,
                            magnitude,
                            score,
                            url,
                            FARM_FINGERPRINT(url) AS groupId,
                            CONCAT(entity.type," - ",entity.type) AS entity_id,
                            FORMAT_TIMESTAMP("%Y-%m-%d", date, "UTC") AS eventTime,
                            entity.mid AS mid, 
                            entity.wikipediaUrl AS wikipediaUrl ,
                            entity.numMentions AS numMentions
                        FROM `gdelt-bq.gdeltv2.geg_gcnlapi`, 
                        UNNEST(entities) entity 
                        WHERE entity.mid is not null
                        AND LOWER(name) LIKE '%{entity}%'
                        AND lang='en' 
                        AND DATE(date) >= "{min_date}" 
                        AND DATE(date) <= "{max_date}"
                  ) b JOIN (
                      # grab the entities from the nested json in the graph
                     SELECT APPROX_TOP_COUNT(entities.name, 1)[OFFSET(0)].value entity,
                        entities.mid mid, 
                        sum(entities.numMentions) as numMentions,
                        avg(entities.avgSalience) as avgSalience
                      FROM `gdelt-bq.gdeltv2.geg_gcnlapi`,
                      UNNEST(entities) entities 
                      WHERE 
                      entities.mid is not null AND
                      lang='en'
                      AND DATE(date) >= "{min_date}" 
                      AND DATE(date) <= "{max_date}" 
                      GROUP BY entities.mid
                  ) a USING(mid)))
                SELECT * 
                FROM ( SELECT *, RANK() OVER (PARTITION BY eventTime ORDER BY numMentions desc) as rank # get ranks
                FROM (
                    SELECT 
                        -- ARRAY_AGG(entity_name) as entity_names,
                        STRING_AGG(name) as entity_names,
                        max(eventTime) AS eventTime,
                        url,
                        avg(numMentions) AS numMentions,
                        avg(avgSalience) AS avgSalience,
                        --sum(numMentions) as numMentions,
                        --avg(magnitude) as avgMagnitude
                    FROM geg_data 
                    GROUP BY url
                    ) 
                    -- grouped_all
                    )
                    WHERE rank < 300 
            """
    
        return self._query_bq(query_string=query).result()
    
    def get_geg_article_data_full_df(
        self
        , entity: str
        , min_date: str = "2023-01-01"
        , max_date: str = "2023-05-26"
    ):
        result = self.get_geg_article_data(entity, min_date, max_date)

        return result.to_dataframe()
    
    
    def get_geg_article_data_v2(
        self
        , entity: str
        , min_date: str = "2023-01-01"
        , max_date: str = "2023-05-26"
    ) -> RowIterator:
        
        # TODO - add arg for avgSalience
        
        query = f"""
            WITH
              entities AS (
              SELECT
                distinct url,
                b.avgSalience AS avgSalience,
                date AS date
              FROM
                `gdelt-bq.gdeltv2.geg_gcnlapi` AS a,
                UNNEST(a.entities) AS b
              WHERE
                LOWER(b.name) LIKE '%{entity}%'
                AND DATE(date) >= "{min_date}" 
                AND DATE(date) <= "{max_date}" 
                AND b.avgSalience > 0.1 )
            SELECT
              entities.url AS url,
              -- entities.url AS source,
              entities.date, 
              -- a.polarity,
              -- a.magnitude,
              -- a.score,
              avgSalience
              FROM entities inner join `gdelt-bq.gdeltv2.geg_gcnlapi` AS a 
              ON a.url=entities.url 
              AND a.date=entities.date
        """
        return self._query_bq(query_string=query).result()
    
    
    
    def get_geg_article_data_v2_full_df(
        self
        , entity: str
        , min_date: str = "2023-01-01"
        , max_date: str = "2023-05-26"
    ):
        result = self.get_geg_article_data_v2(entity, min_date, max_date)

        return result.to_dataframe()
