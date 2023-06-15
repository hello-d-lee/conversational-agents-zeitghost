from google.cloud import bigquery as bq

gdelt_input_record = [
    bq.SchemaField(name="SQLDATE", field_type="TIMESTAMP", mode="REQUIRED")
    , bq.SchemaField(name="Actor1Name", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="Actor2Name", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="GoldsteinScale", field_type="FLOAT64", mode="REQUIRED")
    , bq.SchemaField(name="NumMentions", field_type="INT64", mode="REQUIRED")
    , bq.SchemaField(name="NumSources", field_type="INT64", mode="REQUIRED")
    , bq.SchemaField(name="NumArticles", field_type="INT64", mode="REQUIRED")
    , bq.SchemaField(name="AvgTone", field_type="FLOAT64", mode="REQUIRED")
    , bq.SchemaField(name="SOURCEURL", field_type="STRING", mode="REQUIRED")
]

gdelt_processed_article = [
    bq.SchemaField(name="title", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="text", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="authors", field_type="STRING", mode="REPEATED")
    , bq.SchemaField(name="summary", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="publish_date", field_type="TIMESTAMP", mode="NULLABLE")
    , bq.SchemaField(name="url", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="language", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="date", field_type="DATETIME", mode="REQUIRED")
    , bq.SchemaField(name="Actor1Name", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="Actor2Name", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="GoldsteinScale", field_type="FLOAT64", mode="REQUIRED")
    , bq.SchemaField(name="NumMentions", field_type="INT64", mode="REPEATED")
    , bq.SchemaField(name="NumSources", field_type="INT64", mode="REQUIRED")
    , bq.SchemaField(name="NumArticles", field_type="INT64", mode="REQUIRED")
    , bq.SchemaField(name="AvgTone", field_type="FLOAT64", mode="REQUIRED")
    #, bq.SchemaField(name="entities_name", field_type="STRING", mode="REPEATED")
    #, bq.SchemaField(name="entities_type", field_type="STRING", mode="REPEATED")
    #, bq.SchemaField(name="entities_avgSalience", field_type="FLOAT64", mode="REPEATED")
]

gdelt_processed_record = [
    bq.SchemaField(name="domain", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="url", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="brand", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="description", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="categories", field_type="STRING", mode="REPEATED")
    , bq.SchemaField(name="article_count", field_type="INT64", mode="REQUIRED")
    , bq.SchemaField(name="articles", field_type="RECORD", mode="REPEATED", fields=gdelt_processed_article)
]

gdelt_reduced_articles = [
    bq.SchemaField(name="title", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="text", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="article_url", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="summary", field_type="STRING", mode="REQUIRED")
    , bq.SchemaField(name="date", field_type="TIMESTAMP", mode="NULLABLE")
    , bq.SchemaField(name="entities_name", field_type="STRING", mode="REPEATED")
    , bq.SchemaField(name="entities_type", field_type="STRING", mode="REPEATED")
    , bq.SchemaField(name="entities_numMentions", field_type="INT64", mode="REPEATED")
    , bq.SchemaField(name="entities_avg_Salience", field_type="FLOAT64", mode="REPEATED")
]

# gdelt_geg_articles_to_scrape = [
#     bq.SchemaField(name="url", field_type="STRING", mode="REQUIRED")
#     , bq.SchemaField(name="date", field_type="TIMESTAMP", mode="REQUIRED")
#     , bq.SchemaField(name="avgSalience", field_type="FLOAT", mode="REQUIRED")
# ]

