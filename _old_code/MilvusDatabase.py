from milvus import default_server
from pymilvus import FieldSchema, Collection, connections, DataType, CollectionSchema
import numpy as np


class MilvusDatabase:
    '''
    Instance of the milvus database for storing embeddings

    https://zilliz.com/blog/getting-started-with-a-milvus-connection

    '''

    def __init__(self):
        # Start database
        default_server.start()

        connections.connect(
            host="127.0.0.1",
            port=default_server.listen_port
        )

        self.mc = self.setup_schema()

    def __setup_schema(self) -> Collection:
        ## 1. Define a minimum expandable schema.
        fields = [
            FieldSchema("pk", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=768),
        ]
        schema = CollectionSchema(
            fields,
            enable_dynamic_field=True,
        )

        ## 2. Create a collection.
        mc = Collection("imported_text_collection", schema)

        ## 3. Index the collection.
        mc.create_index(
            field_name="vector",
            index_params={
                "Index_type": "AUTOINDEX",
                "Metric_type": "COSINE"
            }
        )

        return mc

    def insert_data(self):
        ## 1. Input database can be pandas dataframe or list of dicts.
        data_rows = []
        data_rows.extend([
            {
                "vector": np.random.randn(768).tolist(),
                "text": "This is a document",
                "source": "source_url_1"
            },
            {
                "vector": np.random.randn(768).tolist(),
                "text": "This is another document",
                "source": "source_url_2"
            },
        ])

        ## 2. Insert database into milvus.
        self.mc.insert(data_rows)
        self.mc.flush()

    def queryData(self):
        ## 1. Search for answers to your embedded questions.
        self.mc.load()
        results = self.mc.search(
            data=encoder(["my_question_1"]),
            anns_field="vector",
            output_fields=["text", "source"],  # optional return fields
            limit=3,
            param={},  # no params if using milvus defaults
        )

        ## 2. View the answers.
        for n, hits in enumerate(results):
            print(f"{n}th result:")
            for hit in hits:
                print(hit)
