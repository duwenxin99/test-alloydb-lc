# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import uuid

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from PIL import Image

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, Column

DEFAULT_TABLE = "test_table" + str(uuid.uuid4())
DEFAULT_TABLE_SYNC = "test_table_sync" + str(uuid.uuid4())
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4())
IMAGE_TABLE = "test_image_table" + str(uuid.uuid4())
VECTOR_SIZE = 768
IMAGE_VECTOR_SIZE = 1024

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query("foo") for i in range(len(texts))]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f'Must set env var "{key} to: "{desc}"')
    return v


@pytest.mark.asyncio(scope="class")
class TestVectorStore:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for AlloyDB instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for alloydb")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name for AlloyDB")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_instance, db_cluster, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
        )

        yield engine

    @pytest_asyncio.fixture(scope="class")
    def engine_sync(self, db_project, db_region, db_instance, db_cluster, db_name):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
        )
        yield engine

    @pytest_asyncio.fixture(scope="class")
    def vs_sync(self, engine_sync):
        engine_sync.init_vectorstore_table(DEFAULT_TABLE_SYNC, VECTOR_SIZE)
        vs = AlloyDBVectorStore.create_sync(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE_SYNC,
        )
        yield vs
        engine_sync._execute(f'DROP TABLE IF EXISTS "{DEFAULT_TABLE_SYNC}"')

        engine_sync._engine.dispose()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        vs = await AlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )
        yield vs
        await engine._aexecute(f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await engine._engine.dispose()

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine):
        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            metadata_json_column="mymeta",
        )
        vs = await AlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
            metadata_json_column="mymeta",
        )
        yield vs
        await engine._aexecute(f'DROP TABLE IF EXISTS "{CUSTOM_TABLE}"')

    @pytest_asyncio.fixture(scope="class")
    async def image_vs(self, engine):
        await engine.ainit_vectorstore_table(
            IMAGE_TABLE, IMAGE_VECTOR_SIZE, store_metadata=False
        )
        vs = await AlloyDBVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=IMAGE_TABLE,
        )
        yield vs

        await engine._aexecute(f'DROP TABLE IF EXISTS "{IMAGE_TABLE}"')

    @pytest_asyncio.fixture(scope="class")
    async def image_vs_sync(self, engine_sync):
        engine_sync.init_vectorstore_table(
            IMAGE_TABLE, IMAGE_VECTOR_SIZE, store_metadata=False
        )
        vs = AlloyDBVectorStore.create(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=IMAGE_TABLE,
        )
        yield vs

        await engine_sync._aexecute(f'DROP TABLE IF EXISTS "{IMAGE_TABLE}"')

    @pytest_asyncio.fixture(scope="class")
    async def image_uris(self):
        image = Image.new("RGB", (100, 100), color="red")
        image.save("test_image_red.jpg")
        image = Image.new("RGB", (100, 100), color="green")
        image.save("test_image_green.jpg")
        image = Image.new("RGB", (100, 100), color="blue")
        image.save("test_image_blue.jpg")
        image_uris = [
            "test_image_red.jpg",
            "test_image_green.jpg",
            "test_image_blue.jpg",
        ]
        yield image_uris
        for uri in image_uris:
            os.remove(uri)

    async def test_init_with_constructor(self, engine):
        with pytest.raises(Exception):
            AlloyDBVectorStore(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    async def test_post_init(self, engine):
        with pytest.raises(ValueError):
            await AlloyDBVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    async def test_aadd_texts(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas, ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 6
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_texts_edge_cases(self, engine, vs):
        texts = ["Taylor's", '"Swift"', "best-friend"]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_docs(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_documents(docs, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_images(self, engine, image_vs, image_uris):
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]

        await image_vs.aadd_images(image_uris, ids=ids)
        await engine._aexecute(f'TRUNCATE TABLE "{IMAGE_TABLE}"')

    async def test_aadd_embedding(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs._aadd_embeddings(texts, embeddings, metadatas, ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_embedding_without_id(self, engine, vs):
        await vs._aadd_embeddings(texts, embeddings, metadatas)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        assert results[0]["langchain_id"]
        await engine._aexecute(f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_adelete(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        # delete an ID
        await vs.adelete([ids[0]])
        results = await engine._afetch(f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2

    async def test_aadd_texts_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, metadatas, ids)
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 6
        await engine._aexecute(f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_adelete_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, ids=ids)
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        # delete an ID
        await vs_custom.adelete([ids[0]])
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        content = [result["mycontent"] for result in results]
        assert len(results) == 2
        assert "foo" not in content
        await vs_custom.adelete(ids)

    async def test_aadd_docs_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        await vs_custom.aadd_documents(docs, ids=ids)

        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "google.com"
        await engine._aexecute(f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_aadd_embedding_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom._aadd_embeddings(texts, embeddings, metadatas, ids)
        results = await engine._afetch(f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        await engine._aexecute(f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    def test_add_docs(self, engine_sync, vs_sync):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs_sync.add_documents(docs, ids=ids)
        results = engine_sync._fetch(f'SELECT * FROM "{DEFAULT_TABLE_SYNC}"')
        assert len(results) == 3

    def test_add_texts(self, engine_sync, vs_sync):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs_sync.add_texts(texts, ids=ids)
        results = engine_sync._fetch(f'SELECT * FROM "{DEFAULT_TABLE_SYNC}"')
        assert len(results) == 6

    def test_add_images(self, engine_sync, image_vs_sync, image_uris):
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]

        image_vs_sync.add_images(image_uris, ids=ids)
        results = engine_sync._fetch(f'SELECT * FROM "{DEFAULT_TABLE_SYNC}"')
        assert len(results) == 3

    async def test_ignore_metadata_columns(self, vs_custom):
        column_to_ignore = "source"
        vs = await AlloyDBVectorStore.create(
            vs_custom.engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            ignore_metadata_columns=[column_to_ignore],
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_json_column="mymeta",
        )
        assert column_to_ignore not in vs.metadata_columns

    async def test_create_vectorstore_with_invalid_parameters(self, vs_custom):
        with pytest.raises(ValueError):
            await AlloyDBVectorStore.create(
                vs_custom.engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=["random_column"],  # invalid metadata column
            )
        with pytest.raises(ValueError):
            await AlloyDBVectorStore.create(
                vs_custom.engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="langchain_id",  # invalid content column type
                embedding_column="myembedding",
                metadata_columns=["random_column"],
            )
        with pytest.raises(ValueError):
            await AlloyDBVectorStore.create(
                vs_custom.engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="random_column",  # invalid embedding column
                metadata_columns=["random_column"],
            )
        with pytest.raises(ValueError):
            await AlloyDBVectorStore.create(
                vs_custom.engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="langchain_id",  # invalid embedding column data type
                metadata_columns=["random_column"],
            )
