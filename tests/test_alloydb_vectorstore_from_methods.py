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

import numpy as np
import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from PIL import Image

from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, Column

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_SYNC = "test_table_sync" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
IMAGE_TABLE = "test_image_table" + str(uuid.uuid4()).replace("-", "_")
IMAGE_TABLE_SYNC = "test_image_table_sync" + str(uuid.uuid4()).replace("-", "_")
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
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


class FakeImageEmbedding(DeterministicFakeEmbedding):

    def embed_image(self, image_paths):
        return [self.embed_query(path) for path in image_paths]


image_embedding_service = FakeImageEmbedding(size=IMAGE_VECTOR_SIZE)


@pytest.mark.asyncio
class TestVectorStoreFromMethods:
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

    @pytest_asyncio.fixture
    async def engine(self, db_project, db_region, db_instance, db_cluster, db_name):
        engine = await AlloyDBEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
        )
        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=False,
        )
        await engine.ainit_vectorstore_table(IMAGE_TABLE, IMAGE_VECTOR_SIZE)
        yield engine
        await engine._aexecute(f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await engine._aexecute(f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")
        await engine._aexecute(f"DROP TABLE IF EXISTS {IMAGE_TABLE}")
        await engine._engine.dispose()

    @pytest_asyncio.fixture
    def engine_sync(self, db_project, db_region, db_instance, db_cluster, db_name):
        engine = AlloyDBEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            cluster=db_cluster,
            database=db_name,
        )

        engine.init_vectorstore_table(DEFAULT_TABLE_SYNC, VECTOR_SIZE)
        engine.init_vectorstore_table(IMAGE_TABLE_SYNC, IMAGE_VECTOR_SIZE)

        yield engine
        engine._execute(f"DROP TABLE IF EXISTS {DEFAULT_TABLE_SYNC}")
        engine._execute(f"DROP TABLE IF EXISTS {IMAGE_TABLE_SYNC}")
        engine._engine.dispose()

    @pytest_asyncio.fixture(scope="class")
    async def image_uris(self):
        image = Image.new("RGB", (100, 100), color="red")
        image.save("test_image_red_from.jpg")
        image = Image.new("RGB", (100, 100), color="green")
        image.save("test_image_green_from.jpg")
        image = Image.new("RGB", (100, 100), color="blue")
        image.save("test_image_blue_from.jpg")
        image_uris = [
            "test_image_red_from.jpg",
            "test_image_green_from.jpg",
            "test_image_blue_from.jpg",
        ]
        yield image_uris
        for uri in image_uris:
            os.remove(uri)

    async def test_afrom_texts(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await AlloyDBVectorStore.afrom_texts(
            texts,
            embeddings_service,
            engine,
            DEFAULT_TABLE,
            metadatas=metadatas,
            ids=ids,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_from_texts(self, engine_sync):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        AlloyDBVectorStore.from_texts(
            texts,
            embeddings_service,
            engine_sync,
            DEFAULT_TABLE_SYNC,
            metadatas=metadatas,
            ids=ids,
        )
        results = engine_sync._fetch(f"SELECT * FROM {DEFAULT_TABLE_SYNC}")

        assert len(results) == 3
        engine_sync._execute(f"TRUNCATE TABLE {DEFAULT_TABLE_SYNC}")

    async def test_afrom_docs(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await AlloyDBVectorStore.afrom_documents(
            docs,
            embeddings_service,
            engine,
            DEFAULT_TABLE,
            ids=ids,
        )
        results = await engine._afetch(f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_from_docs(self, engine_sync):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        AlloyDBVectorStore.from_documents(
            docs,
            embeddings_service,
            engine_sync,
            DEFAULT_TABLE_SYNC,
            ids=ids,
        )
        results = engine_sync._fetch(f"SELECT * FROM {DEFAULT_TABLE_SYNC}")

        assert len(results) == 3
        engine_sync._aexecute(f"TRUNCATE TABLE {DEFAULT_TABLE_SYNC}")

    async def test_afrom_texts_custom(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await AlloyDBVectorStore.afrom_texts(
            texts,
            embeddings_service,
            engine,
            CUSTOM_TABLE,
            ids=ids,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )
        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

        ids = [str(uuid.uuid4()) for i in range(len(texts))]

    async def test_afrom_docs_custom(self, engine):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        await AlloyDBVectorStore.afrom_documents(
            docs,
            embeddings_service,
            engine,
            CUSTOM_TABLE,
            ids=ids,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )

        results = await engine._afetch(f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "google.com"
        await engine._aexecute(f"TRUNCATE TABLE {CUSTOM_TABLE}")

    async def test_afrom_images(self, engine, image_uris):
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]
        await AlloyDBVectorStore.afrom_images(
            image_uris,
            image_embedding_service,
            engine,
            IMAGE_TABLE,
            metadatas=metadatas,
            ids=ids,
        )
        results = await engine._afetch(f"SELECT * FROM {IMAGE_TABLE}")

        assert len(results) == 3
        await engine._aexecute(f"TRUNCATE TABLE {IMAGE_TABLE}")

    async def test_from_images(self, engine_sync, image_uris):
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]
        AlloyDBVectorStore.from_images(
            image_uris,
            image_embedding_service,
            engine_sync,
            IMAGE_TABLE_SYNC,
            metadatas=metadatas,
            ids=ids,
        )
        results = engine_sync._fetch(f"SELECT * FROM {IMAGE_TABLE_SYNC}")

        assert len(results) == 3
        engine_sync._execute(f"TRUNCATE TABLE {IMAGE_TABLE_SYNC}")
