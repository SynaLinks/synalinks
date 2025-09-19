# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import os
import warnings
from typing import Any
from typing import Dict

import neo4j

from synalinks.src.backend import is_entity
from synalinks.src.backend import is_relation
from synalinks.src.backend import is_similarity_search
from synalinks.src.backend import is_triplet_search
from synalinks.src.backend.common.json_utils import out_mask_json
from synalinks.src.knowledge_bases.database_adapters import DatabaseAdapter
from synalinks.src.utils.async_utils import run_maybe_nested
from synalinks.src.utils.naming import to_snake_case


class Neo4JAdapter(DatabaseAdapter):
    def __init__(
        self,
        uri=None,
        entity_models=None,
        relation_models=None,
        embedding_model=None,
        metric="cosine",
        wipe_on_start=False,
    ):
        self.db_name = os.getenv("NEO4J_DATABASE", "neo4j")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "neo4j")

        super().__init__(
            uri=uri,
            entity_models=entity_models,
            relation_models=relation_models,
            embedding_model=embedding_model,
            metric=metric,
            wipe_on_start=wipe_on_start,
        )

    def wipe_database(self):
        """Wipe all data from the database"""
        run_maybe_nested(
            self.query(
                """
                MATCH (n)
                CALL (n) {
                    DETACH DELETE n
                } IN TRANSACTIONS OF 10000 ROWS
                """,
                read_only=False,
            )
        )
        result = run_maybe_nested(self.query("SHOW VECTOR INDEXES"))
        for indexes in result:
            index_name = indexes["name"]
            query = "DROP INDEX $index"
            params = {
                "index": index_name,
            }
            run_maybe_nested(self.query(query, params))

    def create_vector_index(self):
        """Create vector indexes"""
        for entity_model in self.entity_models:
            node_label = self.sanitize_label(entity_model.get_schema().get("title"))
            index_name = to_snake_case(node_label)
            query = "\n".join(
                [
                    "CREATE VECTOR INDEX $indexName IF NOT EXISTS",
                    f"FOR (n:{node_label}) ON n.embedding",
                    "OPTIONS {indexConfig : {"
                    " `vector.dimensions`: $dimension,"
                    " `vector.similarity_function`: $similarityFunction"
                    "}};",
                ]
            )
            params = {
                "indexName": index_name,
                "dimension": self.embedding_dim,
                "similarityFunction": self.metric,
            }
            run_maybe_nested(
                self.query(query, params=params),
            )
        if self.entity_models:
            run_maybe_nested(
                self.query("CALL db.awaitIndexes(300)"),
            )

    async def query(
        self, query: str, params: Dict[str, Any] = None, read_only=True, **kwargs
    ):
        driver = neo4j.GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        result_list = []
        if read_only:
            if not params:
                params = {}
            params["routing_"] = neo4j.RoutingControl.READ
        try:
            with driver.session(database=self.db_name) as session:
                if params:
                    result = session.run(query, **params, **kwargs)
                else:
                    result = session.run(query, **kwargs)
                for record in reversed(list(result)):
                    data = record.data()
                    if isinstance(data, dict):
                        data = out_mask_json(data, mask=["embedding"])
                    result_list.append(data)
                session.close()
        finally:
            driver.close()
        return result_list

    async def update(
        self,
        data_model,
        threshold=0.8,
    ):
        if is_relation(data_model):
            subj = data_model.get_nested_entity("subj")
            obj = data_model.get_nested_entity("obj")
            relation_label = data_model.get("label")
            subj_label = self.sanitize_label(subj.get("label"))
            subj_vector = subj.get("embedding")
            obj_label = self.sanitize_label(obj.get("label"))
            obj_vector = obj.get("embedding")

            if not subj_vector or not obj_vector:
                warnings.warn(
                    "No embedding found for `subj` or `obj`:"
                    " Entities and relations needs to be embedded. "
                    "Use `Embedding` module before `UpdateKnowledge`. "
                    "Skipping update."
                )
                return

            relation_properties = self.sanitize_properties(data_model.get_json())
            set_clauses = []
            for key in relation_properties.keys():
                if key not in ("subj", "obj"):
                    set_clauses.append(f"r.{key} = ${key}")
            set_statement = "SET " + ", ".join(set_clauses) if set_clauses else ""

            query = "\n".join(
                [
                    "CALL db.index.vector.queryNodes($subjIndexName, 1, $subjVector)",
                    "YIELD node AS s, score AS subj_score",
                    "WHERE subj_score >= $threshold",
                    "CALL db.index.vector.queryNodes($objIndexName, 1, $objVector)",
                    "YIELD node AS o, score AS obj_score",
                    "WHERE obj_score >= $threshold",
                    f"MERGE (s)-[r:{relation_label}]->(o)",
                    (
                        set_statement
                        if set_statement
                        else "// No additional properties to set"
                    ),
                ]
            )
            params = {
                "subjIndexName": to_snake_case(subj_label),
                "objIndexName": to_snake_case(obj_label),
                "threshold": threshold,
                "subjVector": subj_vector,
                "objVector": obj_vector,
                **relation_properties,
            }
            await self.query(query, params=params, read_only=False)
        elif is_entity(data_model):
            node_label = self.sanitize_label(data_model.get("label"))
            vector = data_model.get("embedding")

            if not vector:
                warnings.warn(
                    "Entities need to be embedded. "
                    "Make sure to use `Embedding` module before `UpdateKnowledge`. "
                    "Skipping update."
                )
                return

            node_properties = self.sanitize_properties(data_model.get_json())
            set_clauses = []
            for key in node_properties.keys():
                set_clauses.append(f"n.{key} = ${key}")
            set_statement = "SET " + ", ".join(set_clauses) if set_clauses else ""

            query = "\n".join(
                [
                    "CALL db.index.vector.queryNodes($indexName, 1, $vector)",
                    "YIELD node, score",
                    "WHERE score >= $threshold",
                    "WITH count(node) as existing_count",
                    "WHERE existing_count = 0",
                    f"CREATE (n:{node_label})",
                    (
                        set_statement
                        if set_statement
                        else "// No additional properties to set"
                    ),
                ]
            )
            params = {
                "indexName": to_snake_case(node_label),
                "threshold": threshold,
                "vector": vector,
                **node_properties,
            }
            await self.query(query, params=params, read_only=False)
        else:
            raise ValueError(
                "The parameter `data_model` must be an `Entity` or `Relation` instance"
            )

    async def similarity_search(
        self,
        similarity_search,
        k=10,
        threshold=0.7,
    ):
        if not is_similarity_search(similarity_search):
            raise ValueError(
                "The `similarity_search` argument "
                "should be a `SimilaritySearch` data model"
            )
        text = similarity_search.get("similarity_search")
        entity_label = similarity_search.get("entity_label")
        vector = (await self.embedding_model(texts=[text]))["embeddings"][0]
        index_name = to_snake_case(self.sanitize_label(entity_label))
        query = "\n".join(
            [
                "CALL db.index.vector.queryNodes(",
                " $indexName,",
                " $numberOfNearestNeighbours,",
                " $vector) YIELD node AS node, score",
                "WITH node, score",
                "WHERE score >= $threshold",
                "RETURN node AS node, score",
                "LIMIT $numberOfNearestNeighbours",
            ]
        )
        params = {
            "indexName": index_name,
            "numberOfNearestNeighbours": k,
            "threshold": threshold,
            "vector": vector,
        }
        result = await self.query(query, params=params)
        return result

    async def triplet_search(
        self,
        triplet_search,
        k=10,
        threshold=0.7,
    ):
        if not is_triplet_search(triplet_search):
            raise ValueError(
                "The `triplet_search` argument should be a `TripletSearch` data model"
            )
        subject_label = triplet_search.get("subject_label")
        subject_label = self.sanitize_label(subject_label)
        subject_similarity_search = triplet_search.get("subject_similarity_search")
        relation_label = triplet_search.get("relation_label")
        relation_label = self.sanitize_label(relation_label)
        object_label = triplet_search.get("object_label")
        object_label = self.sanitize_label(object_label)
        object_similarity_search = triplet_search.get("object_similarity_search")

        params = {
            "numberOfNearestNeighbours": k,
            "threshold": threshold,
        }
        query_lines = []

        has_subject_similarity = (
            subject_similarity_search and subject_similarity_search != "?"
        )
        has_object_similarity = (
            object_similarity_search and object_similarity_search != "?"
        )

        if has_subject_similarity and has_object_similarity:
            subject_vector = (
                await self.embedding_model(texts=[subject_similarity_search])
            )["embeddings"][0]
            object_vector = (
                await self.embedding_model(texts=[object_similarity_search])
            )["embeddings"][0]
            params["subjVector"] = subject_vector
            params["objVector"] = object_vector

            params["subjIndexName"] = to_snake_case(subject_label)
            query_lines.append(
                (
                    "CALL db.index.vector.queryNodes("
                    "$subjIndexName, $numberOfNearestNeighbours, $subjVector)"
                )
            )
            query_lines.extend(
                [
                    "YIELD node AS subj, score AS subj_score",
                    "WHERE subj_score >= $threshold",
                ]
            )
            params["objIndexName"] = to_snake_case(object_label)
            query_lines.append(
                (
                    "CALL db.index.vector.queryNodes("
                    "$objIndexName, $numberOfNearestNeighbours, $objVector)"
                )
            )
            query_lines.extend(
                [
                    "YIELD node AS obj, score AS obj_score",
                    "WHERE obj_score >= $threshold",
                    "WITH subj, subj_score, obj, obj_score",
                ]
            )
            query_lines.append(f"MATCH (subj)-[relation:{relation_label}]->(obj)")
            query_lines.append("WITH subj, subj_score, relation, obj, obj_score")
        elif has_subject_similarity:
            subject_vector = (
                await self.embedding_model(texts=[subject_similarity_search])
            )["embeddings"][0]
            params["subjVector"] = subject_vector

            params["subjIndexName"] = to_snake_case(subject_label)
            query_lines.append(
                (
                    "CALL db.index.vector.queryNodes("
                    "$subjIndexName, $numberOfNearestNeighbours, $subjVector)"
                )
            )
            query_lines.extend(
                [
                    "YIELD node AS subj, score AS subj_score",
                    "WHERE subj_score >= $threshold",
                ]
            )
            query_lines.append(f"MATCH (subj)-[relation:{relation_label}]->(obj)")
            query_lines.append("WITH subj, subj_score, relation, obj, 1.0 AS obj_score")

        elif has_object_similarity:
            object_vector = (
                await self.embedding_model(texts=[object_similarity_search])
            )["embeddings"][0]
            params["objVector"] = object_vector
            params["objIndexName"] = to_snake_case(object_label)
            query_lines.append(
                (
                    "CALL db.index.vector.queryNodes("
                    "$objIndexName, $numberOfNearestNeighbours, $objVector)"
                )
            )
            query_lines.extend(
                [
                    "YIELD node AS obj, score AS obj_score",
                    "WHERE obj_score >= $threshold",
                ]
            )
            query_lines.append(f"MATCH (subj)-[relation:{relation_label}]->(obj)")
            query_lines.append("WITH subj, 1.0 AS subj_score, relation, obj, obj_score")
        else:
            query_lines.append(
                (
                    f"MATCH (subj:{subject_label})"
                    f"-[relation:{relation_label}]->"
                    f"(obj:{object_label})"
                )
            )
            query_lines.append(
                "WITH subj, 1.0 AS subj_score, relation, obj, 1.0 AS obj_score"
            )
        query_lines.append(
            (
                "RETURN subj, properties(relation) AS relation, obj, "
                "sqrt(subj_score * obj_score) AS score"
            )
        )
        query_lines.append("LIMIT $numberOfNearestNeighbours")
        query = "\n".join(query_lines)
        return await self.query(query, params)
