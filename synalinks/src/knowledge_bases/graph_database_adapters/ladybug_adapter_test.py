# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Tests for `LadybugAdapter`.

Each test stands up an in-memory ``ladybug://:memory:`` database so
state is fully isolated per test. The embedding model is replaced
with a deterministic stub (no LLM calls) so dedup behaviour is
verifiable.
"""

from typing import List
from typing import Literal
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.backend.pydantic.knowledge import Entity
from synalinks.src.backend.pydantic.knowledge import Relation
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    LadybugAdapter,
)
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    _assert_read_only_cypher,
)
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    _cypher_type_to_json_property,
)
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    sanitize_label,
)
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    sanitize_properties,
)


class Person(Entity):
    label: Literal["Person"]
    name: str
    embedding: List[float] = []


class City(Entity):
    label: Literal["City"]
    name: str
    embedding: List[float] = []


class LivesIn(Relation):
    label: Literal["LivesIn"]
    subj: Person
    obj: City


class _StubEmbeddingModel:
    """A minimal stand-in for ``EmbeddingModel`` that yields canned
    vectors. Used in place of the real model to make adapter tests
    deterministic (no LLM calls) and fast.

    The adapter calls ``await embedding_model(EmbeddingRequest(texts=...))``
    and reads ``probe["embeddings"][0]``, so this single ``__call__`` covers
    the full surface that ``LadybugAdapter`` uses.
    """

    def __init__(self, vectors_by_text=None):
        self._vectors = vectors_by_text or {}

    async def __call__(self, request):
        # The adapter wraps texts in an ``EmbeddingRequest``; accept that
        # (and a bare list / dict, for robustness) and read the texts off it.
        texts = getattr(request, "texts", None)
        if texts is None:
            texts = request.get("texts") if hasattr(request, "get") else request
        return {"embeddings": [self._vectors.get(t, [0.0, 0.0, 0.0]) for t in texts]}


# ----------------------------------------------------------------------
# Sanitization helpers
# ----------------------------------------------------------------------


class SanitizationTest(testing.TestCase):
    def test_sanitize_label_keeps_pascal_case(self):
        self.assertEqual(sanitize_label("Doc"), "Doc")
        self.assertEqual(sanitize_label("MyDoc"), "MyDoc")

    def test_sanitize_label_normalizes_to_pascal_case(self):
        """Mirrors DuckDB's table_identifier: any reasonable input
        form ends up PascalCase, so KBs that mix SQL + graph stay
        consistent in their identifier convention."""
        self.assertEqual(sanitize_label("my_doc"), "MyDoc")
        self.assertEqual(sanitize_label("my-doc"), "MyDoc")
        self.assertEqual(sanitize_label("my doc"), "MyDoc")
        self.assertEqual(sanitize_label("myDoc"), "MyDoc")

    def test_sanitize_label_strips_injection_characters(self):
        """Special chars are dropped by to_pascal_case, so a label
        carrying SQL/Cypher injection text comes out safe."""
        self.assertEqual(sanitize_label("Doc; DROP TABLE x"), "DocDropTableX")

    def test_sanitize_label_rejects_unrepresentable(self):
        for bad in ("", "1Doc", "123", "!!!"):
            with self.assertRaises(ValueError):
                sanitize_label(bad)

    def test_sanitize_properties_drops_label_and_bad_keys(self):
        out = sanitize_properties(
            {"label": "X", "name": "ok", "1bad": "x", "good_key": "y"}
        )
        self.assertEqual(out, {"name": "ok", "good_key": "y"})

    def test_sanitize_properties_drops_subj_obj(self):
        out = sanitize_properties({"subj": {}, "obj": {}, "weight": 1.0})
        self.assertEqual(out, {"weight": 1.0})

    def test_sanitize_properties_normalizes_to_snake_case(self):
        """Property names go through to_snake_case to match
        DuckDB's column_identifier convention, so a KB that mixes
        SQL + graph stays consistent."""
        out = sanitize_properties({"myField": 1, "OtherField": 2, "snake_case_field": 3})
        self.assertEqual(
            out,
            {"my_field": 1, "other_field": 2, "snake_case_field": 3},
        )


# ----------------------------------------------------------------------
# read_only Cypher enforcement
# ----------------------------------------------------------------------


class ReadOnlyCypherTest(testing.TestCase):
    def test_pure_match_allowed(self):
        _assert_read_only_cypher("MATCH (n:Doc) RETURN n LIMIT 10")

    def test_create_rejected(self):
        with self.assertRaisesRegex(ValueError, "CREATE"):
            _assert_read_only_cypher("CREATE (:Doc {id: 'x'})")

    def test_merge_rejected(self):
        with self.assertRaisesRegex(ValueError, "MERGE"):
            _assert_read_only_cypher("MATCH (n) MERGE (m:X)")

    def test_delete_rejected(self):
        with self.assertRaisesRegex(ValueError, "DELETE"):
            _assert_read_only_cypher("MATCH (n) DELETE n")

    def test_detach_rejected(self):
        with self.assertRaisesRegex(ValueError, "DETACH"):
            _assert_read_only_cypher("MATCH (n) DETACH DELETE n")

    def test_set_rejected(self):
        with self.assertRaisesRegex(ValueError, "SET"):
            _assert_read_only_cypher("MATCH (n) SET n.x = 1")

    def test_remove_rejected(self):
        with self.assertRaisesRegex(ValueError, "REMOVE"):
            _assert_read_only_cypher("MATCH (n) REMOVE n.x")

    def test_copy_rejected(self):
        with self.assertRaisesRegex(ValueError, "COPY"):
            _assert_read_only_cypher("COPY Doc FROM '/etc/passwd'")

    def test_install_rejected(self):
        with self.assertRaisesRegex(ValueError, "INSTALL"):
            _assert_read_only_cypher("INSTALL httpfs")

    def test_drop_rejected(self):
        with self.assertRaisesRegex(ValueError, "DROP"):
            _assert_read_only_cypher("DROP TABLE Doc")

    def test_comment_stripped_so_no_false_positive(self):
        # `// CREATE` in a comment must NOT trigger the rejection.
        _assert_read_only_cypher("MATCH (n) RETURN n  // CREATE later")
        _assert_read_only_cypher("/* CREATE (:X) */ MATCH (n) RETURN n")


# ----------------------------------------------------------------------
# Adapter wired up against in-memory Ladybug
# ----------------------------------------------------------------------


class LadybugAdapterTest(testing.TestCase):
    def _adapter(self, *, embedding_model=None, **kwargs):
        # `_get_em` only knows how to coerce strings / dicts /
        # EmbeddingModel instances; patch it to pass our stub straight
        # through. Real EmbeddingModel objects would also work but
        # require mocking litellm — too much ceremony for unit tests
        # that don't exercise the model itself.
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn],
                embedding_model=embedding_model,
                vector_dim=3,
                **kwargs,
            )

    def test_schema_creates_node_and_rel_tables(self):
        adapter = self._adapter()
        self.assertEqual(adapter._existing_tables("NODE"), {"Person", "City"})
        self.assertEqual(adapter._existing_tables("REL"), {"LivesIn"})

    async def test_update_entities_inserts_when_no_duplicate(self):
        adapter = self._adapter(
            embedding_model=_StubEmbeddingModel({"alice": [1.0, 0.0, 0.0]})
        )
        ids = await adapter.update_entities(
            [Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])]
        )
        self.assertEqual(len(ids), 1)
        rows = adapter._con.execute("MATCH (p:Person) RETURN p.name AS name").get_all()
        self.assertEqual(rows, [["Alice"]])

    async def test_update_entities_dedup_returns_existing_id(self):
        """Inserting two near-identical embeddings should yield only one
        node — the second call returns the first's id."""
        adapter = self._adapter(
            embedding_model=_StubEmbeddingModel({}),
            dedup_threshold=0.85,
        )
        first = await adapter.update_entities(
            Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])
        )
        second = await adapter.update_entities(
            Person(
                label="Person",
                name="Alice (dup)",
                embedding=[0.99, 0.01, 0.0],
            )
        )
        self.assertEqual(first, second)
        rows = adapter._con.execute("MATCH (p:Person) RETURN p.name").get_all()
        self.assertEqual(len(rows), 1)

    async def test_update_entities_no_dedup_when_far_apart(self):
        adapter = self._adapter(
            embedding_model=_StubEmbeddingModel({}),
            dedup_threshold=0.85,
        )
        first = await adapter.update_entities(
            Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])
        )
        second = await adapter.update_entities(
            Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0])
        )
        self.assertNotEqual(first, second)
        rows = adapter._con.execute("MATCH (p:Person) RETURN p.name").get_all()
        self.assertEqual(len(rows), 2)

    async def test_update_relations_merges_edge_to_existing_endpoints(self):
        """Endpoints must be in the graph first (vector-resolved by the
        single-query MERGE); update_knowledge_graph does this
        automatically by running update_entities first."""
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        # Seed endpoints.
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        rel = LivesIn(
            label="LivesIn",
            subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
            obj=City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
        )
        await adapter.update_relations(rel)

        rows = adapter._con.execute(
            "MATCH (p:Person)-[:LivesIn]->(c:City) RETURN p.name, c.name"
        ).get_all()
        self.assertEqual(rows, [["Alice", "Paris"]])

    async def test_update_relations_is_idempotent(self):
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        rel = LivesIn(
            label="LivesIn",
            subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
            obj=City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
        )
        await adapter.update_relations(rel)
        await adapter.update_relations(rel)

        edge_count = adapter._con.execute(
            "MATCH ()-[r:LivesIn]->() RETURN count(r) AS c"
        ).get_all()
        self.assertEqual(edge_count, [[1]])

    async def test_update_relations_silently_drops_when_endpoint_missing(self):
        """The legacy MemGraph semantics: when an endpoint doesn't
        vector-match any existing node, the MERGE silently no-ops."""
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        # Note: endpoints NOT seeded.
        result = await adapter.update_relations(
            LivesIn(
                label="LivesIn",
                subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                obj=City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
            )
        )
        self.assertIsNone(result)
        edges = adapter._con.execute(
            "MATCH ()-[r:LivesIn]->() RETURN count(r) AS c"
        ).get_all()
        self.assertEqual(edges, [[0]])

    async def test_cypher_read_only_blocks_writes(self):
        adapter = self._adapter()
        with self.assertRaisesRegex(ValueError, "read_only=True"):
            await adapter.cypher("CREATE (:Person {name: 'X'})")

    async def test_cypher_read_only_false_allows_writes(self):
        adapter = self._adapter()
        await adapter.cypher(
            "CREATE (:Person {name: 'X'})",
            read_only=False,
        )
        rows = await adapter.cypher("MATCH (p:Person) RETURN p.name AS name")
        self.assertEqual(rows, [{"name": "X"}])

    async def test_cypher_csv_output(self):
        adapter = self._adapter()
        await adapter.cypher(
            "CREATE (:Person {name: 'X'})",
            read_only=False,
        )
        out = await adapter.cypher(
            "MATCH (p:Person) RETURN p.name AS name",
            output_format="csv",
        )
        self.assertIn("name", out)
        self.assertIn("X", out)

    async def test_entity_similarity_search(self):
        adapter = self._adapter(
            embedding_model=_StubEmbeddingModel({"alice": [1.0, 0.0, 0.0]}),
        )
        await adapter.update_entities(
            Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])
        )
        await adapter.update_entities(
            Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0])
        )
        results = await adapter.entity_similarity_search("alice", label="Person", k=2)
        # Alice should rank first (zero distance).
        self.assertEqual(len(results), 2)
        first = results[0]
        self.assertIn("distance", first)
        self.assertLessEqual(first["distance"], results[1]["distance"])

    async def test_entity_similarity_search_requires_embedding_model(self):
        adapter = self._adapter()  # no embedding_model
        with self.assertRaisesRegex(ValueError, "embedding_model"):
            await adapter.entity_similarity_search("anything", label="Person")

    async def test_update_knowledge_graph_combines_entities_and_relations(
        self,
    ):
        from synalinks.src.backend.pydantic.knowledge import KnowledgeGraph as KG

        class MiniKG(KG):
            entities: List[Person]
            relations: List[LivesIn]

        kg = MiniKG(
            entities=[
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
            ],
            relations=[
                LivesIn(
                    label="LivesIn",
                    subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                    obj=City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
                )
            ],
        )
        adapter = self._adapter(
            embedding_model=_StubEmbeddingModel({}),
        )
        result = await adapter.update_knowledge_graph(kg)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(len(result["relations"]), 1)

    def test_uri_routing_returns_ladybug_adapter(self):
        from synalinks.src.knowledge_bases import graph_database_adapters

        self.assertIs(
            graph_database_adapters.get("ladybug://:memory:"),
            LadybugAdapter,
        )

    def test_schema_type_mapping_covers_pydantic_shapes(self):
        """Verify the column converter handles the schema shapes
        Pydantic v2 actually emits: scalars, optional (anyOf with
        null), lists of primitives, lists of objects (JSON
        fallback), date / date-time formats, and enums."""
        from datetime import date as _date
        from datetime import datetime as _datetime
        from enum import Enum
        from typing import Optional

        class Color(str, Enum):
            RED = "red"
            BLUE = "blue"

        class Wide(Entity):
            label: Literal["Wide"]
            name: str
            count: int
            score: float
            active: bool
            tags: List[str] = []
            scores: List[float] = []
            counts: List[int] = []
            flags: List[bool] = []
            metadata: dict = {}
            nickname: Optional[str] = None
            born: _date = _date(2000, 1, 1)
            seen_at: _datetime = _datetime(2000, 1, 1, 0, 0, 0)
            color: Color = Color.RED

        adapter = LadybugAdapter(
            uri="ladybug://:memory:",
            entity_models=[Wide],
            vector_dim=3,
        )
        cols = adapter._con.execute("CALL TABLE_INFO('Wide') RETURN name, type").get_all()
        col_types = {name: dtype for name, dtype in cols}

        self.assertEqual(col_types["name"], "STRING")
        self.assertEqual(col_types["count"], "INT64")
        self.assertEqual(col_types["score"], "DOUBLE")
        self.assertEqual(col_types["active"], "BOOL")
        self.assertEqual(col_types["tags"], "STRING[]")
        self.assertEqual(col_types["scores"], "DOUBLE[]")
        self.assertEqual(col_types["counts"], "INT64[]")
        self.assertEqual(col_types["flags"], "BOOL[]")
        # dicts and unspecified-shape objects fall back to JSON.
        self.assertEqual(col_types["metadata"], "JSON")
        # Optional[str] → anyOf [string, null]; non-null variant wins.
        self.assertEqual(col_types["nickname"], "STRING")
        self.assertEqual(col_types["born"], "DATE")
        self.assertEqual(col_types["seen_at"], "TIMESTAMP")
        # Pydantic v2 Enums emit a $ref; resolves to a string field.
        self.assertEqual(col_types["color"], "STRING")

    async def test_update_entities_rebuilds_fts_at_write_time(self):
        """Search paths assume the FTS index is current — the rebuild
        happens at the end of update_entities, not at query time."""
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        # First search hits the FTS index right after a write — must
        # already be current without any explicit query-time rebuild
        # call. (Verified by patching _rebuild_fts_index to error if
        # called from a search path.)
        from unittest.mock import patch

        await adapter.update_entities(
            Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])
        )

        def _explode(*a, **kw):
            raise AssertionError(
                "_rebuild_fts_index must not be called from search paths"
            )

        with patch.object(adapter, "_rebuild_fts_index", _explode):
            results = await adapter.entity_fulltext_search("Alice", label="Person")
        self.assertEqual(len(results), 1)

    # ------------------------------------------------------------------
    # Lifecycle, raw queries, edge cases
    # ------------------------------------------------------------------

    async def test_wipe_database_drops_all_tables(self):
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        # Pre-condition: tables exist with rows.
        self.assertEqual(adapter._existing_tables("NODE"), {"Person", "City"})
        adapter.wipe_database()
        self.assertEqual(adapter._existing_tables("NODE"), set())
        self.assertEqual(adapter._existing_tables("REL"), set())

    async def test_cypher_binds_parameters_safely(self):
        """`$`-parameters bind values, not SQL/Cypher syntax — strings
        containing apostrophes or quotes must round-trip intact."""
        adapter = self._adapter()
        # `read_only=False` because we're CREATEing.
        await adapter.cypher(
            "CREATE (:Person {name: $name})",
            params={"name": 'O\'Brien "the great"'},
            read_only=False,
        )
        rows = await adapter.cypher(
            "MATCH (p:Person) WHERE p.name = $name RETURN p.name AS name",
            params={"name": 'O\'Brien "the great"'},
        )
        self.assertEqual(rows, [{"name": 'O\'Brien "the great"'}])

    async def test_update_entities_with_empty_list(self):
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        self.assertEqual(await adapter.update_entities([]), [])

    async def test_update_relations_with_empty_list(self):
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        self.assertEqual(await adapter.update_relations([]), [])

    async def test_update_knowledge_graph_empty_fields(self):
        """A KG with no entities and no relations still returns the
        canonical dict shape."""
        from synalinks.src.backend.pydantic.knowledge import KnowledgeGraph as KG

        class EmptyKG(KG):
            entities: List[Person] = []
            relations: List[LivesIn] = []

        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        result = await adapter.update_knowledge_graph(EmptyKG())
        self.assertEqual(result, {"entities": [], "relations": []})

    async def test_update_relations_persists_edge_properties(self):
        """When a Relation carries extra fields, they should survive
        through the MERGE + SET into the edge's stored properties."""
        from typing import Literal as _L

        class Knows(Relation):
            label: _L["Knows"]
            subj: Person
            obj: Person
            since: int

        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person],
                relation_models=[Knows],
                embedding_model=_StubEmbeddingModel({}),
                vector_dim=3,
            )
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        await adapter.update_relations(
            Knows(
                label="Knows",
                subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                obj=Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                since=2024,
            )
        )
        rows = adapter._con.execute(
            "MATCH ()-[r:Knows]->() RETURN r.since AS since"
        ).get_all()
        self.assertEqual(rows, [[2024]])

    def test_non_embedding_array_of_number_uses_double_array(self):
        """Regression: an entity with a float-list field that *isn't*
        the embedding should map to DOUBLE[], not FLOAT[vector_dim].
        Pre-fix that bug, any List[float] column collapsed to a
        fixed-size float vector and would reject runtime values of
        the wrong arity."""

        class WithExtraVector(Entity):
            label: Literal["WithExtraVector"]
            name: str
            scores: List[float] = []
            embedding: List[float] = []

        # An embedding model is required for the embedding column to
        # be added — without it the regression target (FLOAT[3])
        # wouldn't exist at all and the test would be moot.
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[WithExtraVector],
                embedding_model=_StubEmbeddingModel({}),
                vector_dim=3,
            )
        cols = {
            name: dtype
            for name, dtype in adapter._con.execute(
                "CALL TABLE_INFO('WithExtraVector') RETURN name, type"
            ).get_all()
        }
        # The actual embedding column gets the fixed-size FLOAT vector.
        self.assertEqual(cols["embedding"], "FLOAT[3]")
        # Any other List[float] field stays variable-length.
        self.assertEqual(cols["scores"], "DOUBLE[]")

    async def test_entity_similarity_search_with_multiple_queries(self):
        """Multi-query input: results merge across queries, with the
        best (smallest) distance per node id retained."""
        embedding_lookup = {
            "alice": [1.0, 0.0, 0.0],
            "paris": [0.0, 1.0, 0.0],
        }
        adapter = self._adapter(embedding_model=_StubEmbeddingModel(embedding_lookup))
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        results = await adapter.entity_similarity_search(
            ["alice", "paris"], label="Person", k=10
        )
        # Both rows show up — multi-query union — and Alice's row
        # carries her best (zero) distance from the "alice" query.
        # PK == name, so the result row carries the PK under its real
        # column name (no alias to "id").
        alice_row = next(r for r in results if r["name"] == "Alice")
        self.assertAlmostEqual(alice_row["distance"], 0.0, places=4)

    async def test_entity_hybrid_fts_search_threshold_filters(self):
        """Thresholds on either signal should drop rows whose source
        score fails the cutoff before RRF fusion."""
        adapter = self._adapter(
            embedding_model=_StubEmbeddingModel({"alice query": [1.0, 0.0, 0.0]}),
        )
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice Cypher", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob Marley", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        # An impossibly high FTS threshold + strict vector threshold
        # together leave at most the vector-best match (Alice).
        results = await adapter.entity_hybrid_fts_search(
            text_or_texts="alice query",
            label="Person",
            fulltext_threshold=9999.0,
            similarity_threshold=0.5,
        )
        # Alice's vector distance to [1,0,0] is 0; her FTS for "alice
        # query" gets filtered by the impossible threshold. Bob's
        # vector distance is well beyond 0.5 so he's filtered too.
        # Only Alice remains (vector-only RRF contribution).
        self.assertEqual(len(results), 1)
        self.assertIn("rrf_score", results[0])

    def test_close_is_idempotent(self):
        adapter = self._adapter()
        adapter.close()
        # Second call must not raise.
        adapter.close()

    def test_repr_includes_path(self):
        adapter = self._adapter()
        text = repr(adapter)
        self.assertIn("LadybugAdapter", text)
        self.assertIn(":memory:", text)

    async def test_bulk_update_entities_preserves_input_order(self):
        """A bulk call with multiple entities returns ids in the same
        order they were passed in — the UNWIND/UNION query reorders
        rows but the adapter remaps by pre-assigned id."""
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        ids = await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                Person(label="Person", name="Carol", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        self.assertEqual(len(ids), 3)
        # All three are distinct (no false dedup).
        self.assertEqual(len(set(ids)), 3)

        # PK == name, so the returned ids should be the names
        # themselves, in input order.
        self.assertEqual(ids, ["Alice", "Bob", "Carol"])

    async def test_bulk_update_entities_dedups_against_existing(self):
        """Existing nodes in the same batch should be matched by the
        bulk dedup pass, not re-inserted."""
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        # Seed the DB.
        first = await adapter.update_entities(
            Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])
        )
        # A batch mixing a near-duplicate with a brand-new entity.
        ids = await adapter.update_entities(
            [
                Person(label="Person", name="Alice-dup", embedding=[0.99, 0.01, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        self.assertEqual(ids[0], first)  # deduped to the existing Alice
        self.assertNotEqual(ids[1], first)  # Bob is fresh

        rows = adapter._con.execute("MATCH (p:Person) RETURN p.name").get_all()
        self.assertEqual(len(rows), 2)  # only Alice + Bob

    async def test_bulk_update_entities_handles_multiple_labels(self):
        """A batch with mixed labels splits into one query per label;
        all entities should land in the right NODE TABLE."""
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        ids = await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        self.assertEqual(len(ids), 3)
        persons = adapter._con.execute("MATCH (p:Person) RETURN p.name").get_all()
        cities = adapter._con.execute("MATCH (c:City) RETURN c.name").get_all()
        self.assertEqual(sorted(p[0] for p in persons), ["Alice", "Bob"])
        self.assertEqual(sorted(c[0] for c in cities), ["Paris"])

    async def test_bulk_update_relations_inserts_all_edges(self):
        """A batch of relations should produce one edge per row when
        endpoints have been seeded."""
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 0.0, 1.0]),
                City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
                City(label="City", name="Rome", embedding=[1.0, 1.0, 0.0]),
            ]
        )
        relations = [
            LivesIn(
                label="LivesIn",
                subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                obj=City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
            ),
            LivesIn(
                label="LivesIn",
                subj=Person(label="Person", name="Bob", embedding=[0.0, 0.0, 1.0]),
                obj=City(label="City", name="Rome", embedding=[1.0, 1.0, 0.0]),
            ),
        ]
        ids = await adapter.update_relations(relations)
        self.assertEqual(len(ids), 2)

        edge_count = adapter._con.execute(
            "MATCH ()-[r:LivesIn]->() RETURN count(r) AS c"
        ).get_all()
        self.assertEqual(edge_count, [[2]])

    async def test_entity_fulltext_search_finds_matching_term(self):
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice Cypher", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob Marley", embedding=[0.0, 1.0, 0.0]),
                Person(label="Person", name="Carol Doe", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        results = await adapter.entity_fulltext_search("Marley", label="Person", k=5)
        # Should return Bob (BM25 hit on "Marley") and no others.
        self.assertEqual(len(results), 1)
        self.assertIn("score", results[0])
        self.assertGreater(results[0]["score"], 0)

    async def test_entity_fulltext_search_rebuilds_after_inserts(self):
        """FTS index is a snapshot — newly inserted rows must show up
        in subsequent search results without an explicit rebuild call."""
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        await adapter.update_entities(
            Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])
        )
        first = await adapter.entity_fulltext_search("Alice", label="Person")
        self.assertEqual(len(first), 1)
        # Add another row and search again — should pick it up.
        await adapter.update_entities(
            Person(label="Person", name="Bob Marley", embedding=[0.0, 1.0, 0.0])
        )
        second = await adapter.entity_fulltext_search("Marley", label="Person")
        self.assertEqual(len(second), 1)

    async def test_entity_fulltext_search_threshold_filters(self):
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        await adapter.update_entities(
            Person(label="Person", name="Marley", embedding=[1.0, 0.0, 0.0])
        )
        # An impossible threshold should yield no rows.
        results = await adapter.entity_fulltext_search(
            "Marley", label="Person", threshold=999.0
        )
        self.assertEqual(results, [])

    async def test_entity_fulltext_search_requires_string_columns(self):
        """A label whose entity model has no string properties has no
        FTS index — search should raise with a clear message."""

        class NumberOnly(Entity):
            label: Literal["NumberOnly"]
            value: int = 0
            embedding: List[float] = []

        adapter = LadybugAdapter(
            uri="ladybug://:memory:",
            entity_models=[NumberOnly],
            vector_dim=3,
        )
        with self.assertRaisesRegex(ValueError, "No FTS index"):
            await adapter.entity_fulltext_search("anything", label="NumberOnly")

    async def test_entity_hybrid_fts_search_fuses_vector_and_fulltext(self):
        adapter = self._adapter(
            embedding_model=_StubEmbeddingModel({"alice query": [1.0, 0.0, 0.0]}),
        )
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice Cypher", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob Marley", embedding=[0.0, 1.0, 0.0]),
                Person(label="Person", name="Other", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        results = await adapter.entity_hybrid_fts_search(
            text_or_texts="alice query", label="Person", k=3
        )
        # All three should appear (vector ranks them all, fts ranks
        # Alice; RRF combines). Alice should rank first because she
        # tops both signals.
        self.assertEqual(len(results), 3)
        # PK == name: result row carries the PK under its real
        # column name (no alias to "id").
        self.assertEqual(results[0]["name"], "Alice Cypher")
        # Each result has an rrf_score, and FTS-only / vector-only
        # contributions are tagged when present.
        self.assertIn("rrf_score", results[0])

    async def test_entity_hybrid_fts_search_falls_back_without_embedding_model(
        self,
    ):
        """Without an embedding model, hybrid degrades to fulltext-only
        rather than raising — same shape as DuckDB's adapter."""
        adapter = LadybugAdapter(
            uri="ladybug://:memory:",
            entity_models=[Person, City],
            relation_models=[LivesIn],
            vector_dim=3,
        )
        # Insert directly via Cypher to bypass the entity-needs-embedding
        # heuristic, then rebuild the FTS index ourselves since we
        # bypassed update_entities (which is what normally triggers
        # the rebuild).
        adapter._con.execute("CREATE (:Person {name: 'Alice'})")
        adapter._rebuild_fts_index("Person")
        results = await adapter.entity_hybrid_fts_search(
            text_or_texts="Alice", label="Person"
        )
        self.assertEqual(len(results), 1)

    async def test_bulk_update_relations_idempotent_in_one_call(self):
        """The same edge passed twice in one batch should result in a
        single MERGE-d edge."""
        adapter = self._adapter(embedding_model=_StubEmbeddingModel({}))
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        rel = LivesIn(
            label="LivesIn",
            subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
            obj=City(label="City", name="Paris", embedding=[0.0, 1.0, 0.0]),
        )
        await adapter.update_relations([rel, rel])

        edge_count = adapter._con.execute(
            "MATCH ()-[r:LivesIn]->() RETURN count(r) AS c"
        ).get_all()
        self.assertEqual(edge_count, [[1]])


# ----------------------------------------------------------------------
# Reverse type mapper (Ladybug type string → JSON schema property)
# ----------------------------------------------------------------------


class CypherTypeToJsonPropertyTest(testing.TestCase):
    def test_scalar_types(self):
        self.assertEqual(
            _cypher_type_to_json_property("STRING", "name"),
            {"title": "Name", "type": "string"},
        )
        self.assertEqual(
            _cypher_type_to_json_property("INT64", "age"),
            {"title": "Age", "type": "integer"},
        )
        self.assertEqual(
            _cypher_type_to_json_property("DOUBLE", "score"),
            {"title": "Score", "type": "number"},
        )
        self.assertEqual(
            _cypher_type_to_json_property("BOOL", "active"),
            {"title": "Active", "type": "boolean"},
        )

    def test_date_and_timestamp_get_format_annotation(self):
        self.assertEqual(
            _cypher_type_to_json_property("DATE", "since"),
            {"title": "Since", "type": "string", "format": "date"},
        )
        self.assertEqual(
            _cypher_type_to_json_property("TIMESTAMP", "created_at"),
            {
                "title": "Created_At",
                "type": "string",
                "format": "date-time",
            },
        )

    def test_variable_length_arrays(self):
        self.assertEqual(
            _cypher_type_to_json_property("STRING[]", "tags"),
            {"title": "Tags", "type": "array", "items": {"type": "string"}},
        )
        self.assertEqual(
            _cypher_type_to_json_property("INT64[]", "ids"),
            {"title": "Ids", "type": "array", "items": {"type": "integer"}},
        )
        self.assertEqual(
            _cypher_type_to_json_property("DOUBLE[]", "scores"),
            {"title": "Scores", "type": "array", "items": {"type": "number"}},
        )
        self.assertEqual(
            _cypher_type_to_json_property("BOOL[]", "flags"),
            {"title": "Flags", "type": "array", "items": {"type": "boolean"}},
        )

    def test_fixed_size_array_collapses_to_plain_array(self):
        # FLOAT[4] (embedding) and INT64[8] both lose the length —
        # JSON schema has no fixed-length array annotation.
        self.assertEqual(
            _cypher_type_to_json_property("FLOAT[4]", "embedding"),
            {
                "title": "Embedding",
                "type": "array",
                "items": {"type": "number"},
            },
        )
        self.assertEqual(
            _cypher_type_to_json_property("INT64[8]", "coords"),
            {
                "title": "Coords",
                "type": "array",
                "items": {"type": "integer"},
            },
        )

    def test_json_maps_to_object(self):
        self.assertEqual(
            _cypher_type_to_json_property("JSON", "payload"),
            {"title": "Payload", "type": "object"},
        )

    def test_unknown_type_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            _cypher_type_to_json_property("INTERVAL", "duration")


# ----------------------------------------------------------------------
# get_symbolic_entities / get_symbolic_relations
# ----------------------------------------------------------------------


class SymbolicSchemaTest(testing.TestCase):
    def _adapter(self):
        # Same helper shape as LadybugAdapterTest, copy-pasted because
        # the surrounding class scope isn't shared.
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn],
                vector_dim=3,
            )

    def test_get_symbolic_entities_returns_one_model_per_node_table(self):
        adapter = self._adapter()
        models = adapter.get_symbolic_entities()
        self.assertEqual([m.name for m in models], ["City", "Person"])
        for m in models:
            self.assertIsInstance(m, SymbolicDataModel)

    def test_get_symbolic_entities_pins_label_with_const(self):
        """The recovered schema carries a ``label`` ``const`` so it can
        be used as a discriminated-union member, same way Pydantic v2
        emits it for the original Entity subclass."""
        adapter = self._adapter()
        schemas = {m.name: m.get_schema() for m in adapter.get_symbolic_entities()}
        person = schemas["Person"]
        self.assertEqual(person["title"], "Person")
        self.assertEqual(person["type"], "object")
        self.assertEqual(person["additionalProperties"], False)
        self.assertEqual(
            person["properties"]["label"],
            {
                "const": "Person",
                "default": "Person",
                "title": "Label",
                "type": "string",
            },
        )

    def test_get_symbolic_entities_hides_embedding_by_default(self):
        """The vector column is internal to the adapter — leaving it
        out matches DuckDB's ``get_symbolic_data_models`` default."""
        adapter = self._adapter()
        for m in adapter.get_symbolic_entities():
            self.assertNotIn("embedding", m.get_schema()["properties"])

    def test_get_symbolic_entities_includes_typed_properties(self):
        """Non-reserved columns survive the round-trip with their
        JSON-schema type. The PK (``name`` here) appears as a plain
        string property — no synthetic ``id`` is injected."""
        adapter = self._adapter()
        schemas = {m.name: m.get_schema() for m in adapter.get_symbolic_entities()}
        props = schemas["Person"]["properties"]
        self.assertNotIn("id", props)
        self.assertEqual(props["name"], {"title": "Name", "type": "string"})

    def test_get_symbolic_entities_empty_when_no_node_tables(self):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(uri="ladybug://:memory:", vector_dim=3)
        self.assertEqual(adapter.get_symbolic_entities(), [])

    def test_get_symbolic_relations_returns_one_model_per_rel_table(self):
        adapter = self._adapter()
        models = adapter.get_symbolic_relations()
        self.assertEqual([m.name for m in models], ["LivesIn"])
        self.assertIsInstance(models[0], SymbolicDataModel)

    def test_get_symbolic_relations_emits_ref_endpoints(self):
        """``subj`` / ``obj`` are ``$ref``-encoded into ``$defs``, with
        the endpoint schemas inlined — same shape Pydantic v2 emits
        for a hand-written Relation subclass."""
        adapter = self._adapter()
        schema = adapter.get_symbolic_relations()[0].get_schema()
        self.assertEqual(schema["title"], "LivesIn")
        self.assertEqual(schema["properties"]["subj"], {"$ref": "#/$defs/Person"})
        self.assertEqual(schema["properties"]["obj"], {"$ref": "#/$defs/City"})
        # The label const is on the relation itself, not on the
        # endpoints (those carry their own).
        self.assertEqual(
            schema["properties"]["label"],
            {
                "const": "LivesIn",
                "default": "LivesIn",
                "title": "Label",
                "type": "string",
            },
        )
        # Both endpoint schemas live in $defs.
        self.assertIn("Person", schema["$defs"])
        self.assertIn("City", schema["$defs"])
        self.assertEqual(schema["$defs"]["Person"]["title"], "Person")
        self.assertEqual(schema["$defs"]["City"]["title"], "City")

    def test_get_symbolic_relations_includes_edge_properties(self):
        """Edge attributes (not subj/obj/label) come back as typed
        properties, distinct from the endpoint schemas in $defs."""
        adapter = self._adapter()
        # The model-driven LivesIn has no extra attributes — extend
        # the table on the fly so the test exercises the property
        # round-trip without needing a separate Pydantic model.
        adapter._con.execute("ALTER TABLE LivesIn ADD since DATE")
        adapter._con.execute("ALTER TABLE LivesIn ADD score DOUBLE")
        schema = adapter.get_symbolic_relations()[0].get_schema()
        self.assertEqual(
            schema["properties"]["since"],
            {"title": "Since", "type": "string", "format": "date"},
        )
        self.assertEqual(
            schema["properties"]["score"],
            {"title": "Score", "type": "number"},
        )

    def test_get_symbolic_relations_handles_self_loop(self):
        """A rel table whose source and destination are the same
        label produces a schema with one ``$defs`` entry, referenced
        twice — not a duplicated def."""
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person],
                vector_dim=3,
            )
        adapter._con.execute("CREATE REL TABLE Knows(FROM Person TO Person)")
        schema = adapter.get_symbolic_relations()[0].get_schema()
        self.assertEqual(list(schema["$defs"].keys()), ["Person"])
        self.assertEqual(schema["properties"]["subj"], {"$ref": "#/$defs/Person"})
        self.assertEqual(schema["properties"]["obj"], {"$ref": "#/$defs/Person"})

    def test_get_symbolic_relations_empty_when_no_rel_tables(self):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                vector_dim=3,
            )
        self.assertEqual(adapter.get_symbolic_relations(), [])


# ----------------------------------------------------------------------
# relation_similarity_search / path_similarity_search
# ----------------------------------------------------------------------


class RelationSearchTest(testing.TestCase):
    """Shared fixture: two persons, two cities, two LivesIn edges
    (Alice→Paris, Bob→Rome). The embedding model is a stub that
    returns deterministic vectors so distances are predictable."""

    def _adapter(self, *, lookup=None):
        lookup = lookup or {}
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn],
                embedding_model=_StubEmbeddingModel(lookup),
                vector_dim=3,
            )

    async def _seed(self, adapter):
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
                City(label="City", name="Rome", embedding=[0.5, 0.5, 0.5]),
            ]
        )
        await adapter.update_relations(
            [
                LivesIn(
                    label="LivesIn",
                    subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                    obj=City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
                ),
                LivesIn(
                    label="LivesIn",
                    subj=Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                    obj=City(label="City", name="Rome", embedding=[0.5, 0.5, 0.5]),
                ),
            ]
        )

    async def test_relation_similarity_search_returns_matching_edges(self):
        """A subject-aligned query should land Alice→Paris first
        (zero distance on the subject side)."""
        adapter = self._adapter(lookup={"alice query": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        results = await adapter.relation_similarity_search("alice query", label="LivesIn")
        self.assertEqual(len(results), 2)
        # The first result is the Alice→Paris edge (smallest distance).
        self.assertEqual(results[0]["name"], "Alice")
        self.assertEqual(results[0]["name_1"], "Paris")
        self.assertLess(results[0]["distance"], results[1]["distance"])

    async def test_relation_similarity_search_flattens_pk_with_collision(
        self,
    ):
        """Person and City both use ``name`` as their PK column. The
        object side should land under ``name_1`` so the subject's
        ``name`` isn't clobbered."""
        adapter = self._adapter(lookup={"q": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        results = await adapter.relation_similarity_search("q", label="LivesIn")
        self.assertIn("name", results[0])
        self.assertIn("name_1", results[0])

    async def test_relation_similarity_search_emits_subj_rel_obj(self):
        adapter = self._adapter(lookup={"q": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        results = await adapter.relation_similarity_search("q", label="LivesIn")
        row = results[0]
        self.assertIn("subj", row)
        self.assertIn("rel", row)
        self.assertIn("obj", row)
        self.assertIn("distance", row)
        self.assertIn("matched_on", row)
        # subj/obj should be the underlying nodes with ``name`` set.
        self.assertEqual(row["subj"]["name"], row["name"])
        self.assertEqual(row["obj"]["name"], row["name_1"])

    async def test_relation_similarity_search_matched_on_tags_both_when_seen_on_each_side(
        self,
    ):
        """With k=10 against only 2 nodes per side, every edge is in
        both top-k lists → matched_on should report 'both'."""
        adapter = self._adapter(lookup={"q": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        results = await adapter.relation_similarity_search("q", label="LivesIn", k=10)
        self.assertTrue(all(r["matched_on"] == "both" for r in results))

    async def test_relation_similarity_search_threshold_filters(self):
        adapter = self._adapter(lookup={"q": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        # A very tight threshold leaves only the closest match
        # (Alice→Paris, subj_dist=0).
        results = await adapter.relation_similarity_search(
            "q", label="LivesIn", threshold=0.001
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Alice")

    async def test_relation_similarity_search_requires_embedding_model(self):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn],
                vector_dim=3,
            )
        with self.assertRaisesRegex(ValueError, "embedding_model"):
            await adapter.relation_similarity_search("x", label="LivesIn")

    async def test_relation_similarity_search_unknown_label_raises(self):
        adapter = self._adapter(lookup={"q": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        with self.assertRaisesRegex(ValueError, "SHOW_CONNECTION"):
            await adapter.relation_similarity_search("q", label="DoesNotExist")

    async def test_path_similarity_search_one_hop_matches_both_endpoints(self):
        """Subject-aligned + object-aligned query should land
        Alice→Paris with zero distances on both sides."""
        adapter = self._adapter(
            lookup={
                "alice query": [1.0, 0.0, 0.0],
                "paris query": [0.0, 0.0, 1.0],
            }
        )
        await self._seed(adapter)
        results = await adapter.path_similarity_search(
            "alice query",
            "paris query",
            subj_label="Person",
            obj_label="City",
            label="LivesIn",
        )
        self.assertGreaterEqual(len(results), 1)
        first = results[0]
        self.assertEqual(first["name"], "Alice")
        self.assertEqual(first["name_1"], "Paris")
        self.assertAlmostEqual(first["subj_distance"], 0.0, places=4)
        self.assertAlmostEqual(first["obj_distance"], 0.0, places=4)

    async def test_path_similarity_search_returns_full_path_shape(self):
        """Each result row must carry the full path: nodes list,
        rels list, hop length."""
        adapter = self._adapter(
            lookup={
                "alice query": [1.0, 0.0, 0.0],
                "paris query": [0.0, 0.0, 1.0],
            }
        )
        await self._seed(adapter)
        results = await adapter.path_similarity_search(
            "alice query",
            "paris query",
            subj_label="Person",
            obj_label="City",
            label="LivesIn",
        )
        row = results[0]
        self.assertIn("subj", row)
        self.assertIn("obj", row)
        self.assertIn("nodes", row)
        self.assertIn("rels", row)
        self.assertIn("length", row)
        self.assertIn("subj_distance", row)
        self.assertIn("obj_distance", row)
        # One-hop path → 2 nodes, 1 rel, length=1.
        self.assertEqual(row["length"], 1)
        self.assertEqual(len(row["nodes"]), 2)
        self.assertEqual(len(row["rels"]), 1)

    async def test_path_similarity_search_per_side_thresholds(self):
        """A tight subject threshold should drop Bob→Rome (Bob's
        subject distance from 'alice query' is large)."""
        adapter = self._adapter(
            lookup={
                "alice query": [1.0, 0.0, 0.0],
                "paris query": [0.0, 0.0, 1.0],
            }
        )
        await self._seed(adapter)
        results = await adapter.path_similarity_search(
            "alice query",
            "paris query",
            subj_label="Person",
            obj_label="City",
            label="LivesIn",
            subj_threshold=0.001,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Alice")

    async def test_path_similarity_search_drops_when_no_path_between_matches(
        self,
    ):
        """If subj_texts and obj_texts each match an endpoint, but no
        path of valid length connects them, AND-semantics drops the
        row."""
        adapter = self._adapter(
            lookup={
                "alice": [1.0, 0.0, 0.0],
                "rome": [0.5, 0.5, 0.5],
            }
        )
        await self._seed(adapter)
        # Alice→Paris exists, Bob→Rome exists, but Alice→Rome does not.
        results = await adapter.path_similarity_search(
            "alice",
            "rome",
            subj_label="Person",
            obj_label="City",
            label="LivesIn",
            subj_threshold=0.001,
            obj_threshold=0.001,
        )
        self.assertEqual(results, [])

    async def test_path_similarity_search_traverses_multi_hop_path(self):
        """A 2-hop chain via an intermediate Person should surface
        when max_hops >= 2 and no direct edge exists."""

        class Knows(Relation):
            label: Literal["Knows"]
            subj: Person
            obj: Person

        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[Knows, LivesIn],
                embedding_model=_StubEmbeddingModel(
                    {"alice query": [1.0, 0.0, 0.0], "paris query": [0.0, 0.0, 1.0]}
                ),
                vector_dim=3,
            )
        # Graph: Alice -Knows-> Bob -LivesIn-> Paris
        # Alice has no direct edge to Paris.
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        await adapter.update_relations(
            [
                Knows(
                    label="Knows",
                    subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                    obj=Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                ),
                LivesIn(
                    label="LivesIn",
                    subj=Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                    obj=City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
                ),
            ]
        )
        # Without label constraint, any 2-hop path counts.
        results = await adapter.path_similarity_search(
            "alice query",
            "paris query",
            subj_label="Person",
            obj_label="City",
            label=None,
            min_hops=2,
            max_hops=2,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["length"], 2)
        self.assertEqual(results[0]["name"], "Alice")
        self.assertEqual(results[0]["name_1"], "Paris")
        node_names = [n["name"] for n in results[0]["nodes"]]
        self.assertEqual(node_names, ["Alice", "Bob", "Paris"])

    async def test_path_similarity_search_label_constrains_every_hop(self):
        """When ``label`` is set, every hop must be of that label.
        A 2-hop mixed-label path should NOT come back."""

        class Knows(Relation):
            label: Literal["Knows"]
            subj: Person
            obj: Person

        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[Knows, LivesIn],
                embedding_model=_StubEmbeddingModel(
                    {"alice query": [1.0, 0.0, 0.0], "paris query": [0.0, 0.0, 1.0]}
                ),
                vector_dim=3,
            )
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        await adapter.update_relations(
            [
                Knows(
                    label="Knows",
                    subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                    obj=Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                ),
                LivesIn(
                    label="LivesIn",
                    subj=Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                    obj=City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
                ),
            ]
        )
        # LivesIn-only chain Alice→Paris doesn't exist (Alice's only
        # outgoing edge is Knows), so the constrained search returns 0.
        results = await adapter.path_similarity_search(
            "alice query",
            "paris query",
            subj_label="Person",
            obj_label="City",
            label="LivesIn",
            min_hops=2,
            max_hops=2,
        )
        self.assertEqual(results, [])

    async def test_path_similarity_search_requires_same_length_inputs(
        self,
    ):
        adapter = self._adapter(lookup={})
        with self.assertRaisesRegex(ValueError, "same length"):
            await adapter.path_similarity_search(
                ["a", "b"], ["c"], subj_label="Person", obj_label="City"
            )

    async def test_path_similarity_search_validates_hop_range(self):
        adapter = self._adapter(lookup={})
        with self.assertRaisesRegex(ValueError, "hop range"):
            await adapter.path_similarity_search(
                "a", "b", subj_label="Person", obj_label="City", min_hops=0
            )
        with self.assertRaisesRegex(ValueError, "hop range"):
            await adapter.path_similarity_search(
                "a",
                "b",
                subj_label="Person",
                obj_label="City",
                min_hops=3,
                max_hops=2,
            )

    async def test_path_similarity_search_empty_inputs_return_empty(self):
        adapter = self._adapter(lookup={})
        await self._seed(adapter)
        self.assertEqual(
            await adapter.path_similarity_search(
                "",
                "anything",
                subj_label="Person",
                obj_label="City",
                label="LivesIn",
            ),
            [],
        )
        self.assertEqual(
            await adapter.path_similarity_search(
                "anything",
                "",
                subj_label="Person",
                obj_label="City",
                label="LivesIn",
            ),
            [],
        )

    async def test_path_similarity_search_requires_embedding_model(self):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn],
                vector_dim=3,
            )
        with self.assertRaisesRegex(ValueError, "embedding_model"):
            await adapter.path_similarity_search(
                "x", "y", subj_label="Person", obj_label="City"
            )

    # ------------------------------------------------------------------
    # Hybrid variants (vec + BM25)
    # ------------------------------------------------------------------

    async def test_relation_hybrid_fts_search_ranks_edges_by_combined_signal(
        self,
    ):
        """Alice→Paris should rank first when 'alice' aligns to the
        subject side perfectly (FTS + vector both fire on Alice)."""
        adapter = self._adapter(lookup={"alice": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        results = await adapter.relation_hybrid_fts_search(
            text_or_texts="alice", label="LivesIn"
        )
        self.assertGreaterEqual(len(results), 1)
        first = results[0]
        self.assertEqual(first["name"], "Alice")
        self.assertEqual(first["name_1"], "Paris")
        # rrf_score is a sum of per-endpoint hybrid contributions.
        self.assertGreater(first["rrf_score"], 0)
        self.assertGreaterEqual(
            first["rrf_score"],
            first["subj_rrf_score"] + first["obj_rrf_score"] - 1e-9,
        )

    async def test_relation_hybrid_fts_search_returns_row_shape(self):
        adapter = self._adapter(lookup={"alice": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        results = await adapter.relation_hybrid_fts_search(
            text_or_texts="alice", label="LivesIn"
        )
        row = results[0]
        for key in (
            "subj",
            "rel",
            "obj",
            "rrf_score",
            "subj_rrf_score",
            "obj_rrf_score",
            "matched_on",
            "name",
            "name_1",
        ):
            self.assertIn(key, row)

    async def test_relation_hybrid_fts_search_matched_on_either(self):
        """If the query aligns only with the subject side (no obj
        match via FTS or vector), matched_on should be 'subj'."""
        # Embedding for the query is far from every city; FTS query
        # 'alice' matches Alice but no city.
        adapter = self._adapter(lookup={"alice": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        # Tight thresholds on the obj side so no city qualifies.
        results = await adapter.relation_hybrid_fts_search(
            text_or_texts="alice",
            label="LivesIn",
            similarity_threshold=0.001,
            fulltext_threshold=999.0,
        )
        # At most the Alice→Paris edge survives, and it must have
        # matched_on=subj (no obj contribution).
        alice_rows = [r for r in results if r["name"] == "Alice"]
        self.assertEqual(len(alice_rows), 1)
        self.assertEqual(alice_rows[0]["matched_on"], "subj")
        self.assertAlmostEqual(alice_rows[0]["obj_rrf_score"], 0.0)

    async def test_relation_hybrid_fts_search_falls_back_without_embedding_model(
        self,
    ):
        """No embedding model → FTS-only; ``rrf_score`` falls back to
        the per-endpoint BM25 score summed across sides."""
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn],
                vector_dim=3,
            )
        # Seed via direct cypher (no embedding required).
        adapter._con.execute(
            "CREATE (:Person {name: 'Alice'}); CREATE (:City {name: 'Paris'}); "
            "CREATE (:City {name: 'Rome'})"
        )
        adapter._con.execute(
            "MATCH (p:Person), (c:City) WHERE p.name='Alice' AND c.name='Paris' "
            "MERGE (p)-[:LivesIn]->(c)"
        )
        adapter._rebuild_fts_index("Person")
        adapter._rebuild_fts_index("City")
        results = await adapter.relation_hybrid_fts_search(
            text_or_texts="Alice", label="LivesIn"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Alice")

    async def test_path_hybrid_fts_search_one_hop_combines_both_sides(self):
        adapter = self._adapter(
            lookup={"alice": [1.0, 0.0, 0.0], "paris": [0.0, 0.0, 1.0]}
        )
        await self._seed(adapter)
        results = await adapter.path_hybrid_fts_search(
            subj_text_or_texts="alice",
            obj_text_or_texts="paris",
            subj_label="Person",
            obj_label="City",
            label="LivesIn",
        )
        self.assertGreaterEqual(len(results), 1)
        first = results[0]
        self.assertEqual(first["name"], "Alice")
        self.assertEqual(first["name_1"], "Paris")
        self.assertEqual(first["length"], 1)
        self.assertGreater(first["rrf_score"], 0)
        # Both endpoint sides contribute non-zero.
        self.assertGreater(first["subj_rrf_score"], 0)
        self.assertGreater(first["obj_rrf_score"], 0)

    async def test_path_hybrid_fts_search_returns_full_path_shape(self):
        adapter = self._adapter(
            lookup={"alice": [1.0, 0.0, 0.0], "paris": [0.0, 0.0, 1.0]}
        )
        await self._seed(adapter)
        results = await adapter.path_hybrid_fts_search(
            subj_text_or_texts="alice",
            obj_text_or_texts="paris",
            subj_label="Person",
            obj_label="City",
            label="LivesIn",
        )
        row = results[0]
        for key in (
            "subj",
            "obj",
            "nodes",
            "rels",
            "length",
            "rrf_score",
            "subj_rrf_score",
            "obj_rrf_score",
            "name",
            "name_1",
        ):
            self.assertIn(key, row)

    async def test_path_hybrid_fts_search_traverses_multi_hop(self):
        """Same multi-hop coverage as the similarity version: an
        intermediate Person bridging Alice and Paris should be
        traversed when max_hops>=2."""

        class Knows(Relation):
            label: Literal["Knows"]
            subj: Person
            obj: Person

        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[Knows, LivesIn],
                embedding_model=_StubEmbeddingModel(
                    {"alice": [1.0, 0.0, 0.0], "paris": [0.0, 0.0, 1.0]}
                ),
                vector_dim=3,
            )
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        await adapter.update_relations(
            [
                Knows(
                    label="Knows",
                    subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                    obj=Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                ),
                LivesIn(
                    label="LivesIn",
                    subj=Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
                    obj=City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
                ),
            ]
        )
        results = await adapter.path_hybrid_fts_search(
            subj_text_or_texts="alice",
            obj_text_or_texts="paris",
            subj_label="Person",
            obj_label="City",
            label=None,
            min_hops=2,
            max_hops=2,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["length"], 2)
        self.assertEqual(
            [n["name"] for n in results[0]["nodes"]],
            ["Alice", "Bob", "Paris"],
        )

    async def test_path_hybrid_fts_search_validates_hop_range(self):
        adapter = self._adapter(lookup={})
        with self.assertRaisesRegex(ValueError, "hop range"):
            await adapter.path_hybrid_fts_search(
                subj_text_or_texts="a",
                obj_text_or_texts="b",
                subj_label="Person",
                obj_label="City",
                min_hops=0,
            )

    async def test_path_hybrid_fts_search_empty_inputs_return_empty(self):
        adapter = self._adapter(lookup={})
        await self._seed(adapter)
        self.assertEqual(
            await adapter.path_hybrid_fts_search(
                subj_text_or_texts="",
                obj_text_or_texts="x",
                subj_label="Person",
                obj_label="City",
            ),
            [],
        )
        self.assertEqual(
            await adapter.path_hybrid_fts_search(
                subj_text_or_texts="x",
                obj_text_or_texts="",
                subj_label="Person",
                obj_label="City",
            ),
            [],
        )

    async def test_path_hybrid_fts_search_falls_back_without_embedding_model(
        self,
    ):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn],
                vector_dim=3,
            )
        adapter._con.execute(
            "CREATE (:Person {name: 'Alice'}); CREATE (:City {name: 'Paris'})"
        )
        adapter._con.execute(
            "MATCH (p:Person), (c:City) WHERE p.name='Alice' AND c.name='Paris' "
            "MERGE (p)-[:LivesIn]->(c)"
        )
        adapter._rebuild_fts_index("Person")
        adapter._rebuild_fts_index("City")
        results = await adapter.path_hybrid_fts_search(
            subj_text_or_texts="Alice",
            obj_text_or_texts="Paris",
            subj_label="Person",
            obj_label="City",
            label="LivesIn",
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Alice")
        self.assertEqual(results[0]["name_1"], "Paris")


# ----------------------------------------------------------------------
# Separate-keywords param on the hybrid methods
# ----------------------------------------------------------------------


class HybridKeywordsTest(testing.TestCase):
    """The hybrid surfaces accept ``keywords`` separate from the
    vector-side texts: vectors search semantically, BM25 searches
    lexically — the natural-language query that drives the vectors
    is typically not the keyword set you'd hand to BM25.
    """

    def _adapter(self, *, lookup=None):
        lookup = lookup or {}
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn],
                embedding_model=_StubEmbeddingModel(lookup),
                vector_dim=3,
            )

    async def test_entity_hybrid_uses_keywords_for_fts_branch(self):
        """When ``keywords`` differs from ``text_or_texts``, BM25 uses
        the keyword string. A vector text that lexically matches
        nothing + a keyword that matches Alice should still rank
        Alice via the FTS contribution."""
        adapter = self._adapter(
            # The vector text "nope" has no lexical match anywhere
            # (BM25 score 0), so any FTS contribution must come from
            # the separate keywords arg.
            lookup={"nope": [0.0, 0.0, 0.0]},
        )
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        results = await adapter.entity_hybrid_fts_search(
            text_or_texts="nope", keywords="Alice", label="Person"
        )
        # Alice gets non-zero FTS contribution from the keyword;
        # Bob doesn't.
        alice_row = next(r for r in results if r["name"] == "Alice")
        self.assertIn("fulltext_score", alice_row)
        self.assertGreater(alice_row["rrf_score"], 0)

    async def test_entity_hybrid_falls_back_to_texts_when_keywords_none(
        self,
    ):
        """Existing call sites that omit ``keywords`` keep working —
        the text is reused for both branches."""
        adapter = self._adapter(lookup={"Alice": [1.0, 0.0, 0.0]})
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        results = await adapter.entity_hybrid_fts_search(
            text_or_texts="Alice", label="Person"
        )
        alice_row = next(r for r in results if r["name"] == "Alice")
        # Both branches contributed → rrf_score reflects both.
        self.assertGreater(alice_row["rrf_score"], 0)
        self.assertIn("fulltext_score", alice_row)
        self.assertIn("distance", alice_row)

    async def test_entity_hybrid_validates_keyword_list_alignment(self):
        """List inputs must be the same length so per-query pairing
        is unambiguous."""
        adapter = self._adapter(lookup={})
        await adapter.update_entities(
            [Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0])]
        )
        with self.assertRaisesRegex(ValueError, "align"):
            await adapter.entity_hybrid_fts_search(
                text_or_texts=["a", "b"], keywords=["x"], label="Person"
            )

    async def test_relation_hybrid_forwards_keywords(self):
        adapter = self._adapter(lookup={"nope": [0.0, 0.0, 0.0]})
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        await adapter.update_relations(
            [
                LivesIn(
                    label="LivesIn",
                    subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                    obj=City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
                )
            ]
        )
        # vector text matches nothing; FTS keyword "Alice" should
        # still pull the subject side, contributing a subj_rrf_score.
        results = await adapter.relation_hybrid_fts_search(
            text_or_texts="nope", keywords="Alice", label="LivesIn"
        )
        self.assertGreaterEqual(len(results), 1)
        self.assertGreater(results[0]["subj_rrf_score"], 0)

    async def test_path_hybrid_uses_per_side_keywords(self):
        """``subj_keywords`` and ``obj_keywords`` drive the BM25
        branches independently on each endpoint."""
        adapter = self._adapter(lookup={"nope": [0.0, 0.0, 0.0]})
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
            ]
        )
        await adapter.update_relations(
            [
                LivesIn(
                    label="LivesIn",
                    subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                    obj=City(label="City", name="Paris", embedding=[0.0, 0.0, 1.0]),
                )
            ]
        )
        results = await adapter.path_hybrid_fts_search(
            subj_text_or_texts="nope",
            obj_text_or_texts="nope",
            subj_keywords="Alice",
            obj_keywords="Paris",
            subj_label="Person",
            obj_label="City",
            label="LivesIn",
        )
        # Both endpoint FTS branches fire; the path surfaces with
        # non-zero contributions on both sides.
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]["subj_rrf_score"], 0)
        self.assertGreater(results[0]["obj_rrf_score"], 0)


# ----------------------------------------------------------------------
# entity_regex_search / entity_hybrid_regex_search
# ----------------------------------------------------------------------


class WidePerson(Entity):
    """Extra string field so regex has something distinct to match on."""

    label: Literal["WidePerson"]
    name: str
    bio: str = ""
    embedding: List[float] = []


class RegexSearchTest(testing.TestCase):
    def _adapter(self, *, lookup=None):
        lookup = lookup or {}
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[WidePerson],
                embedding_model=_StubEmbeddingModel(lookup),
                vector_dim=3,
            )

    async def _seed(self, adapter):
        await adapter.update_entities(
            [
                WidePerson(
                    label="WidePerson",
                    name="Alice",
                    bio="engineer at MegaCorp",
                    embedding=[1.0, 0.0, 0.0],
                ),
                WidePerson(
                    label="WidePerson",
                    name="Bob",
                    bio="doctor",
                    embedding=[0.0, 1.0, 0.0],
                ),
                WidePerson(
                    label="WidePerson",
                    name="Carol",
                    bio="engineer",
                    embedding=[0.0, 0.0, 1.0],
                ),
            ]
        )

    async def test_entity_regex_search_matches_string_fields(self):
        adapter = self._adapter()
        await self._seed(adapter)
        rows = await adapter.entity_regex_search("engineer.*", label="WidePerson")
        names = sorted(r["name"] for r in rows)
        self.assertEqual(names, ["Alice", "Carol"])

    async def test_entity_regex_search_case_insensitive(self):
        adapter = self._adapter()
        await self._seed(adapter)
        rows = await adapter.entity_regex_search(
            "ENGINEER.*",
            label="WidePerson",
            case_sensitive=False,
        )
        names = sorted(r["name"] for r in rows)
        self.assertEqual(names, ["Alice", "Carol"])

    async def test_entity_regex_search_fields_whitelist(self):
        """Restrict the match to a subset of string columns."""
        adapter = self._adapter()
        await self._seed(adapter)
        # Pattern occurs in `bio` but the whitelist limits us to `name`,
        # so nothing matches.
        rows = await adapter.entity_regex_search(
            "engineer.*",
            label="WidePerson",
            fields=["name"],
        )
        self.assertEqual(rows, [])

    async def test_entity_regex_search_empty_pattern_returns_empty(self):
        adapter = self._adapter()
        await self._seed(adapter)
        self.assertEqual(await adapter.entity_regex_search("", label="WidePerson"), [])

    async def test_entity_regex_search_limit(self):
        adapter = self._adapter()
        await self._seed(adapter)
        rows = await adapter.entity_regex_search("engineer.*", label="WidePerson", k=1)
        self.assertEqual(len(rows), 1)

    async def test_entity_hybrid_regex_search_combines_signals(self):
        """Vector hits Alice (perfect embedding match); regex hits
        Carol — RRF should surface both with Carol higher (regex is a
        cleaner exact-shape signal here)."""
        adapter = self._adapter(
            lookup={"engineer": [1.0, 0.0, 0.0]},
        )
        await self._seed(adapter)
        rows = await adapter.entity_hybrid_regex_search(
            text_or_texts="engineer",
            pattern_or_patterns="Carol",
            label="WidePerson",
        )
        # Both Alice (vector) and Carol (regex) should appear.
        names = [r["name"] for r in rows]
        self.assertIn("Alice", names)
        self.assertIn("Carol", names)
        # Each row carries rrf_score.
        for r in rows:
            self.assertIn("rrf_score", r)

    async def test_entity_hybrid_regex_search_no_pattern_uses_similarity(
        self,
    ):
        """With no patterns, the hybrid degrades to plain similarity
        search — same shape as DuckDB's adapter."""
        adapter = self._adapter(lookup={"engineer": [1.0, 0.0, 0.0]})
        await self._seed(adapter)
        rows = await adapter.entity_hybrid_regex_search(
            text_or_texts="engineer",
            pattern_or_patterns=None,
            label="WidePerson",
        )
        # Result shape is the similarity_search shape — has `distance`.
        self.assertIn("distance", rows[0])

    async def test_entity_hybrid_regex_search_no_embedding_model_falls_back(
        self,
    ):
        """Without an embedding model, the vector half can't run; the
        hybrid should return regex-only rows, deduplicated."""
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[WidePerson],
                vector_dim=3,
            )
        adapter._con.execute("CREATE (:WidePerson {name: 'Alice', bio: 'engineer'})")
        adapter._rebuild_fts_index("WidePerson")
        rows = await adapter.entity_hybrid_regex_search(
            text_or_texts="anything",
            pattern_or_patterns="engineer",
            label="WidePerson",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "Alice")


# ----------------------------------------------------------------------
# detect_communities
# ----------------------------------------------------------------------


class Knows(Relation):
    """Single-label edge for community detection on a one-label graph."""

    label: Literal["Knows"]
    subj: Person
    obj: Person


class CommunityDetectionTest(testing.TestCase):
    def _adapter(self):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person],
                relation_models=[Knows],
                vector_dim=3,
            )

    async def _seed_two_clusters_plus_isolated(self, adapter):
        # Cluster A: {Alice, Bob, Carol} fully connected.
        # Cluster B: {Xenia, Yara} two-node chain.
        # Isolated: Zeke (no edges).
        await adapter.update_entities(
            [
                Person(label="Person", name=n)
                for n in ["Alice", "Bob", "Carol", "Xenia", "Yara", "Zeke"]
            ]
        )
        await adapter.update_relations(
            [
                Knows(
                    label="Knows",
                    subj=Person(label="Person", name=s),
                    obj=Person(label="Person", name=o),
                )
                for s, o in [
                    ("Alice", "Bob"),
                    ("Bob", "Carol"),
                    ("Carol", "Alice"),
                    ("Xenia", "Yara"),
                ]
            ]
        )

    async def test_detect_communities_louvain_returns_knowledge_graphs(self):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        result = await adapter.detect_communities(algorithm="louvain")
        # The return is a KnowledgeGraphs wrapper with .knowledge_graphs.
        self.assertEqual(len(result.knowledge_graphs), 3)
        # Each community has its own entities; collected names cover
        # all seeded persons and the partition is disjoint.
        all_names = [e.name for kg in result.knowledge_graphs for e in kg.entities]
        self.assertEqual(
            sorted(all_names),
            ["Alice", "Bob", "Carol", "Xenia", "Yara", "Zeke"],
        )

    async def test_detect_communities_isolated_node_is_singleton(self):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        result = await adapter.detect_communities(algorithm="louvain")
        # Find the community containing Zeke; it should be a singleton.
        zeke_kg = next(
            kg
            for kg in result.knowledge_graphs
            if any(e.name == "Zeke" for e in kg.entities)
        )
        self.assertEqual(len(zeke_kg.entities), 1)
        self.assertEqual(zeke_kg.relations, [])

    async def test_detect_communities_groups_edges_with_their_endpoints(
        self,
    ):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        result = await adapter.detect_communities(algorithm="louvain")
        # The 3-node cluster has 3 edges, the 2-node cluster has 1.
        rel_counts = sorted(len(kg.relations) for kg in result.knowledge_graphs)
        self.assertEqual(rel_counts, [0, 1, 3])

    async def test_detect_communities_reconstructs_registered_subclass(
        self,
    ):
        """The returned entities should be Person instances (not the
        base Entity) so the user's fields like ``name`` are
        preserved on round-trip."""
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        result = await adapter.detect_communities(algorithm="louvain")
        for kg in result.knowledge_graphs:
            for entity in kg.entities:
                self.assertIsInstance(entity, Person)
                self.assertTrue(hasattr(entity, "name"))

    async def test_detect_communities_wcc_supports_multi_label(self):
        """WCC isn't constrained to a single node label the way
        Louvain is. Connect Persons to Cities and verify everything
        in one connected piece ends up in one community."""

        class LivesIn2(Relation):
            label: Literal["LivesIn2"]
            subj: Person
            obj: City

        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn2],
                vector_dim=3,
            )
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice"),
                City(label="City", name="Paris"),
                Person(label="Person", name="Lonely"),
            ]
        )
        await adapter.update_relations(
            [
                LivesIn2(
                    label="LivesIn2",
                    subj=Person(label="Person", name="Alice"),
                    obj=City(label="City", name="Paris"),
                )
            ]
        )
        result = await adapter.detect_communities(algorithm="weakly_connected_components")
        # 2 components: {Alice, Paris} connected, {Lonely} isolated.
        self.assertEqual(len(result.knowledge_graphs), 2)
        sizes = sorted(len(kg.entities) for kg in result.knowledge_graphs)
        self.assertEqual(sizes, [1, 2])

    async def test_detect_communities_louvain_rejects_multi_label(self):
        """Louvain in Ladybug requires a single node label; the
        adapter should raise instead of letting the engine error."""

        class LivesIn3(Relation):
            label: Literal["LivesIn3"]
            subj: Person
            obj: City

        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn3],
                vector_dim=3,
            )
        with self.assertRaisesRegex(ValueError, "Louvain only supports"):
            await adapter.detect_communities(algorithm="louvain")

    async def test_detect_communities_node_labels_whitelist(self):
        """Restricting via node_labels should filter which entities
        get clustered."""
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        # ``Person`` is the only label registered, so the whitelist
        # behaves like the default, but the test exercises the
        # filtering code path with an explicit name.
        result = await adapter.detect_communities(
            algorithm="louvain", node_labels=["Person"]
        )
        self.assertEqual(len(result.knowledge_graphs), 3)

    async def test_detect_communities_unknown_algorithm_raises(self):
        adapter = self._adapter()
        with self.assertRaisesRegex(ValueError, "Unknown algorithm"):
            await adapter.detect_communities(algorithm="notreal")

    async def test_detect_communities_empty_graph(self):
        adapter = self._adapter()
        result = await adapter.detect_communities(algorithm="louvain")
        self.assertEqual(result.knowledge_graphs, [])

    async def test_detect_communities_max_iterations_is_forwarded(self):
        """``max_iterations=1`` should be accepted by Ladybug for both
        Louvain and WCC — proves the kwarg fragment reaches the
        procedure call without a binder exception."""
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        result = await adapter.detect_communities(algorithm="louvain", max_iterations=1)
        # With a tiny iteration cap Louvain may not fully converge,
        # but the call must still succeed and return the seeded nodes.
        all_names = sorted(e.name for kg in result.knowledge_graphs for e in kg.entities)
        self.assertEqual(
            all_names,
            ["Alice", "Bob", "Carol", "Xenia", "Yara", "Zeke"],
        )

    async def test_detect_communities_strongly_connected_components(self):
        """SCC requires bidirectional reachability — a one-way chain
        should land in singleton components, distinct from WCC's
        single-component view of the same edges."""
        adapter = self._adapter()
        # One-way chain: Alice -> Bob -> Carol (no back edges).
        await adapter.update_entities(
            [Person(label="Person", name=n) for n in ["Alice", "Bob", "Carol"]]
        )
        await adapter.update_relations(
            [
                Knows(
                    label="Knows",
                    subj=Person(label="Person", name=s),
                    obj=Person(label="Person", name=o),
                )
                for s, o in [("Alice", "Bob"), ("Bob", "Carol")]
            ]
        )
        scc = await adapter.detect_communities(algorithm="strongly_connected_components")
        # Three singleton components — no cycles, so no SCC has size > 1.
        sizes = sorted(len(kg.entities) for kg in scc.knowledge_graphs)
        self.assertEqual(sizes, [1, 1, 1])

        wcc = await adapter.detect_communities(algorithm="weakly_connected_components")
        # One component for the chain — WCC ignores direction.
        sizes = sorted(len(kg.entities) for kg in wcc.knowledge_graphs)
        self.assertEqual(sizes, [3])


# ----------------------------------------------------------------------
# pagerank
# ----------------------------------------------------------------------


class PageRankTest(testing.TestCase):
    def _adapter(self):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person],
                relation_models=[Knows],
                vector_dim=3,
            )

    async def _seed_hub_and_spokes(self, adapter):
        # Hub-and-spoke: Hub has 3 incoming edges (Alice, Bob, Carol →
        # Hub). PageRank should rank Hub first.
        await adapter.update_entities(
            [Person(label="Person", name=n) for n in ["Hub", "Alice", "Bob", "Carol"]]
        )
        await adapter.update_relations(
            [
                Knows(
                    label="Knows",
                    subj=Person(label="Person", name=s),
                    obj=Person(label="Person", name="Hub"),
                )
                for s in ["Alice", "Bob", "Carol"]
            ]
        )

    async def test_pagerank_returns_rows_sorted_desc(self):
        adapter = self._adapter()
        await self._seed_hub_and_spokes(adapter)
        rows = await adapter.pagerank()
        ranks = [r["rank"] for r in rows]
        self.assertEqual(ranks, sorted(ranks, reverse=True))
        self.assertEqual(rows[0]["node"]["name"], "Hub")

    async def test_pagerank_row_shape_uses_real_pk_column(self):
        adapter = self._adapter()
        await self._seed_hub_and_spokes(adapter)
        rows = await adapter.pagerank()
        row = rows[0]
        # PK column for ``Person`` is ``name`` (first non-label property),
        # not aliased to ``id``.
        self.assertIn("name", row)
        self.assertEqual(row["label"], "Person")
        self.assertIn("node", row)
        self.assertIn("rank", row)
        self.assertEqual(row["name"], row["node"]["name"])

    async def test_pagerank_respects_k_cap(self):
        adapter = self._adapter()
        await self._seed_hub_and_spokes(adapter)
        rows = await adapter.pagerank(k=2)
        self.assertEqual(len(rows), 2)

    async def test_pagerank_empty_graph(self):
        adapter = self._adapter()
        result = await adapter.pagerank()
        self.assertEqual(result, [])

    async def test_pagerank_node_labels_whitelist_filters(self):
        adapter = self._adapter()
        await self._seed_hub_and_spokes(adapter)
        # Person is the only label; the whitelist exercises the
        # filtering path with an explicit name.
        rows = await adapter.pagerank(node_labels=["Person"])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["label"] == "Person" for r in rows))

    async def test_pagerank_unknown_label_falls_through_to_empty(self):
        adapter = self._adapter()
        await self._seed_hub_and_spokes(adapter)
        # ``Nonexistent`` gets filtered out by _resolve_projection_labels,
        # leaving zero node tables → empty result.
        rows = await adapter.pagerank(node_labels=["Nonexistent"])
        self.assertEqual(rows, [])

    async def test_pagerank_forwards_tolerance_and_normalize_initial(self):
        """Tight ``tolerance`` plus explicit ``normalize_initial``
        should reach Ladybug without a binder error and still produce
        a Hub-first ranking on the hub-and-spoke graph."""
        adapter = self._adapter()
        await self._seed_hub_and_spokes(adapter)
        rows = await adapter.pagerank(
            tolerance=1e-9,
            normalize_initial=True,
        )
        self.assertEqual(rows[0]["node"]["name"], "Hub")

    async def test_pagerank_low_max_iterations_still_succeeds(self):
        """Even ``max_iterations=1`` should be accepted by Ladybug —
        proves the kwarg fragment is forwarded correctly."""
        adapter = self._adapter()
        await self._seed_hub_and_spokes(adapter)
        rows = await adapter.pagerank(max_iterations=1)
        # Doesn't matter who wins after a single iteration; the call
        # must succeed and return every node.
        self.assertEqual(len(rows), 4)


# ----------------------------------------------------------------------
# Index-option and search-time params (HNSW efs / FTS conjunctive+b /
# CREATE_VECTOR_INDEX HNSW build / CREATE_FTS_INDEX tokenizer-stemmer)
# ----------------------------------------------------------------------


class IndexOptionsAndSearchParamsTest(testing.TestCase):
    """Tests for the Ladybug-native build/search knobs:
    ``mu`` / ``ml`` / ``pu`` / ``efc`` and
    ``stemmer`` / ``stopwords`` / ``tokenizer`` at construction;
    ``ef_search`` / ``conjunctive`` / ``bm25_b`` per search call."""

    def _adapter(self, **kwargs):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person],
                embedding_model=_StubEmbeddingModel({"alice": [1.0, 0.0, 0.0]}),
                vector_dim=3,
                **kwargs,
            )

    def test_hnsw_params_stored_on_adapter(self):
        adapter = self._adapter(mu=30, ml=10, pu=0.5, efc=200)
        self.assertEqual(adapter.mu, 30)
        self.assertEqual(adapter.ml, 10)
        self.assertEqual(adapter.pu, 0.5)
        self.assertEqual(adapter.efc, 200)

    def test_fts_build_params_stored_on_adapter(self):
        adapter = self._adapter(stemmer="porter", stopwords=None, tokenizer="simple")
        self.assertEqual(adapter.stemmer, "porter")
        self.assertIsNone(adapter.stopwords)
        self.assertEqual(adapter.tokenizer, "simple")

    def test_hnsw_params_reach_ladybug_at_create_time(self):
        """``efc=50`` should make the index create succeed; if the
        fragment was malformed Ladybug would raise during __init__."""
        adapter = self._adapter(efc=50, mu=20)
        rows = adapter._con.execute(
            "CALL SHOW_INDEXES() RETURN table_name, index_name"
        ).get_all()
        self.assertTrue(any(r[0] == "Person" and r[1] == "person_vec" for r in rows))

    def test_fts_build_params_reach_ladybug_at_create_time(self):
        """A valid ``stemmer`` should be accepted at index-create
        time. Confirms the DDL kwargs fragment is well-formed."""
        adapter = self._adapter(stemmer="porter")
        rows = adapter._con.execute(
            "CALL SHOW_INDEXES() RETURN table_name, index_name"
        ).get_all()
        self.assertTrue(any(r[0] == "Person" and r[1] == "person_fts" for r in rows))

    def test_none_build_params_omit_from_ddl(self):
        """``None`` defaults must not appear in the rendered DDL
        kwargs — otherwise Ladybug would see ``stemmer := null`` and
        reject the unknown literal."""
        adapter = self._adapter()  # every build param defaults to None
        # If any None had leaked into the DDL string this would have
        # raised during __init__; reaching here proves the omit-on-
        # None path works.
        self.assertIsNone(adapter.stemmer)
        self.assertIsNone(adapter.mu)

    async def test_entity_similarity_search_forwards_ef_search(self):
        """``ef_search=200`` must not raise — Ladybug accepts ``efs``
        as a query-time HNSW kwarg."""
        adapter = self._adapter()
        await adapter.update_entities(
            [Person(label="Person", name="alice", embedding=[1.0, 0.0, 0.0])]
        )
        rows = await adapter.entity_similarity_search(
            "alice", label="Person", k=5, ef_search=200
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "alice")

    async def test_entity_fulltext_search_forwards_conjunctive_and_b(self):
        adapter = self._adapter()
        await adapter.update_entities(
            [Person(label="Person", name="alice", embedding=[1.0, 0.0, 0.0])]
        )
        # Both kwargs must reach the engine without binder error.
        rows = await adapter.entity_fulltext_search(
            "alice",
            label="Person",
            k=5,
            conjunctive=True,
            bm25_b=0.5,
        )
        self.assertEqual(len(rows), 1)

    async def test_entity_hybrid_fts_forwards_all_new_params(self):
        adapter = self._adapter()
        await adapter.update_entities(
            [Person(label="Person", name="alice", embedding=[1.0, 0.0, 0.0])]
        )
        # The hybrid path takes both surfaces' kwargs and merges them
        # — a successful call exercises the full plumbing.
        rows = await adapter.entity_hybrid_fts_search(
            text_or_texts="alice",
            label="Person",
            k=5,
            ef_search=100,
            conjunctive=True,
            bm25_b=0.6,
        )
        self.assertGreaterEqual(len(rows), 1)


# ----------------------------------------------------------------------
# GraphRAG-style local search
# ----------------------------------------------------------------------


class LocalGraphSearchTest(testing.TestCase):
    def _adapter(self, embedding_model):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person, City],
                relation_models=[LivesIn, Knows],
                embedding_model=embedding_model,
                vector_dim=3,
            )

    async def _seed(self, adapter):
        # Alice [1,0,0] -LivesIn-> NYC [0,1,0]
        # Alice          -Knows-> Bob  [0,0,1]
        # Hermit [.5,.5,.5] has no edges (isolated).
        await adapter.update_entities(
            [
                Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                Person(label="Person", name="Bob", embedding=[0.0, 0.0, 1.0]),
                Person(label="Person", name="Hermit", embedding=[0.5, 0.5, 0.5]),
                City(label="City", name="NYC", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        await adapter.update_relations(
            [
                LivesIn(
                    label="LivesIn",
                    subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                    obj=City(label="City", name="NYC", embedding=[0.0, 1.0, 0.0]),
                ),
                Knows(
                    label="Knows",
                    subj=Person(label="Person", name="Alice", embedding=[1.0, 0.0, 0.0]),
                    obj=Person(label="Person", name="Bob", embedding=[0.0, 0.0, 1.0]),
                ),
            ]
        )

    async def test_local_graph_search_returns_seed_neighborhood(self):
        adapter = self._adapter(_StubEmbeddingModel({"alice": [1.0, 0.0, 0.0]}))
        await self._seed(adapter)
        kg = await adapter.local_graph_search("alice", label="Person", max_hops=1, k=1)
        # Alice's 1-hop neighbourhood: NYC (LivesIn) and Bob (Knows).
        self.assertEqual(sorted(e.name for e in kg.entities), ["Alice", "Bob", "NYC"])
        rels = sorted((r.label, r.subj.name, r.obj.name) for r in kg.relations)
        self.assertEqual(rels, [("Knows", "Alice", "Bob"), ("LivesIn", "Alice", "NYC")])

    async def test_local_graph_search_reconstructs_typed_subclasses_and_direction(
        self,
    ):
        """Entities round-trip to their registered subclass and each
        relation keeps the schema's subj/obj direction even though the
        traversal is undirected."""
        adapter = self._adapter(_StubEmbeddingModel({"alice": [1.0, 0.0, 0.0]}))
        await self._seed(adapter)
        kg = await adapter.local_graph_search("alice", label="Person", max_hops=1, k=1)
        by_name = {e.name: e for e in kg.entities}
        self.assertIsInstance(by_name["Alice"], Person)
        self.assertIsInstance(by_name["NYC"], City)
        livesin = next(r for r in kg.relations if r.label == "LivesIn")
        self.assertIsInstance(livesin.subj, Person)
        self.assertIsInstance(livesin.obj, City)
        self.assertEqual(livesin.subj.name, "Alice")
        self.assertEqual(livesin.obj.name, "NYC")

    async def test_local_graph_search_isolated_seed_returns_only_itself(self):
        adapter = self._adapter(_StubEmbeddingModel({"hermit": [0.5, 0.5, 0.5]}))
        await self._seed(adapter)
        kg = await adapter.local_graph_search("hermit", label="Person", max_hops=2, k=1)
        self.assertEqual([e.name for e in kg.entities], ["Hermit"])
        self.assertEqual(kg.relations, [])

    async def test_local_graph_search_rel_label_constrains_hops(self):
        adapter = self._adapter(_StubEmbeddingModel({"alice": [1.0, 0.0, 0.0]}))
        await self._seed(adapter)
        # Constraining to ``Knows`` reaches Bob but not NYC (LivesIn).
        kg = await adapter.local_graph_search(
            "alice", label="Person", max_hops=1, k=1, rel_label="Knows"
        )
        self.assertEqual(sorted(e.name for e in kg.entities), ["Alice", "Bob"])
        self.assertEqual(
            [(r.label, r.subj.name, r.obj.name) for r in kg.relations],
            [("Knows", "Alice", "Bob")],
        )

    async def test_local_graph_search_threshold_filters_distant_seeds(self):
        # A query text the stub doesn't know maps to [0,0,0], which is
        # far from every stored vector; a tight threshold drops it so
        # nothing seeds the expansion.
        adapter = self._adapter(_StubEmbeddingModel({}))
        await self._seed(adapter)
        kg = await adapter.local_graph_search(
            "unknown", label="Person", max_hops=1, k=3, threshold=0.1
        )
        self.assertEqual(kg.entities, [])
        self.assertEqual(kg.relations, [])

    async def test_local_graph_search_requires_embedding_model(self):
        adapter = self._adapter(embedding_model=None)
        await self._seed(adapter)
        with self.assertRaisesRegex(ValueError, "requires an embedding_model"):
            await adapter.local_graph_search("alice", label="Person")

    async def test_local_graph_search_validates_max_hops(self):
        adapter = self._adapter(_StubEmbeddingModel({"alice": [1.0, 0.0, 0.0]}))
        await self._seed(adapter)
        with self.assertRaisesRegex(ValueError, "max_hops must be >= 1"):
            await adapter.local_graph_search("alice", label="Person", max_hops=0)

    async def test_local_graph_search_empty_input_returns_empty_graph(self):
        adapter = self._adapter(_StubEmbeddingModel({}))
        await self._seed(adapter)
        kg = await adapter.local_graph_search("", label="Person")
        self.assertEqual(kg.entities, [])
        self.assertEqual(kg.relations, [])


# ----------------------------------------------------------------------
# GraphRAG-style global search (community materialization + aggregation)
# ----------------------------------------------------------------------


class GlobalGraphSearchTest(testing.TestCase):
    def _adapter(self):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            return LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Person],
                relation_models=[Knows],
                vector_dim=3,
            )

    async def _seed_two_clusters_plus_isolated(self, adapter):
        # Cluster A: {Alice, Bob, Carol} triangle. Cluster B: {Xenia,
        # Yara} chain. Isolated: {Zeke}. Mirrors CommunityDetectionTest.
        await adapter.update_entities(
            [
                Person(label="Person", name=n)
                for n in ["Alice", "Bob", "Carol", "Xenia", "Yara", "Zeke"]
            ]
        )
        await adapter.update_relations(
            [
                Knows(
                    label="Knows",
                    subj=Person(label="Person", name=s),
                    obj=Person(label="Person", name=o),
                )
                for s, o in [
                    ("Alice", "Bob"),
                    ("Bob", "Carol"),
                    ("Carol", "Alice"),
                    ("Xenia", "Yara"),
                ]
            ]
        )

    async def test_build_communities_stamps_every_node(self):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        stamped = await adapter.build_communities(algorithm="louvain")
        self.assertEqual(stamped, 6)
        # Every node now carries a non-null community and a rank.
        rows = adapter._con.execute(
            "MATCH (p:Person) WHERE p.community IS NOT NULL AND p.rank IS NOT NULL "
            "RETURN count(p)"
        ).get_all()
        self.assertEqual(rows[0][0], 6)

    async def test_build_communities_is_idempotent_and_keeps_types(self):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        first = await adapter.build_communities(algorithm="louvain")
        second = await adapter.build_communities(algorithm="louvain")
        self.assertEqual(first, second)
        # The reserved columns must not leak into reconstruction: a
        # detect_communities run after stamping still yields Person
        # instances (not bare Entity fallbacks).
        result = await adapter.detect_communities(algorithm="louvain")
        for kg in result.knowledge_graphs:
            for entity in kg.entities:
                self.assertIsInstance(entity, Person)

    async def test_global_graph_search_aggregates_per_community(self):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        await adapter.build_communities(algorithm="louvain")
        rows = await adapter.global_graph_search(k=10)
        # Three communities: sizes 3 (triangle), 2 (chain), 1 (isolated).
        self.assertEqual(sorted(r["size"] for r in rows), [1, 2, 3])
        for r in rows:
            self.assertIn("community", r)
            self.assertIn("total_rank", r)
            self.assertEqual(len(r["members"]), r["size"])
        # Ordered by aggregate importance, descending.
        ranks = [r["total_rank"] for r in rows]
        self.assertEqual(ranks, sorted(ranks, reverse=True))

    async def test_global_graph_search_before_build_returns_empty(self):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        # No build_communities call → no community column → empty result.
        rows = await adapter.global_graph_search()
        self.assertEqual(rows, [])

    async def test_global_graph_search_k_caps_communities(self):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        await adapter.build_communities(algorithm="louvain")
        rows = await adapter.global_graph_search(k=1)
        self.assertEqual(len(rows), 1)

    async def test_global_graph_search_members_per_community_caps(self):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        await adapter.build_communities(algorithm="louvain")
        rows = await adapter.global_graph_search(members_per_community=1)
        for r in rows:
            self.assertLessEqual(len(r["members"]), 1)

    async def test_build_communities_without_pagerank_zeroes_rank(self):
        adapter = self._adapter()
        await self._seed_two_clusters_plus_isolated(adapter)
        await adapter.build_communities(algorithm="louvain", with_pagerank=False)
        rows = await adapter.global_graph_search()
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(r["total_rank"], 0.0)

    async def test_build_communities_rejects_reserved_field_collision(self):
        """A model that declares ``community`` as a real field must not
        be silently clobbered by the stamping."""

        class Tagged(Entity):
            label: Literal["Tagged"]
            name: str
            community: int = 0
            embedding: List[float] = []

        class TaggedKnows(Relation):
            label: Literal["TaggedKnows"]
            subj: Tagged
            obj: Tagged

        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            adapter = LadybugAdapter(
                uri="ladybug://:memory:",
                entity_models=[Tagged],
                relation_models=[TaggedKnows],
                vector_dim=3,
            )
        # One edge so Louvain has an edge table to project; the guard
        # we're testing fires after clustering, on the stamping step.
        await adapter.update_entities(
            [
                Tagged(label="Tagged", name="x"),
                Tagged(label="Tagged", name="y"),
            ]
        )
        await adapter.update_relations(
            [
                TaggedKnows(
                    label="TaggedKnows",
                    subj=Tagged(label="Tagged", name="x"),
                    obj=Tagged(label="Tagged", name="y"),
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "reserved field"):
            await adapter.build_communities(algorithm="louvain")


# ----------------------------------------------------------------------
# Free-form graphs: dynamic table creation (no predeclared models)
# ----------------------------------------------------------------------


class FreeNode(Entity):
    """Generic entity: open string label (no Literal), a name property."""

    name: str
    embedding: List[float] = []


class FreeEdge(Relation):
    """Generic relation: open string label, generic endpoints."""

    subj: FreeNode
    obj: FreeNode


class FreeFormGraphTest(testing.TestCase):
    def _adapter(self, *, embedding_model=None):
        with patch(
            "synalinks.src.knowledge_bases.graph_database_adapters."
            "ladybug_adapter._get_em",
            side_effect=lambda x: x,
        ):
            # No entity_models / relation_models: every table is created
            # on demand from the data that arrives.
            return LadybugAdapter(
                uri="ladybug://:memory:",
                embedding_model=embedding_model,
                vector_dim=3,
            )

    async def test_entities_create_node_tables_on_demand(self):
        adapter = self._adapter()
        await adapter.update_entities(
            [
                FreeNode(label="Person", name="Ada"),
                FreeNode(label="Field", name="Computing"),
            ]
        )
        # A table per distinct label value, created without any model.
        self.assertEqual(adapter._existing_tables("NODE"), {"Person", "Field"})
        rows = adapter._con.execute("MATCH (p:Person) RETURN p.name AS n").get_all()
        self.assertEqual(rows, [["Ada"]])

    async def test_knowledge_graph_creates_node_and_rel_tables_on_demand(self):
        from synalinks.src.backend.pydantic.knowledge import KnowledgeGraph as KG

        class FreeGraph(KG):
            entities: List[FreeNode]
            relations: List[FreeEdge]

        adapter = self._adapter()
        graph = FreeGraph(
            entities=[
                FreeNode(label="Person", name="Ada"),
                FreeNode(label="Field", name="Computing"),
            ],
            relations=[
                FreeEdge(
                    label="PIONEER_OF",
                    subj=FreeNode(label="Person", name="Ada"),
                    obj=FreeNode(label="Field", name="Computing"),
                )
            ],
        )
        await adapter.update_knowledge_graph(graph)

        self.assertEqual(adapter._existing_tables("NODE"), {"Person", "Field"})
        self.assertIn("PioneerOf", adapter._existing_tables("REL"))
        edge = adapter._con.execute(
            "MATCH (a)-[r]->(b) RETURN a.name AS s, b.name AS o"
        ).get_all()
        self.assertEqual(edge, [["Ada", "Computing"]])

    async def test_read_back_preserves_free_form_properties(self):
        """Reconstructed free-form entities keep their properties even
        though no model was registered for their labels."""
        adapter = self._adapter(
            embedding_model=_StubEmbeddingModel(
                {
                    "Ada": [1.0, 0.0, 0.0],
                    "Computing": [0.0, 1.0, 0.0],
                }
            )
        )
        await adapter.update_entities(
            [
                FreeNode(label="Person", name="Ada", embedding=[1.0, 0.0, 0.0]),
                FreeNode(label="Field", name="Computing", embedding=[0.0, 1.0, 0.0]),
            ]
        )
        await adapter.update_relations(
            FreeEdge(
                label="PIONEER_OF",
                subj=FreeNode(label="Person", name="Ada", embedding=[1.0, 0.0, 0.0]),
                obj=FreeNode(label="Field", name="Computing", embedding=[0.0, 1.0, 0.0]),
            )
        )
        subgraph = await adapter.local_graph_search(
            "Ada", label="Person", max_hops=2, k=1
        )
        names = sorted((e.label, e.name) for e in subgraph.entities)
        self.assertEqual(names, [("Field", "Computing"), ("Person", "Ada")])

    async def test_missing_primary_key_still_raises(self):
        """A free-form entity whose schema yields no usable PK fails
        loudly rather than creating a broken table."""
        adapter = self._adapter()
        # Base Entity has only `label` — no property to promote to PK.
        with self.assertRaisesRegex(ValueError, "primary key"):
            await adapter.update_entities(Entity(label="Bare"))
