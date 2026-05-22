# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import orjson

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import GenericResult
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.modules.module import Module


@synalinks_export(
    [
        "synalinks.modules.RRFReranker",
        "synalinks.RRFReranker",
    ]
)
class RRFReranker(Module):
    """Merge multiple search results with Reciprocal Rank Fusion (RRF).

    Takes a list of result data models — each a `GenericResult` whose
    ``result`` field is a ranked list of rows — and fuses their rankings
    into a single ranked list. A row's fused score is

        ``score(row) = sum over lists of 1 / (k_rank + rank)``

    where ``rank`` is the row's 1-based position in each list. RRF needs
    only the *ordering* of each list, so it merges heterogeneous result
    sets (similarity, full-text, regex, graph) without having to
    normalize their incompatible score scales.

    Rows are matched across lists by ``id_key`` when given, otherwise by
    a canonical signature of the whole row. The fused ``rrf_score`` is
    written onto each returned row; the output is a `GenericResult`
    sorted by descending score and truncated to ``k``. ``None`` inputs
    are ignored, so it composes with optional retrieval branches.

    Example:

    ```python
    import synalinks
    import asyncio

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(description="The user question")

    async def main():
        kb = synalinks.KnowledgeBase(uri="duckdb://docs.db", data_models=[Document])
        lm = synalinks.LanguageModel(model="ollama/mistral")

        inputs = synalinks.Input(data_model=Query)
        vector_hits = await synalinks.SimilaritySearch(
            knowledge_base=kb, language_model=lm, data_model=Document,
        )(inputs)
        keyword_hits = await synalinks.FullTextSearch(
            knowledge_base=kb, language_model=lm, data_model=Document,
        )(inputs)
        fused = await synalinks.RRFReranker(k=10, id_key="id")(
            [vector_hits, keyword_hits]
        )
        program = synalinks.Program(inputs=inputs, outputs=fused)

    asyncio.run(main())
    ```

    Args:
        k_rank (int): RRF smoothing constant. Lower values weight
            top-ranked rows more strongly. Defaults to 60.
        k (int): Maximum number of fused rows to return. ``None`` returns
            all fused rows. Defaults to None.
        id_key (str): Row field used to identify the same row across
            lists. When ``None`` (default), the whole row is used as its
            identity (a canonical JSON signature).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    def __init__(
        self,
        *,
        k_rank: int = 60,
        k: Optional[int] = None,
        id_key: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        trainable: bool = False,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not isinstance(k_rank, int) or k_rank < 1:
            raise ValueError(f"`k_rank` must be a positive integer, got {k_rank!r}")
        self.k_rank = k_rank
        if k is not None and (not isinstance(k, int) or k < 1):
            raise ValueError(f"`k` must be a positive integer or None, got {k!r}")
        self.k = k
        self.id_key = id_key

    def _row_id(self, row):
        """Identity of a row for cross-list matching."""
        if self.id_key is not None and isinstance(row, dict) and self.id_key in row:
            return row[self.id_key]
        return orjson.dumps(row, option=orjson.OPT_SORT_KEYS, default=str)

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        result_lists: List[List[Any]] = []
        for result in inputs:
            if result is None:
                continue
            rows = result.get("result")
            if rows:
                result_lists.append(rows)
        if not result_lists:
            return None

        scores: Dict[Any, float] = {}
        merged: Dict[Any, Any] = {}
        for rows in result_lists:
            for rank, row in enumerate(rows, start=1):
                uid = self._row_id(row)
                scores[uid] = scores.get(uid, 0.0) + 1.0 / (self.k_rank + rank)
                if uid not in merged:
                    merged[uid] = dict(row) if isinstance(row, dict) else row
                elif isinstance(row, dict) and isinstance(merged[uid], dict):
                    merged[uid].update(row)

        for uid, row in merged.items():
            if isinstance(row, dict):
                row["rrf_score"] = scores[uid]

        order = sorted(scores, key=lambda uid: scores[uid], reverse=True)
        if self.k is not None:
            order = order[: self.k]
        fused = [merged[uid] for uid in order]

        return JsonDataModel(
            json={"result": fused},
            schema=GenericResult.get_schema(),
            name=self.name,
        )

    async def compute_output_spec(self, inputs, training=False):
        return SymbolicDataModel(
            schema=GenericResult.get_schema(),
            name=self.name,
        )

    def get_config(self):
        return {
            "k_rank": self.k_rank,
            "k": self.k,
            "id_key": self.id_key,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
