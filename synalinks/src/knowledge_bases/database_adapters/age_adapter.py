from __future__ import annotations
import psycopg
from psycopg_pool import ConnectionPool


class AGEAdapter:
    """
    Production-ready Apache AGE adapter.
    Works with psycopg3 + psycopg-pool >=3.2.
    """

    def __init__(self, uri: str):
        self.graph = self._extract_graph_name(uri)
        self.conninfo = self._build_conninfo(uri)
        self.pool = ConnectionPool(conninfo=self.conninfo, max_size=4)
        self._init_extension()

    # ---------- helpers ----------

    def _extract_graph_name(self, uri: str) -> str:
        import urllib.parse as up
        parsed = up.urlparse(uri)
        query = up.parse_qs(parsed.query)
        return query.get("graph", ["graph"])[0]

    def _build_conninfo(self, uri: str) -> str:
        import urllib.parse as up
        parsed = up.urlparse(uri)
        return (
            f"host={parsed.hostname or '127.0.0.1'} "
            f"port={parsed.port or 5455} "
            f"user={parsed.username or 'age'} "
            f"password={parsed.password or 'age'} "
            f"dbname={parsed.path.strip('/') or 'age'}"
        )

    # ---------- initialization ----------

    def _init_extension(self):
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("CREATE EXTENSION IF NOT EXISTS age;")
            cur.execute("SET search_path TO ag_catalog, \"$user\", public;")
            cur.execute("SELECT name FROM ag_catalog.ag_graph WHERE name = %s;", (self.graph,))
            if cur.fetchone() is None:
                cur.execute(f"SELECT ag_catalog.create_graph('{self.graph}');")
            conn.commit()

    # ---------- core execution ----------

    def execute(self, query: str):
        """
        Execute Cypher query and return results.
        Handles both write (CREATE/DELETE) and read (MATCH) queries.
        """
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SET search_path TO ag_catalog, \"$user\", public;")
            sql = f"SELECT * FROM cypher('{self.graph}', $$ {query} $$) AS (v agtype);"
            cur.execute(sql)
            rows = cur.fetchall()
            return [self._decode_agtype(r[0]) for r in rows]

    # ---------- utilities ----------

    def close(self):
        """Close the pool cleanly, compatible across psycopg versions."""
        try:
            self.pool.close()
            # psycopg-pool >=3.2 uses wait(), older used wait_close()
            if hasattr(self.pool, "wait"):
                self.pool.wait()
            elif hasattr(self.pool, "wait_close"):
                self.pool.wait_close()
        except Exception:
            pass

    @staticmethod
    def _decode_agtype(value):
        if value is None:
            return None
        try:
            import json
            return json.loads(value)
        except Exception:
            return str(value)
