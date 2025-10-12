from synalinks.src.knowledge_bases.database_adapters.age_adapter import AGEAdapter

def test_age_adapter_basic():
    db = AGEAdapter("age://age:age@localhost:5455/age?graph=test_graph")
    db.execute("CREATE (:Person {name:'TestA'})-[:KNOWS]->(:Person {name:'TestB'})")
    result = db.execute("MATCH (n:Person) RETURN n")
    assert any("TestA" in r or "TestB" in r for r in result)
    db.close()
