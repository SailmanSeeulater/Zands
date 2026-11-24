import json

from zands import store, ui


def test_build_command_loads_sqlite_vec(tmp_path):
    db = tmp_path / "news.db"
    cmd = ui.build_command(db, port=9001, metadata_path=tmp_path / "m.json")
    assert cmd[1:4] == ["-m", "datasette", "serve"]
    assert str(db.resolve()) in cmd
    assert cmd[cmd.index("--port") + 1] == "9001"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--load-extension") + 1] == store.vec_extension_path()
    assert cmd[cmd.index("--metadata") + 1] == str(tmp_path / "m.json")


def test_metadata_named_after_database(tmp_path):
    db = tmp_path / "other.db"
    store.connect(db).close()
    meta = json.loads(ui.write_metadata(db).read_text())
    queries = meta["databases"]["other"]["queries"]
    assert {"recent_retrievals", "feed_health", "current_facts"} <= set(queries)
    conn = store.connect(db)
    for q in queries.values():  # every canned query runs against the real schema
        conn.execute(q["sql"]).fetchall()
