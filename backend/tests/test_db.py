from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from app import db as db_module
from app.db import Database
from sqlalchemy import create_engine, text


@pytest.mark.parametrize("database_url", ["sqlite:///:memory:", "sqlite:///{path}"])
def test_database_enables_foreign_keys_on_sqlite_connections(
    database_url, tmp_path
):
    database = Database(database_url.format(path=tmp_path / "foreign-keys.db"))

    with database.engine.connect() as first_connection:
        assert first_connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        if database_url != "sqlite:///:memory:":
            with database.engine.connect() as second_connection:
                assert second_connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1

    database.dispose()


def test_database_does_not_register_sqlite_listener_for_other_dialects(monkeypatch):
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    create_engine = MagicMock(return_value=engine)
    listen = MagicMock()
    monkeypatch.setattr(db_module, "create_engine", create_engine)
    monkeypatch.setattr(db_module.event, "listen", listen)

    Database("postgresql://manager.example/database")

    listen.assert_not_called()


def test_database_listener_does_not_affect_other_sqlite_engines():
    database = Database("sqlite:///:memory:")
    unrelated_engine = create_engine("sqlite:///:memory:")

    with database.engine.connect() as managed_connection:
        assert managed_connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    with unrelated_engine.connect() as unrelated_connection:
        assert unrelated_connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 0

    unrelated_engine.dispose()
    database.dispose()


def test_sqlite_listener_rejects_connections_without_foreign_key_support():
    cursor = MagicMock()
    cursor.fetchone.return_value = (0,)
    connection = MagicMock()
    connection.cursor.return_value = cursor

    with pytest.raises(RuntimeError, match="foreign key enforcement"):
        db_module._enable_sqlite_foreign_keys(connection, None)

    assert cursor.execute.call_args_list == [
        call("PRAGMA foreign_keys=ON"),
        call("PRAGMA foreign_keys"),
    ]
    cursor.close.assert_called_once_with()
