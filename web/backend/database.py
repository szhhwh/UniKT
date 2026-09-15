"""Database engine, session, and declarative base setup.

Creates the SQLAlchemy engine bound to the SQLite database at DATABASE_PATH,
configures WAL mode and NORMAL synchronous pragma on connection, and provides
the Base declarative model class along with an init_db helper.
"""

import sqlite3

from config import DATABASE_PATH, PREPROCESS_LOGS_DIR, TASK_LOGS_DIR
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import ConnectionPoolEntry


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


# check_same_thread=False: the default QueuePool reuses connections across the
# HTTP threadpool and the scheduler/reader/recover worker threads, which the
# sqlite3 default (check_same_thread=True) rejects. WAL + busy_timeout make
# concurrent writes wait rather than fail.
engine = create_engine(
    f"sqlite:///{DATABASE_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine)


def _on_connect(
    dbapi_connection: sqlite3.Connection, _connection_record: ConnectionPoolEntry
) -> None:
    """Configure WAL, NORMAL synchronous, and a busy timeout on connection."""
    dbapi_connection.execute("PRAGMA journal_mode=WAL")
    dbapi_connection.execute("PRAGMA synchronous=NORMAL")
    dbapi_connection.execute("PRAGMA busy_timeout=5000")


event.listen(engine, "connect", _on_connect)


def init_db() -> None:
    """Create the database directory, log directories, and tables if absent."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TASK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _drop_legacy_columns()


def _drop_legacy_columns() -> None:
    """Drop columns removed from the ORM but still present in old databases.

    create_all only creates missing tables, so a dropped mapping leaves a
    stale NOT NULL column behind that breaks every INSERT. Idempotent; needs
    SQLite >= 3.35 (older runtimes keep the column and fail loudly on insert).
    """
    if sqlite3.sqlite_version_info < (3, 35, 0):
        return
    with engine.begin() as conn:
        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(tasks)")}
        if "exp_dir" in columns:
            conn.exec_driver_sql("ALTER TABLE tasks DROP COLUMN exp_dir")
