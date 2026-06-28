from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import Settings, get_settings


def sqlalchemy_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


@lru_cache
def get_engine(database_url: str | None = None) -> Engine:
    settings = get_settings()
    resolved_url = database_url or settings.database_url
    return create_engine(sqlalchemy_database_url(resolved_url), pool_pre_ping=True)


def get_session_factory(settings: Settings) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(settings.database_url), expire_on_commit=False)


@contextmanager
def session_scope(settings: Settings) -> Iterator[Session]:
    session_factory = get_session_factory(settings)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
