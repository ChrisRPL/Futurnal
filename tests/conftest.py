"""Shared test configuration and dependency stubs."""

from __future__ import annotations

import sys
import types


def _ensure_unstructured_stub() -> None:
    if "unstructured.partition.auto" in sys.modules:
        return

    unstructured_module = types.ModuleType("unstructured")
    partition_module = types.ModuleType("unstructured.partition")
    auto_module = types.ModuleType("unstructured.partition.auto")

    def _placeholder_partition(**_kwargs):  # pragma: no cover - should be monkeypatched per test
        raise RuntimeError("Stub partition invoked; tests must monkeypatch with fixture-specific behaviour")

    auto_module.partition = _placeholder_partition
    partition_module.auto = auto_module
    unstructured_module.partition = partition_module

    sys.modules.setdefault("unstructured", unstructured_module)
    sys.modules.setdefault("unstructured.partition", partition_module)
    sys.modules.setdefault("unstructured.partition.auto", auto_module)


_ensure_unstructured_stub()


def _ensure_neo4j_stub() -> None:
    if "neo4j" in sys.modules:
        return

    neo4j_module = types.ModuleType("neo4j")

    class _FakeDriver:
        def __init__(self) -> None:  # pragma: no cover - simple stub
            pass

        def session(self, *args, **kwargs):  # pragma: no cover - stub context manager
            return self

        def execute_write(self, func):  # pragma: no cover - stub
            func(lambda *a, **kw: None)

        def close(self):  # pragma: no cover - stub
            pass

    def _driver(*args, **kwargs):  # pragma: no cover - stub
        return _FakeDriver()

    neo4j_module.GraphDatabase = types.SimpleNamespace(driver=_driver)
    neo4j_module.Driver = _FakeDriver

    sys.modules.setdefault("neo4j", neo4j_module)


_ensure_neo4j_stub()


def _ensure_apscheduler_stub() -> None:
    if "apscheduler.schedulers.asyncio" in sys.modules:
        return

    apscheduler_module = types.ModuleType("apscheduler")
    schedulers_module = types.ModuleType("apscheduler.schedulers")
    asyncio_module = types.ModuleType("apscheduler.schedulers.asyncio")

    class _AsyncIOScheduler:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_job(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def shutdown(self, *args, **kwargs) -> None:
            pass

    asyncio_module.AsyncIOScheduler = _AsyncIOScheduler
    schedulers_module.asyncio = asyncio_module
    apscheduler_module.schedulers = schedulers_module

    triggers_module = types.ModuleType("apscheduler.triggers")
    cron_module = types.ModuleType("apscheduler.triggers.cron")
    interval_module = types.ModuleType("apscheduler.triggers.interval")

    class _CronTrigger:  # pragma: no cover - stub
        @staticmethod
        def from_crontab(*args, **kwargs):
            return _CronTrigger()

    class _IntervalTrigger:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs) -> None:
            pass

    cron_module.CronTrigger = _CronTrigger
    interval_module.IntervalTrigger = _IntervalTrigger
    triggers_module.cron = cron_module
    triggers_module.interval = interval_module

    sys.modules.setdefault("apscheduler", apscheduler_module)
    sys.modules.setdefault("apscheduler.schedulers", schedulers_module)
    sys.modules.setdefault("apscheduler.schedulers.asyncio", asyncio_module)
    sys.modules.setdefault("apscheduler.triggers", triggers_module)
    sys.modules.setdefault("apscheduler.triggers.cron", cron_module)
    sys.modules.setdefault("apscheduler.triggers.interval", interval_module)


_ensure_apscheduler_stub()


def _ensure_chromadb_stub() -> None:
    if "chromadb" in sys.modules:
        return

    chromadb_module = types.ModuleType("chromadb")
    api_module = types.ModuleType("chromadb.api")
    models_module = types.ModuleType("chromadb.api.models")
    collection_module = types.ModuleType("chromadb.api.models.Collection")

    class Collection:  # pragma: no cover - stub
        def __init__(self) -> None:
            self.storage = {}

        def upsert(self, **kwargs) -> None:
            ids = kwargs.get("ids", [])
            documents = kwargs.get("documents", [])
            metadatas = kwargs.get("metadatas", [])
            embeddings = kwargs.get("embeddings", [])
            for index, identifier in enumerate(ids):
                self.storage[identifier] = {
                    "document": documents[index] if index < len(documents) else None,
                    "metadata": metadatas[index] if index < len(metadatas) else None,
                    "embedding": embeddings[index] if index < len(embeddings) else None,
                }

        def delete(self, ids=None):  # pragma: no cover - stub
            for identifier in ids or []:
                self.storage.pop(identifier, None)

    collection_module.Collection = Collection
    models_module.Collection = Collection

    class ClientAPI:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs) -> None:
            self._collections: dict[str, Collection] = {}

        def get_or_create_collection(self, name: str, metadata=None, embedding_function=None):
            self._collections.setdefault(name, Collection())
            return self._collections[name]

        def close(self) -> None:
            self._collections.clear()

    class PersistentClient(ClientAPI):  # pragma: no cover - stub
        pass

    api_module.ClientAPI = ClientAPI
    models_module.Collection = Collection

    class SentenceTransformerEmbeddingFunction:  # pragma: no cover - stub
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def __call__(self, texts):
            return [[0.0, 0.0, 0.0] for _ in texts]

    embedding_functions_module = types.ModuleType("chromadb.utils.embedding_functions")
    embedding_functions_module.SentenceTransformerEmbeddingFunction = SentenceTransformerEmbeddingFunction

    utils_module = types.ModuleType("chromadb.utils")
    utils_module.embedding_functions = embedding_functions_module

    chromadb_module.PersistentClient = PersistentClient
    chromadb_module.api = api_module
    chromadb_module.utils = utils_module

    sys.modules.setdefault("chromadb", chromadb_module)
    sys.modules.setdefault("chromadb.api", api_module)
    sys.modules.setdefault("chromadb.api.models", models_module)
    sys.modules.setdefault("chromadb.api.models.Collection", collection_module)
    sys.modules.setdefault("chromadb.api.models.Collection.Collection", collection_module)
    sys.modules.setdefault("chromadb.utils", utils_module)
    sys.modules.setdefault("chromadb.utils.embedding_functions", embedding_functions_module)


_ensure_chromadb_stub()


def _ensure_croniter_stub() -> None:
    if "croniter" in sys.modules:
        return

    class _Croniter:  # pragma: no cover - stub
        def __init__(self, expression: str, start_time=None):  # noqa: D401
            self.expression = expression
            self.start_time = start_time

        def get_next(self, *args, **kwargs):
            return 0

    def _is_valid(expression: str) -> bool:
        return bool(expression)

    croniter_module = types.ModuleType("croniter")
    croniter_module.croniter = _Croniter
    croniter_module.is_valid = _is_valid

    sys.modules.setdefault("croniter", croniter_module)


_ensure_croniter_stub()


def _ensure_watchdog_stub() -> None:
    if "watchdog.events" in sys.modules:
        return

    watchdog_module = types.ModuleType("watchdog")
    events_module = types.ModuleType("watchdog.events")

    class FileSystemEventHandler:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs) -> None:
            pass

    events_module.FileSystemEventHandler = FileSystemEventHandler

    observers_module = types.ModuleType("watchdog.observers")

    class Observer:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs) -> None:
            pass

        def schedule(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            pass

    observers_module.Observer = Observer

    watchdog_module.events = events_module
    watchdog_module.observers = observers_module

    sys.modules.setdefault("watchdog", watchdog_module)
    sys.modules.setdefault("watchdog.events", events_module)
    sys.modules.setdefault("watchdog.observers", observers_module)


_ensure_watchdog_stub()


