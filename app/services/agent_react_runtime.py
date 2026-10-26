"""Отдельный процесс для прерываемого выполнения LLM и tools."""

import multiprocessing as mp
import queue
import threading
import time
import faulthandler
from app.services.agent_naive_tools import get_rag, make_client
from app.services.agent_react_tools import DISPATCH

import importlib

def _serve(connection):
    client = None
    try:
        while True:
            request = connection.recv()
            kind = request["kind"]
            payload = request["payload"]

            if kind == "stop":
                connection.send({"ok": True, "value": None})
                break

            try:
                if kind == "prepare":
                    print("[startup] Началась подготовка", flush=True)
                    faulthandler.dump_traceback_later(
                        20,
                        repeat=True,
                    )
                    try:
                        print(
                            "[startup] Загружаются библиотеки поиска",
                            flush=True,
                        )
                        started = time.perf_counter()
                        importlib.import_module("app.services.rag")
                        imports_ms = round(
                            (time.perf_counter() - started) * 1000, 2
                        )
                        print(
                            f"[startup] Библиотеки загружены: {imports_ms} мс",
                            flush=True,
                        )

                        print(
                            "[startup] Создаётся клиент OpenAI",
                            flush=True,
                        )
                        started = time.perf_counter()
                        if client is None:
                            client = make_client()
                        client_init_ms = round(
                            (time.perf_counter() - started) * 1000, 2
                        )
                        print(
                            f"[startup] Клиент создан: {client_init_ms} мс",
                            flush=True,
                        )

                        print(
                         "[startup] Подключается существующий индекс",
                         flush=True,
                        )
                        started = time.perf_counter()
                        get_rag()
                        rag_init_ms = round(
                            (time.perf_counter() - started) * 1000, 2
                        )
                        print(
                            f"[startup] Индекс подключён: {rag_init_ms} мс",
                            flush=True,
                        )

                        value = {
                            "imports_ms": imports_ms,
                            "client_init_ms": client_init_ms,
                            "rag_init_ms": rag_init_ms,
                        }
                    finally:
                        faulthandler.cancel_dump_traceback_later()
                elif kind == "llm":
                    if client is None:
                        client = make_client()

                    response = client.chat.completions.create(
                        **payload,
                        timeout=request["timeout"],
                    )
                    value = response.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                elif kind == "tool":
                    value = str(
                        DISPATCH[payload["name"]](**payload["arguments"])
                    )
                else:
                    raise ValueError("Неизвестная операция")

                result = {"ok": True, "value": value}
            except Exception as exc:
                result = {"ok": False, "error": type(exc).__name__}

            connection.send(result)
    except (EOFError, OSError):
        pass
    finally:
        try:
            if client is not None:
                client.close()
            if get_rag.cache_info().currsize:
                get_rag().close()
        finally:
            connection.close()


class Worker:
    def __init__(self, target=None, startup_timeout_sec=60):
        if (
            type(startup_timeout_sec) not in (int, float)
            or not 1 <= startup_timeout_sec <= 60
        ):
            raise ValueError("startup_timeout_sec должен быть от 1 до 60")

        self.target = target or _serve
        self.broken = False
        self.startup_timeout_sec = startup_timeout_sec
        self.startup = {
            "status": "not_started",
            "timeout_sec": startup_timeout_sec,
            "latency_ms": None,
        }

    def __enter__(self):
        started = time.perf_counter()
        self.startup["status"] = "preparing"

        context = mp.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(
            target=self.target,
            args=(child_connection,),
            daemon=True,
        )
        try:
            self.process.start()
        except BaseException as exc:
            self.connection.close()
            self.startup.update(
                status="error",
                error=type(exc).__name__,
                latency_ms=round(
                    (time.perf_counter() - started) * 1000, 2
                ),
            )
            raise
        finally:
            child_connection.close()

        try:
            response = self.call(
                "prepare",
                {},
                started + self.startup_timeout_sec,
            )
            if not response["ok"]:
                self.startup["error"] = response["error"]
                raise RuntimeError("Ошибка подготовки рабочего процесса")

            self.startup.update(
                status="ready",
                details=response["value"],
            )
        except BaseException as exc:
            self.startup["status"] = (
                "timeout" if isinstance(exc, TimeoutError) else "error"
            )
            self.startup.setdefault("error", type(exc).__name__)
            self.__exit__(None, None, None)
            raise
        finally:
            self.startup["latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 2
            )

        return self

    def call(self, kind, payload, deadline):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            self.broken = True
            raise TimeoutError

        replies = queue.Queue()

        def exchange():
            try:
                self.connection.send(
                    {
                        "kind": kind,
                        "payload": payload,
                        "timeout": remaining,
                    }
                )
                replies.put(self.connection.recv())
            except Exception:
                replies.put(
                    {"ok": False, "error": "WorkerConnectionError"}
                )

        # Ограничиваем также ожидание передачи данных между процессами.
        threading.Thread(target=exchange, daemon=True).start()

        try:
            return replies.get(
                timeout=max(0, deadline - time.perf_counter())
            )
        except queue.Empty:
            self.broken = True
            raise TimeoutError from None

    def __exit__(self, exc_type, exc, traceback):
        try:
            if not self.broken and self.process.is_alive():
                try:
                    self.call("stop", {}, time.perf_counter() + 0.2)
                except Exception:
                    pass

            self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=0.2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=0.2)
        finally:
            self.connection.close()
            if not self.process.is_alive():
                self.process.close()