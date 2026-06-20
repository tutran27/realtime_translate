import asyncio
import concurrent.futures
from dataclasses import dataclass
import logging
import time

from .translator import MachineTranslator


@dataclass
class TranslationResult:
    text: str
    queue_wait_ms: float
    inference_ms: float
    batch_size: int


@dataclass
class TranslationRequest:
    text: str
    source_language: str
    target_language: str
    queued_at: float
    future: asyncio.Future


class TranslationService:
    MAX_BATCH_SIZE = 4

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.translator = None
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="envit5-translation",
        )
        self.queue: asyncio.Queue[TranslationRequest | None] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None

    def initialize_sync(self) -> None:
        if self.translator is not None:
            return
        started_at = time.perf_counter()
        self.translator = MachineTranslator(
            model_name=self.model_name,
            device=self.device,
        )
        self.translator.translate(
            "Hello.",
            src_lang="eng_Latn",
            tgt_lang="vie_Latn",
        )
        logging.info(
            "Translation model ready in %.2fs",
            time.perf_counter() - started_at,
        )

    def translate_batch_sync(
        self,
        requests: list[TranslationRequest],
    ) -> list[str]:
        self.initialize_sync()
        return self.translator.translate_batch(
            [
                (
                    request.text,
                    request.source_language,
                    request.target_language,
                )
                for request in requests
            ]
        )

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self.initialize_sync)
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(
                self._run_worker(),
                name="envit5-translation-worker",
            )

    async def _collect_batch(
        self,
        first: TranslationRequest,
    ) -> list[TranslationRequest]:
        batch = [first]
        while len(batch) < self.MAX_BATCH_SIZE:
            try:
                request = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if request is None:
                await self.queue.put(None)
                break
            if not request.future.cancelled():
                batch.append(request)
        return batch

    async def _run_worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            if first is None:
                return
            if first.future.cancelled():
                continue

            batch = await self._collect_batch(first)
            active = [
                request for request in batch if not request.future.cancelled()
            ]
            if not active:
                continue

            inference_started_at = time.perf_counter()
            try:
                texts = await loop.run_in_executor(
                    self.executor,
                    self.translate_batch_sync,
                    active,
                )
            except Exception as exc:
                for request in active:
                    if not request.future.done():
                        request.future.set_exception(exc)
                continue

            inference_ms = (
                time.perf_counter() - inference_started_at
            ) * 1000
            for request, text in zip(active, texts):
                if request.future.done():
                    continue
                request.future.set_result(
                    TranslationResult(
                        text=text,
                        queue_wait_ms=(
                            inference_started_at - request.queued_at
                        )
                        * 1000,
                        inference_ms=inference_ms,
                        batch_size=len(active),
                    )
                )
            logging.info(
                "Translation batch=%s inference=%.0fms queue_depth=%s",
                len(active),
                inference_ms,
                self.queue.qsize(),
            )

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        if self.worker_task is None:
            await self.start()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(
            TranslationRequest(
                text=text,
                source_language=source_language,
                target_language=target_language,
                queued_at=time.perf_counter(),
                future=future,
            )
        )
        return await future

    async def stop(self) -> None:
        if self.worker_task is not None:
            await self.queue.put(None)
            await self.worker_task
            self.worker_task = None
        self.executor.shutdown(wait=True, cancel_futures=True)
