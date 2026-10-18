"""Bounded CPU preparation and batched submission for independent requests.

Reorder only within each bounded admission window; never retain an old request
to wait for a matching schema. Backend requests overlap the next CPU window.
"""
import asyncio
import time


class SubmissionPipeline:
    def __init__(self, client, prepare_many, format_result, schema_metadata, model,
                 mode='batch', window=64, batch_size=16, token_budget=32768, delay=.002):
        self.client, self.prepare, self.format = client, prepare_many, format_result
        self.schema, self.model, self.mode = schema_metadata, model, mode
        self.window, self.batch_size, self.token_budget, self.delay = window, batch_size, token_budget, delay
        self.queue = asyncio.Queue(maxsize=4096)
        self.slots = asyncio.Semaphore(16)
        self.tasks = set()
        self.worker = asyncio.create_task(self.dispatch())

    async def close(self):
        self.worker.cancel()
        await asyncio.gather(self.worker, return_exceptions=True)
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def submit(self, request):
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((request, future, time.perf_counter()))
        return await future

    async def dispatch(self):
        while True:
            first = await self.queue.get()
            pending = [first]
            deadline = first[2]+self.delay
            while len(pending) < self.window:
                try:
                    pending.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    remaining = deadline-time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        pending.append(await asyncio.wait_for(self.queue.get(), remaining))
                    except asyncio.TimeoutError:
                        break
            try:
                items = await asyncio.to_thread(self.prepare, [p[0] for p in pending])
                prepared = time.perf_counter()
                ordered = list(zip(pending, items, strict=True))
                if self.mode == 'schema':
                    ordered.sort(key=lambda pair: (
                        tuple(len(q.option_ids) for q in pair[1].questions),
                        len(pair[1].input_ids).bit_length(), len(pair[1].input_ids)))
                groups, group, tokens = [], [], 0
                for entry in ordered:
                    count = len(entry[1].input_ids)
                    if group and (len(group) >= self.batch_size or tokens+count > self.token_budget):
                        groups.append(group)
                        group, tokens = [], 0
                    group.append(entry)
                    tokens += count
                if group:
                    groups.append(group)
                for group in groups:
                    await self.slots.acquire()
                    task = asyncio.create_task(self.send(group, prepared))
                    self.tasks.add(task)
                    task.add_done_callback(self.tasks.discard)
            except Exception as exc:
                for _, future, _ in pending:
                    if not future.done():
                        future.set_exception(exc)

    async def send(self, group, prepared):
        try:
            response = await self.client.post('/pooling', json=dict(model=self.model,
                input=[list(item.input_ids) for _, item in group], task='classify', encoding_format='float',
                clef_records=[self.schema(item) for _, item in group]))
            response.raise_for_status()
            rows = sorted(response.json()['data'], key=lambda r: r['index'])
            if [r['index'] for r in rows] != list(range(len(group))):
                raise ValueError('Incomplete batch response')
            scored = time.perf_counter()
            for ((request, future, started), item), row in zip(group, rows, strict=True):
                result = self.format(request, item, row['data'])
                timing = (f'prepare;dur={(prepared-started)*1000:.3f},'
                          f'vllm;dur={(scored-prepared)*1000:.3f},'
                          f'format;dur={(time.perf_counter()-scored)*1000:.3f}')
                if not future.done():
                    future.set_result((result, timing))
        except Exception as exc:
            for (_, future, _), _ in group:
                if not future.done():
                    future.set_exception(exc)
        finally:
            self.slots.release()
