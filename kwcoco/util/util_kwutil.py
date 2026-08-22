"""
This is a staging ground for utilities that may make there way into
:mod:`kwutil` proper at some point in the future.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any, Generic, ParamSpec, TypeVar

import ubelt as ub


P = ParamSpec('P')
T = TypeVar('T')


class _DelayedFuture(Generic[T]):
    """
    todo: move to kwutil

    Wraps a future object so we can execute logic when its result has been
    accessed.
    """

    def __init__(
        self,
        func: Callable[..., T],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        parent: _DelayedBlockingJobQueue,
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.task = (func, args, kwargs)
        self.parent = parent
        self.future = None

    def result(self, timeout: float | None = None) -> T:
        if self.future is None:
            raise Exception('The task has not been submitted yet')
        result = self.future.result(timeout)
        self.parent._job_result_accessed_callback(self)
        return result


class _DelayedBlockingJobQueue:
    """
    todo: move to kwutil

    References:
        .. [GISTnoxdafoxMaxQueuePool] https://gist.github.com/noxdafox/4150eff0059ea43f6adbdd66e5d5e87e

    Ignore:
        >>> self = _DelayedBlockingJobQueue(max_unhandled_jobs=5)
        >>> futures = [
        >>>     self.submit(print, i)
        >>>     for i in range(10)
        >>> ][::-1]
        >>> import time
        >>> time.sleep(0.5)
        >>> print(self._num_submitted_jobs)
        >>> print(self._num_handled_results)
        >>> print('--- First 5 should have printed ---')
        >>> for _ in range(3):
        >>>     f = futures.pop()
        >>>     f.result()
        >>> time.sleep(0.5)
        >>> print(self._num_submitted_jobs)
        >>> print(self._num_handled_results)
        >>> print('--- 3 Results were haneld, so 3 more can join the queue')
        >>> for _ in range(3):
        >>>     f = futures.pop()
        >>>     f.result()
        >>> time.sleep(0.5)
        >>> print(self._num_submitted_jobs)
        >>> print(self._num_handled_results)
        >>> print('--- Handling the rest, but everything should have already been submitted')
        >>> for _ in range(4):
        >>>     f = futures.pop()
        >>>     f.result()
    """

    def __init__(
        self,
        max_unhandled_jobs: int,
        mode: str = 'thread',
        max_workers: int | None = None,
    ) -> None:
        self._unsubmitted: deque[_DelayedFuture[Any]] = deque()
        self.pool = ub.Executor(mode=mode, max_workers=max_workers)  # type: ignore
        self.max_unhandled_jobs = max_unhandled_jobs
        self._num_handled_results = 0
        self._num_submitted_jobs = 0
        self._num_unhandled = 0

    def submit(
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> _DelayedFuture[T]:
        """
        Queues a new job, but wont execute until
        some conditions are met
        """
        delayed = _DelayedFuture(func, args, kwargs, parent=self)
        self._unsubmitted.append(delayed)
        self._submit_if_room()
        return delayed

    def _submit_if_room(self) -> None:
        while self._num_unhandled < self.max_unhandled_jobs and self._unsubmitted:
            delayed = self._unsubmitted.popleft()
            self._num_submitted_jobs += 1
            self._num_unhandled += 1
            delayed.future = self.pool.submit(
                delayed.func, *delayed.args, **delayed.kwargs
            )

    def _job_result_accessed_callback(self, _: _DelayedFuture[Any]) -> None:
        """Called when the user handles a result"""
        self._num_handled_results += 1
        self._num_unhandled -= 1
        self._submit_if_room()

    def shutdown(self) -> Any:
        """
        Calls the shutdown function of the underlying backend.
        """
        return self.pool.shutdown()


class _MaxQueuePool:
    """

    todo: move to kwutil

    This Class wraps a concurrent.futures.Executor
    limiting the size of its task queue.
    If `max_queue_size` tasks are submitted, the next call to submit will block
    until a previously submitted one is completed.

    References:
        .. [GISTnoxdafoxMaxQueuePool] https://gist.github.com/noxdafox/4150eff0059ea43f6adbdd66e5d5e87e

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/geowatch'))
        from geowatch.tasks.fusion.evaluate import *  # NOQA
        from geowatch.tasks.fusion.evaluate import _memo_legend, _redraw_measures
        self = _MaxQueuePool(max_queue_size=0)

        dpath = ub.Path.appdir('kwutil/doctests/maxpoolqueue')
        dpath.delete().ensuredir()
        signal_fpath = dpath / 'signal'

        def waiting_worker():
            counter = 0
            while not signal_fpath.exists():
                counter += 1
            return counter

        future = self.submit(waiting_worker)

        try:
            future.result(timeout=0.001)
        except TimeoutError:
            ...
        signal_fpath.touch()
        result = future.result()

    """

    def __init__(
        self,
        max_queue_size: int | None = None,
        mode: str = 'thread',
        max_workers: int = 0,
    ) -> None:
        if max_queue_size is None:
            max_queue_size = max_workers
        self.pool = ub.Executor(mode=mode, max_workers=max_workers)  # type: ignore
        if 'serial' in self.pool.backend.__class__.__name__.lower():
            self.pool_queue = None
        else:
            from threading import BoundedSemaphore  # NOQA

            self.pool_queue = BoundedSemaphore(max_queue_size)

    def submit(
        self,
        function: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Any:
        """Submits a new task to the pool, blocks if Pool queue is full."""
        if self.pool_queue is not None:
            self.pool_queue.acquire()

        future = self.pool.submit(function, *args, **kwargs)
        future.add_done_callback(self.pool_queue_callback)

        return future

    def pool_queue_callback(self, _: Any) -> None:
        """Called once task is done, releases one queue slot."""
        if self.pool_queue is not None:
            self.pool_queue.release()

    def shutdown(self) -> Any:
        """
        Calls the shutdown function of the underlying backend.
        """
        return self.pool.shutdown()
