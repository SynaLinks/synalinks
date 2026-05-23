#    Copyright 2013-2018 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Async-related utilities
"""

import abc
import asyncio
import contextlib
import contextvars
import functools
import inspect
import itertools
import os.path
import pathlib
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from weakref import WeakSet

from greenlet import greenlet


def run_maybe_nested(coro):
    return run(allow_nested_run(coro))


def create_task(awaitable, name=None):
    if isinstance(awaitable, asyncio.Task):
        task = awaitable
    else:
        task = asyncio.create_task(awaitable)
    if name is None:
        name = getattr(awaitable, "__qualname__", None)
    task.name = name
    return task


def _close_loop(loop):
    if loop is not None:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            try:
                shutdown_default_executor = loop.shutdown_default_executor
            except AttributeError:
                pass
            else:
                loop.run_until_complete(shutdown_default_executor())
        finally:
            loop.close()


class AsyncManager:
    def __init__(self):
        self.task_tree = dict()
        self.resources = dict()

    def track_access(self, access):
        """
        Register the given ``access`` to have been handled by the current
        async task.

        :param access: Access that were done.
        :type access: ConcurrentAccessBase

        This allows `concurrently` to check that concurrent tasks did not
        step on each other's toes.
        """
        try:
            task = asyncio.current_task()
        except RuntimeError:
            pass
        else:
            self.resources.setdefault(task, set()).add(access)

    async def concurrently(self, awaitables):
        """
        Await concurrently for the given awaitables, and cancel them as soon as
        one raises an exception.
        """
        awaitables = list(awaitables)

        # Avoid creating asyncio.Tasks when it's not necessary, as it will
        # disable a the blocking path optimization of Target._execute_async()
        # that uses blocking calls as long as there is only one asyncio.Task
        # running on the event loop.
        if len(awaitables) == 1:
            return [await awaitables[0]]

        tasks = list(map(create_task, awaitables))

        current_task = asyncio.current_task()
        task_tree = self.task_tree

        try:
            node = task_tree[current_task]
        except KeyError:
            is_root_task = True
            node = set()
        else:
            is_root_task = False
        task_tree[current_task] = node

        task_tree.update({child: set() for child in tasks})
        node.update(tasks)

        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        finally:

            def get_children(task):
                immediate_children = task_tree[task]
                return frozenset(
                    itertools.chain(
                        [task],
                        immediate_children,
                        itertools.chain.from_iterable(
                            map(get_children, immediate_children)
                        ),
                    )
                )

            # Get the resources created during the execution of each subtask
            # (directly or indirectly)
            resources = {
                task: frozenset(
                    itertools.chain.from_iterable(
                        self.resources.get(child, []) for child in get_children(task)
                    )
                )
                for task in tasks
            }
            for (task1, resources1), (task2, resources2) in itertools.combinations(
                resources.items(), 2
            ):
                for res1, res2 in itertools.product(resources1, resources2):
                    if issubclass(res2.__class__, res1.__class__) and res1.overlap_with(
                        res2
                    ):
                        raise RuntimeError(
                            "Overlapping resources manipulated "
                            "in concurrent async tasks: "
                            "{} (task {}) and {} (task {})".format(
                                res1, task1.name, res2, task2.name
                            )
                        )

            if is_root_task:
                self.resources.clear()
                task_tree.clear()

    async def map_concurrently(self, f, keys):
        """
        Similar to `concurrently`,
        but maps the given function ``f`` on the given ``keys``.

        :return: A dictionary with ``keys`` as keys, and function result as
            values.
        """
        keys = list(keys)
        return dict(zip(keys, await self.concurrently(map(f, keys))))


def compose(*coros):
    """
    Compose coroutines, feeding the output of each as the input of the next
    one.

    ``await compose(f, g)(x)`` is equivalent to ``await f(await g(x))``

    .. note:: In Haskell, ``compose f g h`` would be equivalent to ``f <=< g <=< h``
    """

    async def f(*args, **kwargs):
        empty_dict = {}
        for coro in reversed(coros):
            x = coro(*args, **kwargs)
            # Allow mixing corountines and regular functions
            if asyncio.isfuture(x):
                x = await x
            args = [x]
            kwargs = empty_dict

        return x

    return f


class _AsyncPolymorphicFunction:
    """
    A callable that allows exposing both a synchronous and asynchronous API.

    When called, the blocking synchronous operation is called. The ```asyn``
    attribute gives access to the asynchronous version of the function, and all
    the other attribute access will be redirected to the async function.
    """

    def __init__(self, asyn, blocking):
        self.asyn = asyn
        self.blocking = blocking

    def __get__(self, *args, **kwargs):
        return self.__class__(
            asyn=self.asyn.__get__(*args, **kwargs),
            blocking=self.blocking.__get__(*args, **kwargs),
        )

    def __call__(self, *args, **kwargs):
        return self.blocking(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.asyn, attr)


class memoized_method:
    """
    Decorator to memmoize a method.

    It works for:

        * async methods (coroutine functions)
        * non-async methods
        * method already decorated with `devlib.asyn.asyncf`.

    .. note:: This decorator does not rely on hacks to hash unhashable data. If
        such input is required, it will either have to be coerced to a hashable
        first (e.g. converting a list to a tuple), or the code of
        `devlib.asyn.memoized_method` will have to be updated to do so.
    """

    def __init__(self, f):
        memo = self

        sig = inspect.signature(f)

        def bind(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            key = (bound.args[1:], tuple(sorted(bound.kwargs.items())))

            return (key, bound.args, bound.kwargs)

        def get_cache(self):
            try:
                cache = self.__dict__[memo.name]
            except KeyError:
                cache = {}
                self.__dict__[memo.name] = cache
            return cache

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def wrapper(self, *args, **kwargs):
                cache = get_cache(self)
                key, args, kwargs = bind(self, *args, **kwargs)
                try:
                    return cache[key]
                except KeyError:
                    x = await f(*args, **kwargs)
                    cache[key] = x
                    return x

        else:

            @functools.wraps(f)
            def wrapper(self, *args, **kwargs):
                cache = get_cache(self)
                key, args, kwargs = bind(self, *args, **kwargs)
                try:
                    return cache[key]
                except KeyError:
                    x = f(*args, **kwargs)
                    cache[key] = x
                    return x

        self.f = wrapper
        self._name = f.__name__

    @property
    def name(self):
        return "__memoization_cache_of_" + self._name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __get__(self, obj, owner=None):
        return self.f.__get__(obj, owner)

    def __set__(self, obj, value):
        raise RuntimeError("Cannot monkey-patch a memoized function")

    def __set_name__(self, owner, name):
        self._name = name


class _Genlet(greenlet):
    """
    Generator-like object based on ``greenlets``. It allows nested `_Genlet`
    to make their parent yield on their behalf, as if callees could decide to
    be annotated ``yield from`` without modifying the caller.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Forward the context variables to the greenlet, which will not happen
        # by default:
        # https://greenlet.readthedocs.io/en/latest/contextvars.html
        self.gr_context = contextvars.copy_context()

    @classmethod
    def from_coro(cls, coro):
        """
        Create a `_Genlet` from a given coroutine, treating it as a
        generator.
        """
        f = lambda value: self.consume_coro(coro, value)
        self = cls(f)
        return self

    def consume_coro(self, coro, value):
        """
        Send ``value`` to ``coro`` then consume the coroutine, passing all its
        yielded actions to the enclosing `_Genlet`. This allows crossing
        blocking calls layers as if they were async calls with `await`.
        """
        excep = None
        while True:
            try:
                if excep is None:
                    future = coro.send(value)
                else:
                    future = coro.throw(excep)

            except StopIteration as e:
                return e.value
            else:
                parent = self.parent
                # Switch back to the consumer that returns the values via
                # send()
                try:
                    value = parent.switch(future)
                except BaseException as e:
                    excep = e
                    value = None
                else:
                    excep = None

    @classmethod
    def get_enclosing(cls):
        """
        Get the immediately enclosing `_Genlet` in the callstack or
        ``None``.
        """
        g = greenlet.getcurrent()
        while not (isinstance(g, cls) or g is None):
            g = g.parent
        return g

    def _send_throw(self, value, excep):
        self.parent = greenlet.getcurrent()

        # Switch back to the function yielding values
        if excep is None:
            result = self.switch(value)
        else:
            result = self.throw(excep)

        if self:
            return result
        else:
            raise StopIteration(result)

    def gen_send(self, x):
        """
        Similar to generators' ``send`` method.
        """
        return self._send_throw(x, None)

    def gen_throw(self, x):
        """
        Similar to generators' ``throw`` method.
        """
        return self._send_throw(None, x)


class _AwaitableGenlet:
    """
    Wrap a coroutine with a `_Genlet` and wrap that to be awaitable.
    """

    @classmethod
    def wrap_coro(cls, coro):
        async def coro_f():
            # Make sure every new task will be instrumented since a task cannot
            # yield futures on behalf of another task. If that were to happen,
            # the task B trying to do a nested yield would switch back to task
            # A, asking to yield on its behalf. Since the event loop would be
            # currently handling task B, nothing would handle task A trying to
            # yield on behalf of B, leading to a deadlock.
            loop = asyncio.get_running_loop()
            _install_task_factory(loop)

            # Create a top-level _AwaitableGenlet that all nested runs will use
            # to yield their futures
            _coro = cls(coro)

            return await _coro

        return coro_f()

    def __init__(self, coro):
        self._coro = coro

    def __await__(self):
        coro = self._coro
        is_started = inspect.iscoroutine(coro) and coro.cr_running

        def genf():
            gen = _Genlet.from_coro(coro)
            value = None
            excep = None

            # The coroutine is already started, so we need to dispatch the
            # value from the upcoming send() to the gen without running
            # gen first.
            if is_started:
                try:
                    value = yield
                except BaseException as e:
                    excep = e

            while True:
                try:
                    if excep is None:
                        future = gen.gen_send(value)
                    else:
                        future = gen.gen_throw(excep)
                except StopIteration as e:
                    return e.value
                finally:
                    _set_current_context(gen.gr_context)

                try:
                    value = yield future
                except BaseException as e:
                    excep = e
                    value = None
                else:
                    excep = None

        gen = genf()
        if is_started:
            # Start the generator so it waits at the first yield point
            gen.gen_send(None)

        return gen


def _allow_nested_run(coro):
    if _Genlet.get_enclosing() is None:
        return _AwaitableGenlet.wrap_coro(coro)
    else:
        return coro


def allow_nested_run(coro):
    """
    Wrap the coroutine ``coro`` such that nested calls to `run` will be
    allowed.

    .. warning:: The coroutine needs to be consumed in the same OS thread it
        was created in.
    """
    return _allow_nested_run(coro)


# This thread runs coroutines that cannot be ran on the event loop in the
# current thread. Instead, they are scheduled in a separate thread where
# another event loop has been setup, so we can wrap coroutines before
# dispatching them there.
_CORO_THREAD_EXECUTOR = ThreadPoolExecutor(
    # Allow for a ridiculously large number so that we will never end up
    # queuing one job after another. This is critical as we could otherwise end
    # up in deadlock, if a job triggers another job and waits for it.
    max_workers=2**64,
)


def _check_executor_alive(executor):
    try:
        executor.submit(lambda: None)
    except RuntimeError:
        return False
    else:
        return True


_PATCHED_LOOP_LOCK = threading.Lock()
_PATCHED_LOOP = WeakSet()


def _install_task_factory(loop):
    """
    Install a task factory on the given event ``loop`` so that top-level
    coroutines are wrapped using `allow_nested_run`. This ensures that
    the nested `run` infrastructure will be available.
    """

    def install(loop):
        if sys.version_info >= (3, 11):

            def default_factory(loop, coro, context=None):
                return asyncio.Task(coro, loop=loop, context=context)

        else:

            def default_factory(loop, coro, context=None):
                return asyncio.Task(coro, loop=loop)

        make_task = loop.get_task_factory() or default_factory

        def factory(loop, coro, context=None):
            # Make sure each Task will be able to yield on behalf of its nested
            # await beneath blocking layers
            coro = _AwaitableGenlet.wrap_coro(coro)
            return make_task(loop, coro, context=context)

        loop.set_task_factory(factory)

    with _PATCHED_LOOP_LOCK:
        if loop in _PATCHED_LOOP:
            return
        else:
            install(loop)
            _PATCHED_LOOP.add(loop)


def _set_current_context(ctx):
    """
    Get all the variable from the passed ``ctx`` and set them in the current
    context.
    """
    for var, val in ctx.items():
        var.set(val)


class _CoroRunner(abc.ABC):
    """
    ABC for an object that can execute multiple coroutines in a given
    environment.

    This allows running coroutines for which it might be an assumption, such as
    the awaitables yielded by an async generator that are all attached to a
    single event loop.
    """

    @abc.abstractmethod
    def _run(self, coro):
        pass

    def run(self, coro):
        # Ensure we have a fresh coroutine. inspect.getcoroutinestate() does not
        # work on all objects that asyncio creates on some version of Python, such
        # as iterable_coroutine
        assert not (inspect.iscoroutine(coro) and coro.cr_running)
        return self._run(coro)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass


class _ThreadCoroRunner(_CoroRunner):
    """
    Run the coroutines on a thread picked from a
    `concurrent.futures.ThreadPoolExecutor`.

    Critically, this allows running multiple coroutines out of the same thread,
    which will be reserved until the runner ``__exit__`` method is called.
    """

    def __init__(self, future, jobq, resq):
        self._future = future
        self._jobq = jobq
        self._resq = resq

    @staticmethod
    def _thread_f(jobq, resq):
        def handle_jobs(runner):
            while True:
                job = jobq.get()
                if job is None:
                    return
                else:
                    ctx, coro = job
                    try:
                        value = ctx.run(runner.run, coro)
                    except BaseException as e:
                        value = None
                        excep = e
                    else:
                        excep = None

                    resq.put((ctx, excep, value))

        with _LoopCoroRunner(None) as runner:
            handle_jobs(runner)

    @classmethod
    def from_executor(cls, executor):
        jobq = queue.SimpleQueue()
        resq = queue.SimpleQueue()

        try:
            future = executor.submit(cls._thread_f, jobq, resq)
        except RuntimeError as e:
            if _check_executor_alive(executor):
                raise e
            else:
                raise RuntimeError(
                    "Devlib relies on nested asyncio implementation requiring threads. "
                    "These threads are not available while shutting down the interpreter."
                )

        return cls(
            jobq=jobq,
            resq=resq,
            future=future,
        )

    def _run(self, coro):
        ctx = contextvars.copy_context()
        self._jobq.put((ctx, coro))
        ctx, excep, value = self._resq.get()

        _set_current_context(ctx)

        if excep is None:
            return value
        else:
            raise excep

    def __exit__(self, *args, **kwargs):
        self._jobq.put(None)
        self._future.result()


class _LoopCoroRunner(_CoroRunner):
    """
    Run a coroutine on the given event loop.

    The passed event loop is assumed to not be running. If ``None`` is passed,
    a new event loop will be created in ``__enter__`` and closed in
    ``__exit__``.
    """

    def __init__(self, loop):
        self.loop = loop
        self._owned = False

    def _run(self, coro):
        loop = self.loop

        # Back-propagate the contextvars that could have been modified by the
        # coroutine. This could be handled by asyncio.Runner().run(...,
        # context=...) or loop.create_task(..., context=...) but these APIs are
        # only available since Python 3.11
        ctx = None

        async def capture_ctx():
            nonlocal ctx
            try:
                return await _allow_nested_run(coro)
            finally:
                ctx = contextvars.copy_context()

        try:
            return loop.run_until_complete(capture_ctx())
        finally:
            _set_current_context(ctx)

    def __enter__(self):
        loop = self.loop
        if loop is None:
            owned = True
            loop = asyncio.new_event_loop()
        else:
            owned = False

        asyncio.set_event_loop(loop)

        self.loop = loop
        self._owned = owned
        return self

    def __exit__(self, *args, **kwargs):
        if self._owned:
            asyncio.set_event_loop(None)
            _close_loop(self.loop)


class _GenletCoroRunner(_CoroRunner):
    """
    Run a coroutine assuming one of the parent coroutines was wrapped with
    `allow_nested_run`.
    """

    def __init__(self, g):
        self._g = g

    def _run(self, coro):
        return self._g.consume_coro(coro, None)


def _get_runner():
    executor = _CORO_THREAD_EXECUTOR
    g = _Genlet.get_enclosing()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # We have an coroutine wrapped with allow_nested_run() higher in the
    # callstack, that we will be able to use as a conduit to yield the
    # futures.
    if g is not None:
        return _GenletCoroRunner(g)
    # No event loop setup, so we can just make our own
    elif loop is None:
        return _LoopCoroRunner(None)
    # There is an event loop setup, but it is not currently running so we
    # can just re-use it.
    #
    # TODO: for now, this path is dead since asyncio.get_running_loop() will
    # always raise a RuntimeError if the loop is not running, even if
    # asyncio.set_event_loop() was used.
    elif not loop.is_running():
        return _LoopCoroRunner(loop)
    # There is an event loop currently running in our thread, so we cannot
    # just create another event loop and install it since asyncio forbids
    # that. The only choice is doing this in a separate thread that we
    # fully control.
    else:
        return _ThreadCoroRunner.from_executor(executor)


def run(coro):
    """
    Similar to `asyncio.run` but can be called while an event loop is
    running if a coroutine higher in the callstack has been wrapped using
    `allow_nested_run`.

    Note that context variables from `contextvars` will be available in
    the coroutine, and unlike with `asyncio.run`, any update to them will
    be reflected in the context of the caller. This allows context variable
    updates to cross an arbitrary number of run layers, as if all those layers
    were just part of the same coroutine.
    """
    runner = _get_runner()
    with runner as runner:
        return runner.run(coro)


def asyncf(f):
    """
    Decorator used to turn a coroutine into a blocking function, with an
    optional asynchronous API.

    **Example**::

        @asyncf
        async def foo(x):
            await do_some_async_things(x)
            return x

        # Blocking call, just as if the function was synchronous, except it may
        # use asynchronous code inside, e.g. to do concurrent operations.
        foo(42)

        # Asynchronous API, foo.asyn being a corountine
        await foo.asyn(42)

    This allows the same implementation to be both used as blocking for ease of
    use and backward compatibility, or exposed as a corountine for callers that
    can deal with awaitables.
    """

    @functools.wraps(f)
    def blocking(*args, **kwargs):
        # Since run() needs a corountine, make sure we provide one
        async def wrapper():
            x = f(*args, **kwargs)
            # Async generators have to be consumed and accumulated in a list
            # before crossing a blocking boundary.
            if inspect.isasyncgen(x):

                def genf():
                    asyncgen = x.__aiter__()
                    while True:
                        try:
                            yield run(asyncgen.__anext__())
                        except StopAsyncIteration:
                            return

                return genf()
            else:
                return await x

        return run(wrapper())

    return _AsyncPolymorphicFunction(
        asyn=f,
        blocking=blocking,
    )


class _AsyncPolymorphicCMState:
    def __init__(self):
        self.nesting = 0
        self.runner = None

    def _update_nesting(self, n):
        x = self.nesting
        assert x >= 0
        x = x + n
        self.nesting = x
        return bool(x)

    def _get_runner(self):
        runner = self.runner
        if runner is None:
            assert not self.nesting
            runner = _get_runner()
            runner.__enter__()
        self.runner = runner
        return runner

    def _cleanup_runner(self, force=False):
        def cleanup():
            self.runner = None
            if runner is not None:
                runner.__exit__(None, None, None)

        runner = self.runner
        if force:
            cleanup()
        else:
            assert runner is not None
            if not self._update_nesting(0):
                cleanup()


class _AsyncPolymorphicCM:
    """
    Wrap an async context manager such that it exposes a synchronous API as
    well for backward compatibility.
    """

    def __init__(self, async_cm):
        self.cm = async_cm
        self._state = threading.local()

    def _get_state(self):
        try:
            return self._state.x
        except AttributeError:
            state = _AsyncPolymorphicCMState()
            self._state.x = state
            return state

    def _delete_state(self):
        try:
            del self._state.x
        except AttributeError:
            pass

    def __aenter__(self, *args, **kwargs):
        return self.cm.__aenter__(*args, **kwargs)

    def __aexit__(self, *args, **kwargs):
        return self.cm.__aexit__(*args, **kwargs)

    @staticmethod
    def _exit(state):
        state._update_nesting(-1)
        state._cleanup_runner()

    def __enter__(self, *args, **kwargs):
        state = self._get_state()
        runner = state._get_runner()

        # Increase the nesting count _before_ we start running the
        # coroutine, in case it is a recursive context manager
        state._update_nesting(1)

        try:
            coro = self.cm.__aenter__(*args, **kwargs)
            return runner.run(coro)
        except BaseException:
            self._exit(state)
            raise

    def __exit__(self, *args, **kwargs):
        coro = self.cm.__aexit__(*args, **kwargs)

        state = self._get_state()
        runner = state._get_runner()

        try:
            return runner.run(coro)
        finally:
            self._exit(state)

    def __del__(self):
        self._get_state()._cleanup_runner(force=True)


def asynccontextmanager(f):
    """
    Same as `contextlib.asynccontextmanager` except that it can also be
    used with a regular ``with`` statement for backward compatibility.
    """
    f = contextlib.asynccontextmanager(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        cm = f(*args, **kwargs)
        return _AsyncPolymorphicCM(cm)

    return wrapper


class ConcurrentAccessBase(abc.ABC):
    """
    Abstract Base Class for resources tracked by `concurrently`.
    """

    @abc.abstractmethod
    def overlap_with(self, other):
        """
        Return ``True`` if the resource overlaps with the given one.

        :param other: Resources that should not overlap with ``self``.
        :type other: devlib.utils.asym.ConcurrentAccessBase

        .. note:: It is guaranteed that ``other`` will be a subclass of our
            class.
        """


class PathAccess(ConcurrentAccessBase):
    """
    Concurrent resource representing a file access.

    :param namespace: Identifier of the namespace of the path. One of "target" or "host".
    :type namespace: str

    :param path: Normalized path to the file.
    :type path: str

    :param mode: Opening mode of the file. Can be ``"r"`` for read and ``"w"``
        for writing.
    :type mode: str
    """

    def __init__(self, namespace, path, mode):
        assert namespace in ("host", "target")
        self.namespace = namespace
        assert mode in ("r", "w")
        self.mode = mode
        self.path = (
            os.path.abspath(path) if namespace == "host" else os.path.normpath(path)
        )

    def overlap_with(self, other):
        path1 = pathlib.Path(self.path).resolve()
        path2 = pathlib.Path(other.path).resolve()
        return (
            self.namespace == other.namespace
            and "w" in (self.mode, other.mode)
            and (path1 == path2 or path1 in path2.parents or path2 in path1.parents)
        )

    def __str__(self):
        mode = {
            "r": "read",
            "w": "write",
        }[self.mode]
        return "{} ({})".format(self.path, mode)
