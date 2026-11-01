"""Confirms the crash-isolation wrappers added for server.py's own HTTP-
facing call path (llm_decompose.decompose_prompt_isolated/ask_clarifying_
questions_isolated) actually survive what they're built for: a worker
process dying outright (the real failure mode -- a native GGML assertion
abort()s the whole process, confirmed live-reproduced against the real
local Qwen3 model) or hanging past its timeout, in both cases WITHOUT
taking the caller down and WITHOUT leaving the pool permanently broken
for the next call. decompose_prompt/ask_clarifying_questions themselves
are untouched (see tests/test_llm_decompose.py's own in-process
monkeypatch-based suite) -- this file only covers the new wrapper layer."""

import concurrent.futures
import os
import time
from unittest.mock import patch

import pytest

import llm_decompose


class _FakeFuture:
    def __init__(self, exc):
        self._exc = exc

    def result(self, timeout=None):
        raise self._exc


class _FakeBrokenPool:
    def submit(self, fn, *args):
        return _FakeFuture(concurrent.futures.process.BrokenProcessPool("simulated native worker crash"))


class _FakeTimeoutPool:
    def submit(self, fn, *args):
        return _FakeFuture(concurrent.futures.TimeoutError("simulated hang"))


def test_decompose_prompt_isolated_falls_back_on_worker_crash():
    reset_calls = []
    with (
        patch("llm_decompose._get_llm_pool", return_value=_FakeBrokenPool()),
        patch("llm_decompose._reset_llm_pool", side_effect=lambda: reset_calls.append(1)),
    ):
        result = llm_decompose.decompose_prompt_isolated("a widget with a hole")

    assert result.fell_back is True
    assert len(result.points) == 1
    assert result.points[0].description == "a widget with a hole"
    assert result.points[0].position_mm == (0.0, 0.0, 0.0)
    assert reset_calls == [1]  # the pool must be reset so the NEXT call gets a fresh worker


def test_decompose_prompt_isolated_falls_back_on_timeout():
    reset_calls = []
    with (
        patch("llm_decompose._get_llm_pool", return_value=_FakeTimeoutPool()),
        patch("llm_decompose._reset_llm_pool", side_effect=lambda: reset_calls.append(1)),
    ):
        result = llm_decompose.decompose_prompt_isolated("a bracket", timeout=0.01)

    assert result.fell_back is True
    assert reset_calls == [1]


def test_ask_clarifying_questions_isolated_falls_back_on_worker_crash():
    with (
        patch("llm_decompose._get_llm_pool", return_value=_FakeBrokenPool()),
        patch("llm_decompose._reset_llm_pool"),
    ):
        result = llm_decompose.ask_clarifying_questions_isolated("a vague request")

    assert result.questions == []
    assert result.raw_llm_output == ""


def _real_crash_worker():
    os._exit(1)  # the closest deterministic stand-in for a native abort() -- the process dies outright,
                 # no Python exception is ever raised inside it to catch


def _slow_worker(seconds):
    time.sleep(seconds)
    return "finished"


def test_reset_llm_pool_kills_a_still_running_worker():
    """The accumulating-orphaned-process gap: a worker that's still mid-
    task when _reset_llm_pool() is called must actually be killed, not
    merely detached -- confirmed, live-reproduced: ProcessPoolExecutor.
    shutdown(wait=False) alone leaves it running for however long its
    own task takes (minutes, for a real LLM generation)."""
    llm_decompose._reset_llm_pool()
    try:
        pool = llm_decompose._get_llm_pool()
        future = pool.submit(_slow_worker, 30)
        time.sleep(0.5)  # give the worker a moment to actually spawn
        proc = next(iter(pool._processes.values()))
        assert proc.is_alive()

        llm_decompose._reset_llm_pool()
        time.sleep(0.5)  # give the OS a moment to reap the killed process

        assert not proc.is_alive()  # actually killed, not just abandoned to run for 30s
    finally:
        llm_decompose._reset_llm_pool()


def test_llm_pool_recovers_after_a_real_worker_crash():
    """The actual multiprocessing mechanics, not mocked: a worker that
    genuinely dies produces a real BrokenProcessPool on .result(), and
    after _reset_llm_pool(), a brand new pool accepts and completes
    ordinary work normally -- confirming the recovery path survives a
    real process death, not just a simulated exception."""
    llm_decompose._reset_llm_pool()
    try:
        pool = llm_decompose._get_llm_pool()
        future = pool.submit(_real_crash_worker)
        with pytest.raises(concurrent.futures.process.BrokenProcessPool):
            future.result(timeout=15)

        llm_decompose._reset_llm_pool()
        pool2 = llm_decompose._get_llm_pool()
        future2 = pool2.submit(sum, [1, 2, 3])
        assert future2.result(timeout=15) == 6
    finally:
        llm_decompose._reset_llm_pool()
