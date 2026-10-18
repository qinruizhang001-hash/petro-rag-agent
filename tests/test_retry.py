"""
指数退避重试的离线单测（零依赖）
===============================
``utils/retry.py`` 故意不 import openai / httpx，就是为了能在不装任何三方包
的环境里被测到 —— 而这个模块恰恰是"上游限流时唯一挡在中间的代码"，
最不该只能靠人工跑一次真实限流来验证。

这里用**假的异常类**模拟真实 SDK 的异常形态（类名 + status_code），
覆盖三件事：
  1. 哪些错误该重试、哪些该立刻失败（4xx 不该重试）
  2. 退避时长是否按指数增长并有上限
  3. 尝试次数用尽后抛出的是**原异常**，不是包装过的异常
"""
from __future__ import annotations

import unittest

from utils.retry import backoff_delay, call_with_retry, is_transient, status_of


class FakeAPIError(Exception):
    """模拟 openai.APIStatusError：带 status_code。"""

    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class FakeWrappedError(Exception):
    """模拟只有 .response.status_code 的错误形态。"""

    def __init__(self, status_code: int):
        super().__init__("wrapped")
        self.response = FakeResponse(status_code)


class RateLimitError(Exception):
    """类名命中即可，不需要 status_code（与 openai 的实际形态一致）。"""


class APIConnectionError(Exception):
    pass


class TestIsTransient(unittest.TestCase):
    def test_rate_limit_status_is_retried(self):
        self.assertTrue(is_transient(FakeAPIError(429)))

    def test_server_errors_are_retried(self):
        for code in (500, 502, 503, 504):
            self.assertTrue(is_transient(FakeAPIError(code)), code)

    def test_client_errors_are_not_retried(self):
        # 401 鉴权失败 / 404 模型名不对 / 400 参数非法：重试只会重复同一个错误
        for code in (400, 401, 403, 404, 422):
            self.assertFalse(is_transient(FakeAPIError(code)), code)

    def test_connection_errors_are_retried_by_name(self):
        self.assertTrue(is_transient(APIConnectionError("boom")))
        self.assertTrue(is_transient(RateLimitError("slow down")))

    def test_plain_errors_are_not_retried(self):
        self.assertFalse(is_transient(ValueError("bad json")))
        self.assertFalse(is_transient(KeyError("missing")))

    def test_status_from_wrapped_response(self):
        self.assertEqual(status_of(FakeWrappedError(503)), 503)
        self.assertTrue(is_transient(FakeWrappedError(503)))
        self.assertFalse(is_transient(FakeWrappedError(404)))

    def test_status_none_when_absent(self):
        self.assertIsNone(status_of(ValueError("x")))


class TestBackoffDelay(unittest.TestCase):
    def test_grows_exponentially(self):
        d1 = backoff_delay(1, base=1.0, factor=2.0, max_delay=100, jitter=0)
        d2 = backoff_delay(2, base=1.0, factor=2.0, max_delay=100, jitter=0)
        d3 = backoff_delay(3, base=1.0, factor=2.0, max_delay=100, jitter=0)
        self.assertEqual((d1, d2, d3), (1.0, 2.0, 4.0))

    def test_capped_by_max_delay(self):
        self.assertEqual(backoff_delay(10, base=1.0, factor=2.0, max_delay=8.0, jitter=0), 8.0)

    def test_jitter_stays_within_band(self):
        for _ in range(200):
            d = backoff_delay(3, base=1.0, factor=2.0, max_delay=100, jitter=0.25)
            self.assertGreaterEqual(d, 3.0)
            self.assertLessEqual(d, 5.0)

    def test_attempt_zero_is_treated_as_first(self):
        self.assertEqual(
            backoff_delay(0, base=1.0, factor=2.0, max_delay=100, jitter=0), 1.0
        )


class TestCallWithRetry(unittest.TestCase):
    def setUp(self):
        self.sleeps: list[float] = []

    def _sleeper(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def test_success_first_try_does_not_sleep(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        self.assertEqual(
            call_with_retry(fn, attempts=3, sleep=self._sleeper, jitter=0), "ok"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.sleeps, [])

    def test_retries_then_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise APIConnectionError("boom")
            return "recovered"

        result = call_with_retry(
            fn, attempts=3, base_delay=1.0, sleep=self._sleeper, jitter=0
        )
        self.assertEqual(result, "recovered")
        self.assertEqual(len(calls), 3)
        # 两次退避：1s、2s
        self.assertEqual(self.sleeps, [1.0, 2.0])

    def test_exhausted_attempts_reraise_original_exception(self):
        calls = []

        def fn():
            calls.append(1)
            raise RateLimitError("still limited")

        with self.assertRaises(RateLimitError):
            call_with_retry(fn, attempts=3, sleep=self._sleeper, jitter=0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(self.sleeps), 2)  # 最后一次失败后不再等待

    def test_non_transient_error_fails_fast(self):
        calls = []

        def fn():
            calls.append(1)
            raise FakeAPIError(401)

        with self.assertRaises(FakeAPIError):
            call_with_retry(fn, attempts=3, sleep=self._sleeper, jitter=0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.sleeps, [])

    def test_passes_through_args_and_kwargs(self):
        def fn(a, b=None):
            return (a, b)

        self.assertEqual(
            call_with_retry(fn, 1, b=2, attempts=1, sleep=self._sleeper), (1, 2)
        )

    def test_rejects_zero_attempts(self):
        with self.assertRaises(ValueError):
            call_with_retry(lambda: 1, attempts=0, sleep=self._sleeper)


if __name__ == "__main__":
    unittest.main()
