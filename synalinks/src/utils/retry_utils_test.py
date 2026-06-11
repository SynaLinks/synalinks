# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import types

from synalinks.src import testing
from synalinks.src.utils import retry_utils


class _Response:
    def __init__(self, headers):
        self.headers = headers


class _RateLimitError(Exception):
    def __init__(self, headers=None, retry_after=None):
        self.response = _Response(headers) if headers is not None else None
        if retry_after is not None:
            self.retry_after = retry_after


def _retry_state(exc, attempt_number=1):
    """Minimal stand-in for a tenacity RetryCallState carrying a failed outcome."""
    outcome = types.SimpleNamespace(exception=lambda: exc, failed=True)
    return types.SimpleNamespace(outcome=outcome, attempt_number=attempt_number)


class RetryAfterSecondsTest(testing.TestCase):
    def test_reads_retry_after_header(self):
        exc = _RateLimitError(headers={"retry-after": "30"})
        self.assertEqual(retry_utils.retry_after_seconds(exc), 30.0)

    def test_reads_capitalized_header(self):
        exc = _RateLimitError(headers={"Retry-After": "12"})
        self.assertEqual(retry_utils.retry_after_seconds(exc), 12.0)

    def test_falls_back_to_attribute(self):
        exc = _RateLimitError(retry_after="7")
        self.assertEqual(retry_utils.retry_after_seconds(exc), 7.0)

    def test_none_when_absent(self):
        self.assertIsNone(retry_utils.retry_after_seconds(_RateLimitError()))

    def test_none_when_unparseable(self):
        # HTTP-date form isn't a plain seconds value -> fall back to backoff.
        exc = _RateLimitError(headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        self.assertIsNone(retry_utils.retry_after_seconds(exc))


class RateLimitAwareWaitTest(testing.TestCase):
    def test_honors_retry_after(self):
        wait = retry_utils.rate_limit_aware_wait(max_wait=60)
        exc = _RateLimitError(headers={"retry-after": "30"})
        self.assertEqual(wait(_retry_state(exc)), 30.0)

    def test_caps_at_max_wait(self):
        wait = retry_utils.rate_limit_aware_wait(max_wait=60)
        exc = _RateLimitError(headers={"retry-after": "200"})
        self.assertEqual(wait(_retry_state(exc)), 60)

    def test_falls_back_to_exponential_without_header(self):
        wait = retry_utils.rate_limit_aware_wait(max_wait=60)
        # No Retry-After -> exponential backoff, which grows with the attempt.
        first = wait(_retry_state(_RateLimitError(), attempt_number=1))
        later = wait(_retry_state(_RateLimitError(), attempt_number=3))
        self.assertGreater(later, first)
