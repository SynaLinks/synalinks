"""Retry helpers shared by the language- and embedding-model wrappers.

The model wrappers retry failed `litellm` calls with tenacity. Plain exponential
backoff gives up too quickly under sustained provider rate limiting (e.g. a small
Azure OpenAI TPM quota), so these helpers honor the server-instructed
``Retry-After`` delay on 429s and only fall back to exponential backoff for other
errors.
"""

from tenacity import wait_exponential


def retry_after_seconds(exc):
    """Server-instructed retry delay (seconds) for a rate-limit error.

    Reads the standard ``Retry-After`` header that providers (OpenAI, Azure
    OpenAI, Anthropic, ...) attach to a 429 response, falling back to the
    ``retry_after`` attribute litellm/openai sometimes set on the exception.
    Returns ``None`` when absent or unparseable (so the caller uses exponential
    backoff instead). ``Retry-After`` given as a delay in seconds is honored;
    the HTTP-date form is treated as unparseable and falls back to backoff.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        value = getattr(exc, "retry_after", None)
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def rate_limit_aware_wait(max_wait=60.0):
    """A tenacity ``wait`` that honors a rate-limit ``Retry-After`` header.

    On a 429 carrying ``Retry-After`` it waits exactly that long (capped at
    ``max_wait`` so a hostile/huge value can't hang the call); for every other
    error it falls back to the previous exponential backoff. Honoring the
    server-instructed delay is what lets a run ride out sustained throttling
    instead of exhausting short retries.
    """
    fallback = wait_exponential(multiplier=1, min=1, max=10)

    def _wait(retry_state):
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome is not None else None
        retry_after = retry_after_seconds(exc) if exc is not None else None
        if retry_after is not None:
            return min(retry_after, max_wait)
        return fallback(retry_state)

    return _wait
