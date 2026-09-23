# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Host-side, allowlisted HTTP egress exposed to confined code as a tool."""

import asyncio
import http.client
import ipaddress
import socket
import urllib.parse
import urllib.request
from typing import Dict
from typing import List
from typing import Optional

EGRESS_MAX_BYTES = 5 * 1024 * 1024


def host_allowed(host: str, patterns: List[str]) -> bool:
    """Whether ``host`` matches the egress allowlist.

    Case-insensitive, trailing dot ignored. A bare entry matches that host
    exactly; a ``*.example.com`` entry matches any subdomain **and** the apex
    ``example.com``. No entry matches everything; an empty allowlist denies all.
    """
    host = (host or "").lower().rstrip(".")
    for pat in patterns:
        pat = pat.lower().rstrip(".")
        if pat.startswith("*."):
            suffix = pat[2:]
            if host == suffix or host.endswith("." + suffix):
                return True
        elif host == pat:
            return True
    return False


def reject_private(host: str, port: Optional[int], scheme: str) -> str:
    """Validate ``host`` resolves to a public address; return that pinned IP.

    A second gate beyond the hostname allowlist: refuses loopback, private
    (RFC 1918), link-local (incl. the ``169.254.169.254`` cloud-metadata IP),
    reserved, multicast and unspecified addresses (``PermissionError`` if any
    resolved address is non-public). Returns the first validated IP so the
    caller can **pin** the connection to it, closing the DNS-rebinding window
    where the host could re-resolve to an internal address between this check
    and the connect.
    """
    port = port or (443 if scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise PermissionError(f"cannot resolve host {host!r}: {exc}")
    chosen = None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise PermissionError(
                f"host {host!r} resolves to non-public address {ip} "
                "(set block_private_egress=False to allow internal targets)"
            )
        if chosen is None:
            chosen = info[4][0]
    if chosen is None:
        raise PermissionError(f"host {host!r} did not resolve to any address")
    return chosen


def make_egress_tool(patterns: List[str], timeout: float, block_private: bool):
    """Build the bound ``http_fetch`` callable enforcing ``patterns`` host-side.

    Exposed inside the sandbox like any other bound function (over the host RPC
    bridge, which works even when confinement has cut the network), giving
    confined code an allowlisted egress path and *only* that path. When
    ``block_private`` is True, hosts resolving to non-public addresses are also
    refused (SSRF guard).
    """

    allow = list(patterns)

    def fetch(url, method, headers, data):
        # Blocking, so it runs in a worker thread. Redirects are re-checked,
        # so an allowlisted host cannot bounce the request off-list, and
        # (unless ``block_private`` is False) every hop's resolved address must
        # be public **and the connection is pinned to that validated IP**, so
        # a host cannot re-resolve to an internal address after the check (no
        # DNS-rebinding TOCTOU). TLS still validates the cert against the
        # original hostname (SNI).
        # host -> validated public IP, populated by ``_check`` per hop; the pinned
        # connection classes below dial this IP instead of re-resolving the name.
        pinned: Dict[str, str] = {}

        def _check(u):
            parts = urllib.parse.urlsplit(u)
            if parts.scheme not in ("http", "https"):
                raise PermissionError(f"scheme not allowed: {parts.scheme!r}")
            host = parts.hostname or ""
            if not host_allowed(host, allow):
                raise PermissionError(f"host not in allowlist: {host!r}")
            if block_private:
                pinned[host] = reject_private(host, parts.port, parts.scheme)

        _check(url)

        class _PinnedHTTPConnection(http.client.HTTPConnection):
            def connect(self):
                target = pinned.get(self.host, self.host)
                self.sock = socket.create_connection(
                    (target, self.port), self.timeout, self.source_address
                )

        class _PinnedHTTPSConnection(http.client.HTTPSConnection):
            def connect(self):
                target = pinned.get(self.host, self.host)
                sock = socket.create_connection(
                    (target, self.port), self.timeout, self.source_address
                )
                # server_hostname is the *name*, not the pinned IP, so SNI and cert
                # hostname verification still check the certificate against the host.
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

        class _PinnedHTTPHandler(urllib.request.HTTPHandler):
            def http_open(self, req):
                return self.do_open(_PinnedHTTPConnection, req)

        class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(self, req):
                return self.do_open(_PinnedHTTPSConnection, req)

        class _Guard(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                _check(newurl)  # re-checks allowlist + re-pins the redirect target
                return super().redirect_request(req, fp, code, msg, hdrs, newurl)

        body = data.encode("utf-8") if isinstance(data, str) else data
        req = urllib.request.Request(
            url, data=body, method=(method or "GET").upper(), headers=headers or {}
        )
        opener = urllib.request.build_opener(
            _Guard, _PinnedHTTPHandler, _PinnedHTTPSHandler
        )
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read(EGRESS_MAX_BYTES + 1)
            truncated = len(raw) > EGRESS_MAX_BYTES
            text = raw[:EGRESS_MAX_BYTES].decode("utf-8", errors="replace")
            return {
                "status": resp.status,
                "headers": dict(resp.headers.items()),
                "body": text,
                "truncated": truncated,
                "url": resp.geturl(),
            }

    async def http_fetch(url, method="GET", headers=None, data=None):
        """Fetch an allowlisted HTTP(S) URL via the host and return the response.

        Args:
            url (str): Absolute ``http(s)://`` URL; its host (and any redirect
                target) must be on the sandbox's egress allowlist.
            method (str): HTTP method (default ``"GET"``).
            headers (dict): Optional request headers.
            data (str): Optional request body.

        Returns:
            dict: ``status``, ``headers``, ``body`` (text, capped), ``truncated``
            and the final ``url``. Raises ``PermissionError`` if the host is not
            allowlisted (or resolves to a non-public address).
        """
        return await asyncio.to_thread(fetch, url, method, headers, data)

    return http_fetch
