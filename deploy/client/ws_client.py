"""Thin WebSocket client for deploy/policy_server.py (robot side; numpy + msgpack + websockets only).

Same shape as OpenWAM's benchmarks/utils/transport.py::WSPolicyClient: one persistent connection, obs/reset
reconnect once on a dropped socket, ping fails fast.
"""

import pathlib
import sys

try:
    from openpi_client import msgpack_numpy
except ImportError:  # robot machine with the repo checked out but openpi-client not installed
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "packages" / "openpi-client" / "src"))
    from openpi_client import msgpack_numpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import protocol as P  # noqa: E402


class ServerError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class PolicyClient:
    def __init__(self, url: str, timeout: float = 10.0, open_timeout: float = 5.0):
        self.url = url if url.startswith("ws") else f"ws://{url}"
        self.timeout = timeout
        self.open_timeout = open_timeout
        self._ws = None
        self._packer = msgpack_numpy.Packer()

    def _connect(self):
        if self._ws is None:
            from websockets.sync.client import connect

            self._ws = connect(self.url, max_size=None, compression=None, open_timeout=self.open_timeout,
                               ping_interval=None, proxy=None)

    def _roundtrip(self, message: dict, reconnect: bool = True) -> dict:
        from websockets.exceptions import ConnectionClosed

        body = self._packer.pack(message)
        for attempt in (0, 1) if reconnect else (0,):
            self._connect()
            try:
                self._ws.send(body)
                raw = self._ws.recv(timeout=self.timeout)
                break
            except (OSError, ConnectionClosed):
                self.close()
                if attempt == 1 or not reconnect:
                    raise
        reply = msgpack_numpy.unpackb(raw)
        if reply.get("type") == P.ERROR:
            raise ServerError(reply.get("code", ""), reply.get("message", ""))
        return reply

    def infer(self, obs: dict) -> dict:
        # no resend: a retried obs would be a second, stale inference
        return self._roundtrip({**obs, "type": P.OBS}, reconnect=False)

    def reset(self) -> dict:
        return self._roundtrip({"type": P.RESET})

    def ping(self) -> dict:
        return self._roundtrip({"type": P.PING}, reconnect=False)

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
