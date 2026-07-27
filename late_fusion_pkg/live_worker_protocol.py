"""Length-prefixed JSON protocol used by live detector workers."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


MAX_MESSAGE_BYTES = 256 * 1024 * 1024


class ProtocolError(RuntimeError):
    """Raised when a worker protocol message is malformed."""


def _recvall(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProtocolError("socket closed while reading message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message too large: {len(data)} bytes")
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_json(sock: socket.socket) -> dict[str, Any]:
    header = _recvall(sock, 4)
    (size,) = struct.unpack(">I", header)
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"invalid message size: {size}")
    data = _recvall(sock, size)
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("expected JSON object")
    return payload
