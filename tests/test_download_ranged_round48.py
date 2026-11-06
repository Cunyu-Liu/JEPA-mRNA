import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools.download_ranged_round48 import download_part


PAYLOAD = b"abcdefghijkl"


class RangeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        start = int(self.headers["Range"].split("=")[1].split("-")[0])
        if self.path == "/ignore":
            body = PAYLOAD
            self.send_response(200)
        else:
            body = PAYLOAD[start:] + (b"extra" if self.path == "/extra" else b"")
            self.send_response(206)
            range_start = 0 if self.path == "/wrong" else start
            self.send_header("Content-Range", f"bytes {range_start}-11/12")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def test_resume_validates_range_and_caps_response(server_url, tmp_path):
    path = tmp_path / "chunk"
    path.write_bytes(PAYLOAD[:4])
    result = download_part(server_url + "/extra", path, 0, 11)
    assert result["status"] == "downloaded"
    assert path.read_bytes() == PAYLOAD


def test_wrong_content_range_does_not_append(server_url, tmp_path):
    path = tmp_path / "chunk"
    path.write_bytes(PAYLOAD[:4])
    with pytest.raises(RuntimeError, match="invalid Content-Range"):
        download_part(server_url + "/wrong", path, 0, 11, retries=1)
    assert path.read_bytes() == PAYLOAD[:4]


def test_ignored_resume_does_not_overwrite(server_url, tmp_path):
    path = tmp_path / "chunk"
    path.write_bytes(PAYLOAD[:4])
    with pytest.raises(RuntimeError, match="ignored"):
        download_part(server_url + "/ignore", path, 0, 11, retries=1)
    assert path.read_bytes() == PAYLOAD[:4]


def test_complete_chunk_is_not_refetched(server_url, tmp_path):
    path = tmp_path / "chunk"
    path.write_bytes(PAYLOAD)
    result = download_part(server_url + "/wrong", path, 0, 11)
    assert result == {"part": "chunk", "bytes": 12, "status": "reused"}
