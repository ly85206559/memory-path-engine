import io
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from scripts.download_longmemeval import download_file


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class DownloadLongMemEvalTests(unittest.TestCase):
    def test_download_file_retries_on_temporary_dns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "longmemeval.json"
            with patch(
                "scripts.download_longmemeval.urlopen",
                side_effect=[
                    URLError(socket.gaierror(-3, "Temporary failure in name resolution")),
                    _FakeResponse(b'{"ok": true}'),
                ],
            ) as mocked_urlopen, patch("scripts.download_longmemeval.time.sleep") as mocked_sleep:
                path = download_file(
                    url="https://example.test/longmemeval.json",
                    output_path=output_path,
                    force=True,
                    timeout=5.0,
                    retries=2,
                    retry_backoff_seconds=0.1,
                )

                self.assertEqual(path, output_path)
                self.assertEqual(output_path.read_text(encoding="utf-8"), '{"ok": true}')
                self.assertEqual(mocked_urlopen.call_count, 2)
                mocked_sleep.assert_called_once_with(0.1)

    def test_download_file_does_not_retry_client_http_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "longmemeval.json"
            with patch(
                "scripts.download_longmemeval.urlopen",
                side_effect=HTTPError(
                    url="https://example.test/longmemeval.json",
                    code=404,
                    msg="not found",
                    hdrs=None,
                    fp=None,
                ),
            ) as mocked_urlopen, patch("scripts.download_longmemeval.time.sleep") as mocked_sleep:
                with self.assertRaises(HTTPError):
                    download_file(
                        url="https://example.test/longmemeval.json",
                        output_path=output_path,
                        force=True,
                        timeout=5.0,
                        retries=2,
                        retry_backoff_seconds=0.1,
                    )

        self.assertEqual(mocked_urlopen.call_count, 1)
        mocked_sleep.assert_not_called()
