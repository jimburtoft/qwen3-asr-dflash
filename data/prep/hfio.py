# SPDX-License-Identifier: Apache-2.0
"""Minimal, rate-limit-friendly HTTP access to Hugging Face dataset files at a pinned revision.

One resolver call per file (the signed CDN URL is cached and re-resolved on 403/expiry); parquet
footers / column chunks are fetched with HTTP Range requests so only the needed columns and row
groups are downloaded.
"""
import io
import os
import time

import requests

HF = "https://huggingface.co"
_sess = {}


def _s():
    """Per-process requests session (safe after fork)."""
    pid = os.getpid()
    if pid not in _sess:
        _sess.clear()
        _sess[pid] = requests.Session()
    return _sess[pid]


_TOKEN = os.environ.get("HF_TOKEN")
_HDR = {"Authorization": f"Bearer {_TOKEN}"} if _TOKEN else {}


def resolve_url(repo, rev, path):
    return f"{HF}/datasets/{repo}/resolve/{rev}/{path}"


def _resolve(repo, rev, path):
    """Return (final_url, size). Follows redirects without downloading the body."""
    for attempt in range(8):
        try:
            r = _s().get(resolve_url(repo, rev, path), headers={**_HDR, "Range": "bytes=0-0"},
                             allow_redirects=True, stream=True, timeout=60)
            r.close()
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            if r.status_code not in (200, 206):
                raise IOError(f"resolve {path}: HTTP {r.status_code}")
            cr = r.headers.get("Content-Range", "")
            size = int(cr.split("/")[-1]) if "/" in cr else int(r.headers["Content-Length"])
            return r.url, size
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(5 * (attempt + 1))
    raise IOError(f"resolve failed: {repo}@{rev}/{path}")


def list_files(repo, rev):
    from huggingface_hub import HfApi
    for attempt in range(6):
        try:
            return HfApi(token=_TOKEN).list_repo_files(repo, repo_type="dataset", revision=rev)
        except Exception:  # noqa: BLE001
            time.sleep(20 * (attempt + 1))
    raise IOError(f"list_repo_files failed: {repo}")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (enough for pyarrow.parquet)."""

    def __init__(self, repo, rev, path):
        self.repo, self.rev, self.path = repo, rev, path
        self.url, self.size = _resolve(repo, rev, path)
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, a, b):
        """GET bytes [a, b] with a hard wall-clock deadline (stalled transfers are retried)."""
        want = b - a + 1
        deadline_s = 60 + want / (2 << 20)  # >= 2 MB/s or retry
        for attempt in range(12):
            try:
                t0 = time.time()
                with _s().get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=(20, 30), stream=True) as r:
                    if r.status_code in (403, 410):
                        self.url, _ = _resolve(self.repo, self.rev, self.path)
                        continue
                    if r.status_code not in (200, 206):
                        raise requests.ConnectionError(f"HTTP {r.status_code}")
                    buf = bytearray()
                    for c in r.iter_content(1 << 20):
                        buf += c
                        if r.status_code == 200 and a == 0 and len(buf) >= want:
                            break
                        if time.time() - t0 > deadline_s:
                            raise requests.Timeout("deadline")
                    if r.status_code == 206 and len(buf) == want:
                        return bytes(buf)
                    if r.status_code == 200 and a == 0 and len(buf) >= want:
                        return bytes(buf[:want])
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ContentDecodingError):
                _sess.clear()
            time.sleep(min(30, 2 ** attempt))
        raise IOError(f"range read failed {self.path} {a}-{b}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray()
        a = self.pos
        CH = 64 << 20
        while a < self.pos + n:
            b = min(self.pos + n, a + CH) - 1
            out += self._get(a, b)
            a = b + 1
        self.pos += n
        return bytes(out)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[: len(d)] = d
        return len(d)


class CachedRangeFile(HttpRangeFile):
    """HttpRangeFile with an explicit prefetch cache (one request per merged byte range)."""

    def __init__(self, repo, rev, path):
        super().__init__(repo, rev, path)
        self.blocks = []  # list of (start, bytes)

    def prefetch(self, ranges, max_gap=1 << 20):
        rs = sorted((a, b) for a, b in ranges if b > a)
        merged = []
        for a, b in rs:
            if merged and a - merged[-1][1] <= max_gap:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            self.blocks.append((a, self._read_span(a, b)))

    def drop_cache(self):
        self.blocks = []

    def _read_span(self, a, b):
        out = bytearray()
        CH = 64 << 20
        while a < b:
            e = min(b, a + CH)
            out += self._get(a, e - 1)
            a = e
        return bytes(out)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        for s, blob in self.blocks:
            if s <= self.pos and self.pos + n <= s + len(blob):
                d = blob[self.pos - s: self.pos - s + n]
                self.pos += n
                return d
        d = self._read_span(self.pos, self.pos + n)
        self.pos += n
        return d


def chunk_ranges(md, rg_indices, columns=None):
    """Byte ranges of the given row groups (optionally only the named leaf columns)."""
    out = []
    for g in rg_indices:
        rg = md.row_group(g)
        for c in range(rg.num_columns):
            cc = rg.column(c)
            name = cc.path_in_schema
            if columns is not None and name not in columns and name.split(".")[0] not in columns:
                continue
            start = cc.data_page_offset
            if cc.has_dictionary_page and cc.dictionary_page_offset and cc.dictionary_page_offset < start:
                start = cc.dictionary_page_offset
            out.append((start, start + cc.total_compressed_size))
    return out


def download(repo, rev, path, dest):
    """Download a whole file (streamed) to dest; skip if already complete."""
    url, size = _resolve(repo, rev, path)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest
    tmp = dest + ".part"
    for attempt in range(8):
        try:
            with _s().get(url, stream=True, timeout=(20, 60)) as r:
                if r.status_code in (403, 410):
                    url, size = _resolve(repo, rev, path)
                    continue
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for c in r.iter_content(8 << 20):
                        fh.write(c)
            if os.path.getsize(tmp) == size:
                os.replace(tmp, dest)
                return dest
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            pass
        time.sleep(min(60, 2 ** attempt))
    raise IOError(f"download failed {path}")


def stream(repo, rev, path):
    """Return a streaming response (raw file-like) for a whole file."""
    url, _ = _resolve(repo, rev, path)
    r = _s().get(url, stream=True, timeout=(20, 60))
    r.raise_for_status()
    r.raw.decode_content = True
    return r
