#!/usr/bin/env python3
"""Fetch the frozen six-language NLLB error-audit inputs without running analysis.

Only metadata and six selected members are retrieved from Meta's 2.27 GB ZIP.
The Python standard library verifies each ZIP member's CRC. No archive paths
are extracted: explicitly named members are read and written to fixed paths.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

LANGUAGES = ("ace_Latn", "asm_Beng", "khm_Khmr", "lao_Laoo", "mya_Mymr", "som_Latn")
TARGET = "eng_Latn"
EXPECTED_LINES = 1012
OUTPUT_ZIP = "https://dl.fbaipublicfiles.com/large_objects/nllb/translations/nllb_200_moe_54b/flores_translations.zip"
EXPECTED_ZIP_SIZE = 2268203489
REFERENCES_TAR = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
EXPECTED_REFERENCES_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
AGENT = "RMT-Assignment4-reproducible-data-retrieval/1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def open_request(url: str, *, headers: dict | None = None, method: str = "GET"):
    request = urllib.request.Request(
        url, method=method,
        headers={"User-Agent": AGENT, "Accept-Encoding": "identity", **(headers or {})},
    )
    return urllib.request.urlopen(request, timeout=90)


class RemoteZipFile(io.RawIOBase):
    """A bounded HTTP Range reader compatible with zipfile.ZipFile."""

    def __init__(self, url: str, expected_size: int, cache_dir: Path | None = None):
        super().__init__()
        self.url = url
        self.position = 0
        self.cache_dir = cache_dir
        self.cache = []
        self.request_count = 0
        self.downloaded_bytes = 0
        with open_request(url, method="HEAD") as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
            self.last_modified = response.headers.get("Last-Modified")
        if self.size != expected_size:
            raise ValueError(f"ZIP size changed: expected {expected_size}, received {self.size}")
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            for path in cache_dir.glob("range_*.bin"):
                match = re.fullmatch(r"range_(\d+)_(\d+)\.bin", path.name)
                if match:
                    start, stop = map(int, match.groups())
                    if path.stat().st_size == stop - start:
                        self.cache.append((start, stop, path))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset: int, whence: int = 0):
        new = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if whence not in (0, 1, 2) or new < 0:
            raise ValueError("Invalid seek")
        self.position = new
        return new

    def read(self, size: int = -1):
        stop = self.size if size < 0 else min(self.size, self.position + size)
        start = self.position
        if stop <= start:
            return b""
        count = stop - start
        if count > 20_000_000:
            raise ValueError("Unexpected large range; refusing a full archive download")
        for cstart, cstop, path in self.cache:
            if cstart <= start and cstop >= stop:
                with path.open("rb") as stream:
                    stream.seek(start - cstart)
                    data = stream.read(count)
                self.position = stop
                return data
        headers = {"Range": f"bytes={start}-{stop - 1}"}
        if self.etag:
            headers["If-Range"] = self.etag
        with open_request(self.url, headers=headers) as response:
            expected_range = f"bytes {start}-{stop - 1}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("Server did not honor the requested range; no full download attempted")
            data = response.read(count + 1)
        if len(data) != count:
            raise ValueError("Incomplete or oversized range response")
        self.request_count += 1
        self.downloaded_bytes += len(data)
        if self.cache_dir is not None:
            path = self.cache_dir / f"range_{start}_{stop}.bin"
            path.write_bytes(data)
            self.cache.append((start, stop, path))
        self.position = stop
        return data


def validate_text(data: bytes, label: str):
    text = data.decode("utf-8")
    lines = text.splitlines()
    if len(lines) != EXPECTED_LINES:
        raise ValueError(f"{label}: expected {EXPECTED_LINES} lines, received {len(lines)}")
    return len(lines)


def write_original(path: Path, data: bytes, originals_dir: Path | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Existing input differs; refusing to overwrite {path}")
    path.write_bytes(data)
    if originals_dir is not None:
        originals_dir.mkdir(parents=True, exist_ok=True)
        original = originals_dir / path.name
        if original.exists() and original.read_bytes() != data:
            raise ValueError(f"Original copy differs: {original}")
        original.write_bytes(data)


def read_references(local_dir: Path | None):
    codes = (*LANGUAGES, TARGET)
    if local_dir is not None:
        return {code: (local_dir / f"{code}.devtest").read_bytes() for code in codes}, "local copy"
    with open_request(REFERENCES_TAR) as response:
        data = response.read(30_000_001)
    if len(data) > 30_000_000 or sha256(data) != EXPECTED_REFERENCES_SHA256:
        raise ValueError("FLORES archive size or SHA-256 differs from the frozen source")
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for code in codes:
            name = f"./flores200_dataset/devtest/{code}.devtest"
            member = archive.getmember(name)
            if not member.isfile() or member.size > 5_000_000:
                raise ValueError(f"Unsafe or unexpected reference member {name}")
            with archive.extractfile(member) as stream:
                result[code] = stream.read()
    return result, "verified original archive"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "error_audit")
    parser.add_argument("--refs-dir", type=Path, help="Optional existing extracted original FLORES devtest directory")
    parser.add_argument("--originals-dir", type=Path, help="Optional second location for unchanged downloaded hypotheses")
    parser.add_argument("--cache-dir", type=Path, help="Optional local cache of HTTP byte ranges")
    args = parser.parse_args()
    manifest_path = args.output_dir / "provenance.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    previous_hashes = {row["path"]: row["sha256"] for row in previous["files"]} if previous else {}
    files = []
    with RemoteZipFile(OUTPUT_ZIP, EXPECTED_ZIP_SIZE, args.cache_dir) as remote:
        with zipfile.ZipFile(remote) as archive:
            for language in LANGUAGES:
                member = f"flores_translations/flores200-{language}-{TARGET}-devtest.hyp"
                info = archive.getinfo(member)
                if info.is_dir() or info.file_size > 5_000_000:
                    raise ValueError(f"Unexpected hypothesis member {member}")
                data = archive.read(info)
                lines = validate_text(data, member)
                relative = f"hypotheses/{Path(member).name}"
                if relative in previous_hashes and sha256(data) != previous_hashes[relative]:
                    raise ValueError(f"Published file changed: {relative}")
                write_original(args.output_dir / relative, data, args.originals_dir)
                files.append({"path": relative, "role": "published_system_hypothesis", "source_language": language,
                              "target_language": TARGET, "url": OUTPUT_ZIP, "archive_member": member,
                              "bytes": len(data), "lines": lines, "sha256": sha256(data),
                              "zip_crc32": f"{info.CRC:08x}", "zip_compressed_bytes": info.compress_size})
                print(f"Retrieved {language} -> {TARGET}: {lines} lines", flush=True)
        archive_info = {"url": OUTPUT_ZIP, "bytes": remote.size, "etag": remote.etag,
                        "last_modified": remote.last_modified, "retrieval": "HTTP Range selected ZIP members"}
        network_info = {"range_requests": remote.request_count, "downloaded_zip_bytes": remote.downloaded_bytes}
    references, retrieval = read_references(args.refs_dir)
    for language, data in references.items():
        lines = validate_text(data, language)
        relative = f"references/{language}.devtest"
        if relative in previous_hashes and sha256(data) != previous_hashes[relative]:
            raise ValueError(f"Reference input changed: {relative}")
        write_original(args.output_dir / relative, data)
        files.append({"path": relative, "role": "reference" if language == TARGET else "source_text",
                      "language": language, "url": REFERENCES_TAR,
                      "archive_member": f"./flores200_dataset/devtest/{language}.devtest",
                      "archive_sha256": EXPECTED_REFERENCES_SHA256,
                      "bytes": len(data), "lines": lines, "sha256": sha256(data)})
    manifest = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "model": "NLLB-200 MoE 54.5B",
                "split": "FLORES-200 devtest", "fixed_source_languages": list(LANGUAGES), "target_language": TARGET,
                "archive": archive_info, "network": network_info, "references_retrieval": retrieval,
                "analysis_status": "retrieval_and_line_count_validation_only; no scoring or annotation",
                "licence_notice": "See ../../docs/licenses. Original NLLB model terms CC-BY-NC 4.0; FLORES data CC-BY-SA 4.0. No new licence is asserted over third-party materials.",
                "files": files}
    if previous:
        # Re-running the downloader verifies input identity without altering the
        # evidence of when the original retrieval happened.
        print("Verified all files against the existing provenance hashes.")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Input retrieval complete: {len(files)} files. No result calculations were performed.")


if __name__ == "__main__":
    main()
