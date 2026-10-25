"""Download only NIR images, material labels, validity masks and splits from official MCubeS ZIP."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path, PurePosixPath
import struct
import threading
import time
import zipfile
import zlib

import requests

URL = "https://drive.usercontent.google.com/download?id=14egTCyC0Pampb7imrXVwaDRffHN7FZxh&export=download&confirm=t"
SOURCE = "https://github.com/kyotovision-public/multimodal-material-segmentation"
ALLOWED = {"NIR_warped", "NIR_warped_mask", "GT", "list_folder"}
LOCAL = threading.local()


def get_range(start, end):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    last = None
    for attempt in range(4):
        try:
            response = LOCAL.session.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 120))
            response.raise_for_status()
            expected = f"bytes {start}-{end}/"
            if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                raise RuntimeError(f"Server did not honor requested range {start}-{end}: "
                                   f"{response.status_code} {response.headers.get('Content-Range')}")
            if len(response.content) != end - start + 1:
                raise RuntimeError("Incomplete range response")
            return response.content, int(response.headers["Content-Range"].split("/")[-1])
        except (requests.RequestException, RuntimeError) as exc:
            last = exc
            if attempt < 3:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Range download failed: {last}")


class RangeReader(io.RawIOBase):
    def __init__(self):
        _, self.size = get_range(0, 0)
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.position + n, self.size)
        if end <= self.position:
            return b""
        data, _ = get_range(self.position, end - 1)
        self.position = end
        return data


def extract_group(group, root, total_size):
    start = group[0]["header_offset"]
    end = min(total_size, max(p["header_offset"] + p["compress_size"] + 2048 for p in group)) - 1
    data, _ = get_range(start, end)
    for entry in group:
        relative = PurePosixPath(entry["filename"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "multimodal_dataset":
            raise ValueError("Unsafe archive member path")
        target = root.joinpath(*relative.parts)
        offset = entry["header_offset"] - start
        header = struct.unpack_from("<4s5H3I2H", data, offset)
        if header[0] != b"PK\x03\x04":
            raise ValueError("Invalid local ZIP header")
        local_name_len, local_extra_len = header[-2:]
        begin = offset + 30 + local_name_len + local_extra_len
        packed = data[begin:begin + entry["compress_size"]]
        if entry["compress_type"] == zipfile.ZIP_DEFLATED:
            content = zlib.decompress(packed, -15)
        elif entry["compress_type"] == zipfile.ZIP_STORED:
            content = packed
        else:
            raise ValueError(f"Unsupported ZIP compression {entry['compress_type']}")
        if len(content) != entry["file_size"] or zlib.crc32(content) != entry["CRC"]:
            raise ValueError(f"ZIP integrity failure: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        temporary.write_bytes(content)
        temporary.replace(target)
    return len(group), len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "mcubes_raw")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--include-rgb", action="store_true", help="Also download polL_color for RGB detector training")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1 through 8")
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "archive_index.json"
    if index_path.exists():
        saved = json.loads(index_path.read_text(encoding="utf-8"))
        entries, total_size = saved["entries"], saved["total_size"]
    else:
        reader = RangeReader()
        with zipfile.ZipFile(reader) as archive:
            entries = [{"filename": x.filename, "header_offset": x.header_offset,
                        "compress_size": x.compress_size, "file_size": x.file_size,
                        "compress_type": x.compress_type, "CRC": x.CRC}
                       for x in archive.infolist() if not x.is_dir()]
        total_size = reader.size
        index_path.write_text(json.dumps({"source": SOURCE, "url": URL, "total_size": total_size,
                                          "entries": entries}, indent=2), encoding="utf-8")
    categories = ALLOWED | ({"polL_color"} if args.include_rgb else set())
    selected = [e for e in entries if len(PurePosixPath(e["filename"]).parts) >= 3
                and PurePosixPath(e["filename"]).parts[1] in categories]
    print("Selected archive contents:", flush=True)
    for category in sorted(categories):
        subset = [e for e in selected if PurePosixPath(e["filename"]).parts[1] == category]
        print(f"  {category}: {len(subset)} files; {sum(e['file_size'] for e in subset)/1024**2:.1f} MB", flush=True)
    print(f"Selected compressed data: {sum(e['compress_size'] for e in selected)/1024**2:.1f} MB "
          f"from {total_size/1024**3:.2f} GB full archive", flush=True)
    if args.index_only:
        return
    pending = []
    for entry in selected:
        target = args.output.joinpath(*PurePosixPath(entry["filename"]).parts)
        if target.is_file() and target.stat().st_size == entry["file_size"]:
            if zlib.crc32(target.read_bytes()) == entry["CRC"]:
                continue
        pending.append(entry)
    pending.sort(key=lambda x: x["header_offset"])
    groups = []
    for entry in pending:
        if (not groups or entry["header_offset"] - groups[-1][-1]["header_offset"]
                - groups[-1][-1]["compress_size"] > 65536
                or entry["header_offset"] - groups[-1][0]["header_offset"] > 16 * 1024**2):
            groups.append([])
        groups[-1].append(entry)
    completed, transferred = len(selected) - len(pending), 0
    print(f"Resuming: {completed}/{len(selected)} files verified; {len(groups)} range batches remain", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_group, group, args.output, total_size) for group in groups]
        for future in as_completed(futures):
            count, nbytes = future.result()
            completed += count
            transferred += nbytes
            print(f"Downloaded/verified {completed}/{len(selected)} files; "
                  f"transferred {transferred/1024**2:.1f} MB", flush=True)
    print(f"Ready: {(args.output / 'multimodal_dataset').resolve()}", flush=True)


if __name__ == "__main__":
    main()
