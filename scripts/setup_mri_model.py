"""Install exactly the pinned official SegmentAnyBone checkpoint, without login.

No dependencies, other checkpoints, source archives, or clinical data are sent
or downloaded. Existing valid weights are verified and reused unchanged.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


CHECKPOINT_URL = ('https://raw.githubusercontent.com/mazurowski-lab/SlicerSegmentHumanBody/'
                  '5f026279291d2980a01274a0d0aaa6c1a4d68960/SegmentHumanBody/models/bone_sam.pth')
CHECKPOINT_BYTES = 46_221_025
CHECKPOINT_SHA256 = '43209385a63290f7a10bb80eb1bbb47186abd64c73f507b3fcae6d11542e1d49'


def _digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def install(destination=None):
    destination = Path(destination) if destination is not None else Path(__file__).resolve().parents[1] / 'models' / 'bone_sam.pth'
    if destination.is_file():
        if destination.stat().st_size != CHECKPOINT_BYTES or _digest(destination) != CHECKPOINT_SHA256:
            raise ValueError('Existing bone_sam.pth fails pinned size/SHA-256 verification; it was not overwritten.')
        print('SegmentAnyBone checkpoint verified; no download needed.')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.bone_sam-', suffix='.part', dir=destination.parent)
    temporary = Path(name)
    try:
        digest, total = hashlib.sha256(), 0
        with os.fdopen(descriptor, 'wb') as stream:
            with urllib.request.urlopen(CHECKPOINT_URL, timeout=30) as response:
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != CHECKPOINT_BYTES:
                    raise ValueError('Unexpected declared SegmentAnyBone byte length.')
                while True:
                    chunk = response.read(min(1024 * 1024, CHECKPOINT_BYTES - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > CHECKPOINT_BYTES:
                        raise ValueError('SegmentAnyBone download exceeds the pinned byte limit.')
                    stream.write(chunk)
                    digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if total != CHECKPOINT_BYTES or digest.hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('SegmentAnyBone download failed pinned byte/SHA-256 verification.')
        # Another installer must not have its file silently replaced mid-download.
        if destination.exists():
            if destination.stat().st_size != CHECKPOINT_BYTES or _digest(destination) != CHECKPOINT_SHA256:
                raise ValueError('Destination changed during download; existing file preserved.')
        else:
            os.replace(temporary, destination)
        print('Official SegmentAnyBone checkpoint installed and verified (CC BY-NC 4.0).')
        return destination
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    install()
