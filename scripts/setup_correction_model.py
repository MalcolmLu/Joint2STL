"""Verify/install one pinned public ScribblePrompt model. No clinical uploads."""
from pathlib import Path
import hashlib
import os
import tempfile
import urllib.request

URL=('https://huggingface.co/spaces/halleewong/ScribblePrompt/resolve/'
     '706d6559618e5f1104565a6a19c9ea069fca7454/checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt')
BYTES=15_977_486
SHA256='43f57ee8fa8ec529c31be281e06749f9e629b30157bbbcc9baf200cddec1acbe'


def valid(path):
    return path.is_file() and path.stat().st_size==BYTES and hashlib.sha256(path.read_bytes()).hexdigest()==SHA256


def install(destination=None):
    destination=Path(destination) if destination is not None else Path(__file__).resolve().parents[1]/'models/scribbleprompt_unet.pth'
    if destination.exists():
        if not valid(destination):raise ValueError('Existing correction checkpoint fails size/SHA verification; file preserved.')
        print('ScribblePrompt checkpoint verified; no download needed.');return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix='joint2stl-correction-',suffix='.part',dir=destination.parent)
    temporary=Path(name)
    try:
        count=0;digest=hashlib.sha256()
        with os.fdopen(descriptor,'wb') as stream,urllib.request.urlopen(URL,timeout=30) as response:
            while True:
                block=response.read(min(1024**2,BYTES-count+1))
                if not block:break
                count+=len(block)
                if count>BYTES:raise ValueError('Correction checkpoint exceeds pinned byte size.')
                stream.write(block);digest.update(block)
            stream.flush();os.fsync(stream.fileno())
        if count!=BYTES or digest.hexdigest()!=SHA256:raise ValueError('Correction model download failed pinned byte/SHA verification.')
        if destination.exists():
            if not valid(destination):raise ValueError('Correction destination changed during download; existing file preserved.')
        else:os.replace(temporary,destination)
        print('Official ScribblePrompt model installed and verified (Apache-2.0).');return destination
    finally:temporary.unlink(missing_ok=True)


if __name__=='__main__':install()
