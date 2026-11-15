"""Download one pinned, public official checkpoint; no authentication or CT upload."""
from pathlib import Path
import hashlib,os,urllib.request

root=Path(__file__).resolve().parents[1];out=root/'models';out.mkdir(exist_ok=True)
path=out/'lite_medsam.pth'
sha='79d8c9dca6db4d69d3f905579e5250af05e859fff9c1f543e89a513c3028ce76'
if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==sha:
    print('LiteMedSAM checkpoint verified; no download needed.')
else:
    url='https://drive.usercontent.google.com/download?id=18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z&export=download&confirm=t'
    temp=out/'lite_medsam.pth.part'
    try:
        with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as f:
            total=0
            while True:
                data=response.read(1024*1024)
                if not data:break
                total+=len(data)
                if total>80*1024**2:raise ValueError('Unexpected model size')
                f.write(data)
        if hashlib.sha256(temp.read_bytes()).hexdigest()!=sha:raise ValueError('Checkpoint SHA256 mismatch')
        os.replace(temp,path)
        print('Official LiteMedSAM checkpoint downloaded and verified.')
    finally:
        if temp.exists():temp.unlink()
