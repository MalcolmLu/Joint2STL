"""Create a local launcher, not a self-contained or signed application bundle."""
from pathlib import Path
import plistlib
import shlex
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from joint2stl import __version__
bundle=root/'dist/Joint2STL.app'
binary=bundle/'Contents/MacOS/Joint2STL'
binary.parent.mkdir(parents=True,exist_ok=True)
resources=bundle/'Contents/Resources';resources.mkdir(exist_ok=True)
binary.write_text('#!/bin/zsh\nset -e\ncd '+shlex.quote(str(root))+'\nmkdir -p "$HOME/Library/Logs"\nexec .venv/bin/python -m joint2stl "$@" >> "$HOME/Library/Logs/Joint2STL.log" 2>&1\n')
binary.chmod(0o755)
info={'CFBundleExecutable':'Joint2STL','CFBundleIdentifier':'local.joint2stl.workbench',
      'CFBundleName':'Joint2STL','CFBundleDisplayName':'骨模 Joint2STL','CFBundlePackageType':'APPL',
      'CFBundleShortVersionString':__version__,'CFBundleVersion':__version__,'NSHighResolutionCapable':True,
      'LSMinimumSystemVersion':'12.0','CFBundleIconFile':'AppIcon.icns'}
with (bundle/'Contents/Info.plist').open('wb') as f:plistlib.dump(info,f)
try:
    from PIL import Image,ImageDraw
    image=Image.new('RGBA',(1024,1024),(0,0,0,0));draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((32,32,992,992),radius=215,fill='#132435')
    draw.rounded_rectangle((305,155,450,465),radius=70,fill='#71d8c1')
    draw.ellipse((230,360,520,600),fill='#71d8c1')
    draw.rounded_rectangle((575,570,720,875),radius=70,fill='#88bafa')
    draw.ellipse((505,440,795,675),fill='#88bafa')
    draw.line((485,555,565,510),fill='#132435',width=45)
    image.save(resources/'AppIcon.icns',format='ICNS')
except ImportError:
    info.pop('CFBundleIconFile',None)
    with (bundle/'Contents/Info.plist').open('wb') as f:plistlib.dump(info,f)
print(bundle)
