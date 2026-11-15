"""Assemble the private, relocatable Windows x64 package from verified inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageDraw


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


NON_PAYLOAD_SUFFIXES = {'.dcm', '.dicom', '.ima', '.j2stl', '.stl', '.pyc', '.pyo'}


def ignore_bytecode(_folder, items):
    excluded = {'__pycache__'}
    folder = Path(_folder)
    if (folder.parent.name, folder.name) == ('pydicom', 'data'):
        excluded.add('test_files')
    return {name for name in items if name in excluded or Path(name).suffix.lower() in NON_PAYLOAD_SUFFIXES}


def copy_filtered(source, destination, skip):
    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        if skip(child.name):
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, ignore=ignore_bytecode)
        else:
            shutil.copy2(child, target)


def app_icon(out):
    # Two separate surfaces and the joint space between them are readable at 16px.
    icon = Image.new('RGBA', (256, 256))
    d = ImageDraw.Draw(icon)
    d.rounded_rectangle((4, 4, 251, 251), 53, fill='#10243a')
    d.rounded_rectangle((12, 12, 243, 243), 46, outline='#21405b', width=4)
    d.polygon([(69, 63), (178, 53), (204, 82), (165, 111), (74, 119), (47, 93)], fill='#7ed9c7')
    d.polygon([(81, 149), (170, 128), (206, 154), (181, 193), (79, 207), (51, 178)], fill='#b6d9f0')
    icon.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base-python', type=Path, required=True)
    a = parser.parse_args()
    source, work = a.source.resolve(), a.build.resolve()
    site = Path(sys.prefix) / 'Lib/site-packages'
    root = work / 'payload'
    root.mkdir()
    app = root / 'app'
    app.mkdir()
    runtime = root / 'runtime'
    runtime.mkdir()
    base_python = a.base_python.resolve()
    for name in ('python.exe', 'pythonw.exe', 'python312.dll', 'python3.dll',
                 'vcruntime140.dll', 'vcruntime140_1.dll', 'LICENSE.txt'):
        shutil.copy2(base_python / name, runtime / name)
    shutil.copytree(base_python / 'DLLs', runtime / 'DLLs', ignore=ignore_bytecode)
    skip_stdlib = {'site-packages', 'test', 'idlelib', 'turtledemo', '__pycache__', 'ensurepip'}
    copy_filtered(base_python / 'Lib', runtime / 'Lib', lambda name: name in skip_stdlib)
    for extra in ('include', 'libs', 'tcl', 'licenses'):
        if (base_python / extra).is_dir():
            shutil.copytree(base_python / extra, runtime / extra, ignore=ignore_bytecode)
    stdlib_zip = base_python / 'python312.zip'
    if stdlib_zip.is_file():
        shutil.copy2(stdlib_zip, runtime / stdlib_zip.name)
    # Use a Windows path override to ignore the system registry, PYTHONPATH,
    # site-user packages and the build venv while retaining the bundled wheels.
    (runtime / 'python312._pth').write_text(
        'python312.zip\n.\nDLLs\nLib\nLib\\site-packages\n..\\app\nimport site\n', encoding='ascii')

    src_copy = work / 'wheel-source'
    src_copy.mkdir()
    for p in ('pyproject.toml', 'README.md'):
        shutil.copy2(source / p, src_copy / p)
    shutil.copytree(source / 'joint2stl', src_copy / 'joint2stl', ignore=ignore_bytecode)
    wheels = work / 'wheels'
    wheels.mkdir()
    subprocess.run([sys.executable, '-m', 'pip', '--disable-pip-version-check', 'wheel', '--no-deps',
                    '--no-build-isolation', '--no-index', '-w', str(wheels), str(src_copy)], check=True)
    wheel, = wheels.glob('joint2stl-0.4.4-*.whl')
    with ZipFile(wheel) as archive:
        for name in archive.namelist():
            rel = PurePosixPath(name)
            assert '..' not in rel.parts and not rel.is_absolute() and rel.parts[0] in ('joint2stl', 'joint2stl-0.4.4.dist-info')
        archive.extractall(app)
    assert digest(app / 'joint2stl/app.py') == digest(source / 'joint2stl/app.py')
    shutil.copytree(source / 'models', app / 'models', ignore=ignore_bytecode)
    weights = [json.loads((app / 'models' / f).read_text(encoding='utf-8'))
               for f in ('model_manifest.json', 'mri_model_manifest.json', 'correction_model_manifest.json')]
    fixed_weights = {'lite_medsam.pth': '79d8c9dca6db4d69d3f905579e5250af05e859fff9c1f543e89a513c3028ce76',
                     'bone_sam.pth': '43209385a63290f7a10bb80eb1bbb47186abd64c73f507b3fcae6d11542e1d49',
                     'scribbleprompt_unet.pth': '43f57ee8fa8ec529c31be281e06749f9e629b30157bbbcc9baf200cddec1acbe'}
    for name, expected in fixed_weights.items():
        assert digest(app / 'models' / name) == expected, name

    tests_and_tools = ('__editable__', '_pytest', 'pytest', 'iniconfig', 'pluggy', 'pip-', 'joint2stl-')
    def skip_site(name):
        return name in {'pip', 'py.py', '__pycache__'} or name.endswith(('.whl', '.pyc')) or name.startswith(tests_and_tools)

    print('Staging frozen CPU/CUDA and GUI wheels', flush=True)
    copy_filtered(site, runtime / 'Lib/site-packages', skip_site)
    for p in (runtime / 'Lib/site-packages').glob('*.dist-info/direct_url.json'):
        p.unlink()  # no build-host path or editable environment metadata
    for name in ('start-joint2stl.pyw', 'THIRD-PARTY-NOTICES.txt'):
        shutil.copyfile(work / name, root / name)
    (root / '安装说明.txt').write_text((work / '安装说明.txt').read_text(encoding='utf-8'), encoding='utf-8-sig')
    shutil.copyfile(work / 'sitecustomize.py', app / 'sitecustomize.py')
    app_icon(root / 'Joint2STL.ico')
    actual_files = [f for f in root.rglob('*') if f.is_file()]
    prohibited = {'.dcm', '.dicom', '.ima', '.j2stl', '.stl'}
    assert not any(f.suffix.lower() in prohibited for f in actual_files)
    assert set((app / 'models').glob('*.pth')) == {app / 'models' / f for f in fixed_weights}
    result = {'python_from': str(base_python), 'app_sha256': digest(app / 'joint2stl/app.py'),
              'app_wheel_sha256': digest(wheel), 'files': len(actual_files),
              'uncompressed_bytes': sum(f.stat().st_size for f in actual_files),
              'models': weights, 'no_imaging_projects_stls_or_dicoms': True,
              'runtime_distributions': sorted(f.name for f in (runtime / 'Lib/site-packages').glob('*.dist-info'))}
    (work / 'payload.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Staged', result['files'], 'files,', result['uncompressed_bytes'], 'bytes', flush=True)


if __name__ == '__main__':
    main()
