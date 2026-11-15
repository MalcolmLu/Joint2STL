"""Build an isolated source ZIP, with an explicit option for verified weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = ('.gitattributes', '.gitignore', 'README.md', 'pyproject.toml', 'Start_Windows.cmd',
             'Install_NVIDIA_Windows.cmd', '启动 Joint2STL.command', 'Install_AI_Mac.command')
MODELS = ('model_manifest.json', 'mri_model_manifest.json', 'correction_model_manifest.json')
PACKAGING = ('Joint2STL.iss', 'README.md', 'stage_installer.py', 'sitecustomize.py',
             'start-joint2stl.pyw', '安装说明.txt', 'THIRD-PARTY-NOTICES.txt')
WEIGHTS = {
    'lite_medsam.pth': '79d8c9dca6db4d69d3f905579e5250af05e859fff9c1f543e89a513c3028ce76',
    'bone_sam.pth': '43209385a63290f7a10bb80eb1bbb47186abd64c73f507b3fcae6d11542e1d49',
    'scribbleprompt_unet.pth': '43f57ee8fa8ec529c31be281e06749f9e629b30157bbbcc9baf200cddec1acbe',
}


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inputs(include_models):
    files = [ROOT / name for name in TOP_LEVEL]
    files += [ROOT / 'docs/PROJECT_REFERENCES.md']
    files += [ROOT / 'packaging/windows' / name for name in PACKAGING]
    files += [ROOT / 'models' / name for name in MODELS]
    for folder in ('joint2stl', 'tests', 'scripts'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and not p.is_symlink()
                     and (p.suffix in ('.py', '.txt', '.json') or p.name == 'LICENSE'))
    if include_models:
        for name, expected in WEIGHTS.items():
            p = ROOT / 'models' / name
            if not p.is_file() or sha256(p) != expected:
                raise ValueError(f'Model {name} missing or wrong checksum. Run its setup script first.')
            files.append(p)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-models', action='store_true',
                        help='Bundle the three checked weights for permitted local/offline use.')
    args = parser.parse_args()
    version = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    destination = ROOT / 'dist'
    files = inputs(args.include_models)
    for p in files:
        if not p.is_file() or not p.resolve().is_relative_to(ROOT):
            raise ValueError('Unexpected source bundle input: ' + str(p.relative_to(ROOT)))
    destination.mkdir(exist_ok=True)
    stem = f'Joint2STL-{version}-source' + ('-with-models' if args.include_models else '')
    archive = destination / (stem + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f'Joint2STL-{version}/' + p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist()) != len(files):
            raise OSError('Source ZIP failed read-back validation.')
    digest = sha256(archive)
    (destination / (stem + '-SHA256SUMS.txt')).write_text(f'{digest}  {archive.name}\n', encoding='ascii')
    print(json.dumps({'archive': str(archive), 'files': len(files), 'bytes': archive.stat().st_size,
                      'sha256': digest, 'bundled_weights': args.include_models}))


if __name__ == '__main__':
    main()
