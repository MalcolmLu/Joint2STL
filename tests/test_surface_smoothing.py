import numpy as np
import pytest
import trimesh

from joint2stl.model import Session, Volume
from joint2stl.meshing import mesh_from_mask, segment_meshes, export_stl
from joint2stl.surface_smoothing import smooth_surface


def rough_sphere():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=25)
    rng = np.random.default_rng(2026)
    directions = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1)[:, None]
    mesh.vertices += directions * rng.normal(0, .16, (len(mesh.vertices), 1))
    return mesh


@pytest.mark.parametrize('profile,limit', [('light', .15), ('standard', .3), ('strong', .5)])
def test_smoothing_reduces_ripple_with_bounded_movement_volume_and_preserved_topology(profile, limit):
    source = rough_sphere(); original = source.vertices.copy()
    smooth = smooth_surface(source, profile)
    movement = np.linalg.norm(smooth.vertices - original, axis=1)
    assert 0 < movement.max() <= limit + 1e-9
    assert abs(smooth.volume / source.volume - 1) <= .01
    assert np.std(np.linalg.norm(smooth.vertices, axis=1)) < np.std(np.linalg.norm(original, axis=1)) * .9
    np.testing.assert_array_equal(smooth.faces, source.faces)
    np.testing.assert_array_equal(source.vertices, original)
    assert smooth.is_watertight and smooth.is_winding_consistent


def test_smoothing_keeps_negative_inner_cavity_and_input_masks():
    volume = Volume(np.zeros((36, 36, 36), np.int16), np.eye(4))
    mask = np.zeros(volume.hu.shape, bool);mask[3:33, 3:33, 3:33] = True;mask[11:25, 11:25, 11:25] = False
    original = mask.copy()
    source = mesh_from_mask(mask, volume, sigma_mm=0)
    smooth = smooth_surface(source, 'standard')
    assert any(part.volume < 0 for part in smooth.split(only_watertight=False))
    assert smooth.is_watertight and smooth.is_winding_consistent and smooth.volume > 0
    assert abs(smooth.volume / source.volume - 1) <= .01
    np.testing.assert_array_equal(mask, original)


def test_profile_change_invalidates_surface_cache_and_export_uses_smoothed_preview(tmp_path):
    volume = Volume(np.zeros((40, 40, 40), np.int16), np.eye(4))
    zz, yy, xx = np.ogrid[:40, :40, :40]
    mask = (zz-20)**2 + (yy-20)**2 + (xx-20)**2 < 15**2
    session = Session(volume); bone = session.add_segment('股骨')
    session.apply(volume.full_box, mask.astype(np.uint16) * bone.id, 'initial')
    session.parameters['mesh_smoothing'] = 'off'
    unfiltered = segment_meshes(session)[0][1]
    session.parameters['mesh_smoothing'] = 'standard'
    preview = segment_meshes(session, preview=True)[0][1]
    assert preview is not unfiltered
    assert np.max(np.abs(preview.vertices-unfiltered.vertices)) > 0
    assert segment_meshes(session, preview=True)[0][1] is preview
    directory, manifest = export_stl(session, tmp_path)
    item = manifest['segments'][0]
    assert item['surface_smoothing'] == preview.metadata['surface_smoothing']
    assert item['watertight'] and item['winding_consistent'] and item['positive_volume']
    np.testing.assert_allclose(item['bounds_lps_mm'], preview.bounds, atol=2e-5)
