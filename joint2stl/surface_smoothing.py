"""Surface smoothing with limits on movement, signed volume and face folding."""
from __future__ import annotations

import numpy as np
import trimesh

PROFILES = {'off': None, 'light': (10, .2, .15), 'standard': (20, .1, .3), 'strong': (30, .05, .5)}


def smooth_surface(mesh, profile='standard'):
    if profile not in PROFILES:
        raise ValueError('未知表面平滑程度。')
    settings = PROFILES[profile]
    if settings is None:
        mesh.metadata['surface_smoothing'] = {'profile': 'off', 'max_displacement_mm': 0.0, 'volume_change_fraction': 0.0}
        return mesh
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

    iterations, passband, limit_mm = settings
    vertices = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    points = vtkPoints(); points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = vtkCellArray()
    offsets = np.arange(len(faces) + 1, dtype=np.int64) * 3
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(faces.ravel(), deep=True))
    poly = vtkPolyData(); poly.SetPoints(points); poly.SetPolys(cells)
    smoothing = vtkWindowedSincPolyDataFilter(); smoothing.SetInputData(poly)
    smoothing.SetNumberOfIterations(iterations); smoothing.SetPassBand(passband)
    smoothing.BoundarySmoothingOff(); smoothing.FeatureEdgeSmoothingOff()
    smoothing.NonManifoldSmoothingOff(); smoothing.NormalizeCoordinatesOn(); smoothing.Update()
    proposed = vtk_to_numpy(smoothing.GetOutput().GetPoints().GetData()).astype(np.float64, copy=True)
    if proposed.shape != vertices.shape or not np.isfinite(proposed).all():
        raise ValueError('表面平滑未生成有效坐标，已保留原分割。')
    displacement = proposed - vertices
    distances = np.linalg.norm(displacement, axis=1)
    # A single sharp tip must not throttle smoothing of the whole surface.
    # Limit each vertex locally, then apply the shared topology/volume checks.
    scales=np.minimum(1.0,limit_mm/np.maximum(distances,1e-12))
    displacement*=scales[:,None]
    blend = 1.0
    original_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                 vertices[faces[:, 2]] - vertices[faces[:, 0]])
    original_volume = float(mesh.volume)
    # Reduce the blend until every face keeps its orientation and the signed
    # volume (including negative inner cavity shells) changes by at most1%.
    candidate = None
    for _ in range(16):
        positions = vertices + blend * displacement
        normals = np.cross(positions[faces[:, 1]] - positions[faces[:, 0]],
                            positions[faces[:, 2]] - positions[faces[:, 0]])
        valid_faces = np.all(np.einsum('ij,ij->i', original_normals, normals) > 0)
        candidate = trimesh.Trimesh(vertices=positions, faces=faces, process=False)
        delta = (float(candidate.volume) - original_volume) / max(abs(original_volume), 1e-9)
        if valid_faces and abs(delta) <= .01:
            break
        blend *= .5
    else:
        positions = vertices.copy(); blend = 0.0; delta = 0.0
        candidate = trimesh.Trimesh(vertices=positions, faces=faces, process=False)
    movement = np.linalg.norm(positions - vertices, axis=1)
    candidate.metadata.update(mesh.metadata)
    candidate.metadata['surface_smoothing'] = {
        'method': 'vtk_windowed_sinc', 'profile': profile, 'iterations': iterations,
        'displacement_limit_mode':'per_vertex',
        'displacement_limit_mm': limit_mm, 'max_displacement_mm': float(movement.max()),
        'p95_displacement_mm': float(np.percentile(movement, 95)),
        'volume_change_fraction': delta, 'blend': blend,
        'faces_preserved': True, 'face_orientation_preserved': True,
    }
    return candidate
