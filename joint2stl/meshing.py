from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
import trimesh

from .model import Session, Volume
from .compute import gaussian_filter, last_backend
from . import __version__
from .surface_smoothing import smooth_surface


def clean_oriented_surface(vertices,faces):
    """Retain nested cavity orientation inherited from the scalar isosurface."""
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True,validate=False)
    mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    # Orient the complete boundary, not each shell independently: an inner
    # cavity correctly has negative signed volume relative to its outer shell.
    mesh.fix_normals(multibody=False)
    return mesh


def mesh_from_mask(mask, volume: Volume, offset_zyx=(0, 0, 0), sigma_mm=0.35, step=1, backend='auto', surface_profile='off', preserve_detail=True):
    if not np.any(mask):
        raise ValueError('空骨标签没有可导出的表面。')
    if not np.isfinite(sigma_mm) or not 0 <= sigma_mm <= 3:
        raise ValueError('表面平滑参数无效。')
    requested_sigma = float(sigma_mm)
    sigma_limit = float(min(volume.spacing_zyx)*.5)
    sigma_mm = min(requested_sigma,sigma_limit) if preserve_detail else requested_sigma
    pad = max(3, int(np.ceil(sigma_mm / min(volume.spacing_zyx) * 4)))
    field = np.pad(mask, pad).astype(np.float32)
    if sigma_mm:
        field = gaussian_filter(field, sigma_mm / volume.spacing_zyx, backend)
    if field.max() <= 0.5:
        raise ValueError('骨块过小或平滑过强，请降低平滑强度。')
    vertices, faces, _, _ = marching_cubes(field, 0.5, step_size=step, allow_degenerate=False)
    global_zyx = vertices - pad + np.asarray(offset_zyx)
    world = volume.world(global_zyx)
    mesh = clean_oriented_surface(world,faces)
    mesh.metadata['surface_filter_backend']=last_backend() if sigma_mm else 'none'
    mesh.metadata['voxel_filter']={'requested_sigma_mm':requested_sigma,'applied_sigma_mm':sigma_mm,
                                  'preserve_detail':bool(preserve_detail),'sigma_limit_mm':sigma_limit}
    return smooth_surface(mesh, surface_profile)


def segment_meshes(session: Session, ids=None, preview=False, progress=lambda *_: None, *, include_hidden=False):
    from .mr_policy import require_mri_modeling_source
    require_mri_modeling_source(session)
    selected = [s for s in session.segments if (s.visible or include_hidden) and (ids is None or s.id in ids)]
    if not selected:
        raise ValueError('先勾选需要显示或导出的骨标签。')
    # Preview and export share native-resolution surfaces. Build bounding boxes
    # lazily; an unchanged scene can be exported directly from cached geometry.
    objects = None
    output = []
    for i, segment in enumerate(selected):
        key=(session.revision,segment.id,float(session.parameters['surface_mm']),session.parameters.get('compute_backend','auto'),session.parameters.get('mesh_smoothing','standard'),session.parameters.get('preserve_detail',True))
        cached=session.cached_mesh(key)
        if cached is not None:
            mesh,box=cached;output.append((segment,mesh,box))
            progress(int(i/len(selected)*90),f'复用精细表面：{segment.name}')
            continue
        if objects is None:objects=ndi.find_objects(session.labels)
        if segment.id > len(objects) or objects[segment.id - 1] is None:
            continue
        box = objects[segment.id - 1]
        mask = session.labels[box] == segment.id
        progress(int(i / len(selected) * 90), f'构建表面：{segment.name}')
        mesh = mesh_from_mask(mask, session.volume, [s.start for s in box],
                              session.parameters['surface_mm'], step=1,
                              backend=session.parameters.get('compute_backend','auto'),
                              surface_profile=session.parameters.get('mesh_smoothing','standard'),
                              preserve_detail=session.parameters.get('preserve_detail',True))
        session.remember_mesh(key,mesh,box)
        output.append((segment, mesh, box))
    if not output:
        raise ValueError('勾选的骨标签均为空。')
    progress(100, '表面已构建')
    return output


def export_stl(session: Session, parent: str | Path, progress=lambda *_: None, *, ids=None, include_hidden=False,
               mode='separate', combined_name='选定骨骼_组合.stl', selection_scope='checked'):
    if mode not in ('separate','combined','both'):
        raise ValueError('STL 导出方式无效。')
    from .mr_policy import modeling_source_metadata
    modeling_source=modeling_source_metadata(session)
    if modeling_source and session.parameters.get('mr_sagittal_only',True) and modeling_source['prior_offplane_model_report_indices']:
        raise ValueError('MRI 项目含非矢状平面的旧模型标注，请从矢状采集重新建立复核掩膜后导出；原项目已保留。')
    meshes = segment_meshes(session, ids=ids, preview=False, progress=progress,include_hidden=include_hidden)
    separate = mode in ('separate','both')
    # A new batch directory prevents silent overwrites of any earlier export.
    base = Path(parent) / ('Joint2STL_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    destination = base
    number = 2
    while destination.exists():
        destination = base.with_name(base.name + f'_{number}')
        number += 1
    destination.mkdir(parents=True)
    manifest = {'version': __version__, 'units': 'mm', 'coordinates': 'DICOM patient LPS',
                'export_mode':mode,'selection_scope':selection_scope,
                'source_modality':session.volume.modality,'source_intensity_units':session.volume.intensity_units,
                'modeling_source':modeling_source,
                'structure_assessments':[{'id':s.id,'name':s.name,'kind':s.kind,'structure':s.structure,
                    'presence':s.presence,'presence_source':s.presence_source} for s in session.segments if s.structure or s.kind=='ligament'],
                'selected_label_ids':[segment.id for segment,_,_ in meshes],
                'source_reconstruction': session.volume.reconstruction,
                'joint': session.joint, 'side': session.side,
                'roi_zyx': [[s.start, s.stop] for s in session.roi],
                'surface_sigma_mm': session.parameters['surface_mm'],
                'mesh_smoothing': session.parameters.get('mesh_smoothing','standard'),
                'preserve_detail':session.parameters.get('preserve_detail',True),
                'preprocessing_operations': session.preprocess_reports,
                'repair_operations':session.repair_reports,
                'requested_compute_backend':session.parameters.get('compute_backend','auto'),
                'notes': ['STL has no encoded units; import as mm.',
                          'Zero padding closes scan/segmentation truncation boundaries.',
                          'Mesh closure does not establish anatomical accuracy.',
                          'Self-intersections have not been checked.',
                          'Combined STL retains individual bone positions/shells; no Boolean union.'],
                'segments': [], 'files': [], 'combined':None}
    prepared = []
    try:
        for i, (segment, mesh, box) in enumerate(meshes):
            progress(int(i / len(meshes) * 90), f'导出并回读：{segment.name}')
            name = re.sub(r'[^\w\-\u4e00-\u9fff]+', '_', segment.name).strip('_') or 'bone'
            file = destination / f'{segment.id:02d}_{name}.stl'
            # Binary STL stores float32 positions. Tiny triangles can collapse
            # after this conversion, creating zero-area or repeated faces.
            # Validate the actual representable geometry before writing it;
            # this avoids changing segmentation/smoothing to repair a format issue.
            quantized=mesh.vertices.astype(np.float32).astype(np.float64)
            quantization_error=float(np.max(np.abs(quantized-mesh.vertices)))
            stl_mesh=clean_oriented_surface(quantized,mesh.faces.copy())
            if separate:
                stl_mesh.export(file)
                loaded = trimesh.load_mesh(file, process=True,validate=False)
            else:loaded=stl_mesh
            q = {'id': segment.id, 'name': segment.name, 'filename': file.name if separate else None,
                 'kind':segment.kind,'structure':segment.structure,'presence':segment.presence,'presence_source':segment.presence_source,
                 'validation_basis':'written_individual_stl' if separate else 'prepared_geometry;see_combined_file',
                 'anatomical_name_user_confirmed': segment.reviewed,
                 'anatomical_name_source': 'manual' if segment.reviewed else (segment.recognition.get('name_source') or ('geometry' if segment.recognition.get('name') else 'unresolved')),
                 'anatomical_recognition': segment.recognition,
                 'faces': int(len(loaded.faces)), 'watertight': bool(loaded.is_watertight),
                 'winding_consistent': bool(loaded.is_winding_consistent),
                 'positive_volume': bool(loaded.volume > 0),
                 'surface_filter_backend':mesh.metadata.get('surface_filter_backend','unknown'),
                 'voxel_filter':mesh.metadata.get('voxel_filter',{}),
                 'surface_smoothing':mesh.metadata.get('surface_smoothing',{}),
                 'float32_quantization_max_coordinate_error_mm':quantization_error,
                 'format_collapsed_faces_removed':int(len(mesh.faces)-len(stl_mesh.faces)),
                 'bounds_lps_mm': loaded.bounds.tolist(),
                 'roundtrip_max_vertex_bound_error_mm': float(np.max(np.abs(mesh.bounds - loaded.bounds))) if separate else None,
                 'touches_volume_boundary': any(s.start == 0 or s.stop == n for s, n in zip(box, session.labels.shape)),
                 'touches_current_roi_boundary': any(s.start == r.start or s.stop == r.stop for s, r in zip(box, session.roi))}
            if session.volume.is_mr:
                indices=np.flatnonzero(np.any(session.labels[session.roi]==segment.id,axis=(1,2)))+session.roi[0].start+1
                q['annotation_coverage']={'native_slice_indices_1based':indices.tolist(),
                    'native_slice_count':len(indices),'scope':'current ROI','completeness':'not_assessed',
                    'note':'Mask coverage and mesh closure do not establish complete anatomical delineation.'}
            manifest['segments'].append(q)
            prepared.append((segment,stl_mesh))
            if separate:
                manifest['files'].append({k:q[k] for k in ['filename','faces','watertight','winding_consistent','positive_volume']})
        if mode in ('combined','both'):
            progress(94,'组合所选骨骼并回读 STL')
            name=re.sub(r'[^\w\-\u4e00-\u9fff]+','_',Path(combined_name).stem).strip('_') or 'selected_bones'
            file=destination/(name+'.stl')
            number=2
            while file.exists():
                file=destination/(name+f'_{number}.stl');number+=1
            assembled=trimesh.util.concatenate([mesh for _,mesh in prepared])
            assembled.export(file)
            loaded=trimesh.load_mesh(file,process=True,validate=False)
            ranges=[];start=0
            for segment,mesh in prepared:
                ranges.append({'id':segment.id,'name':segment.name,'first_triangle':start,'triangle_count':len(mesh.faces)})
                start+=len(mesh.faces)
            q={'filename':file.name,'faces':len(loaded.faces),'watertight':bool(loaded.is_watertight),
               'winding_consistent':bool(loaded.is_winding_consistent),'positive_volume':bool(loaded.volume>0),
               'zero_area_faces':int(np.count_nonzero(loaded.area_faces<=0)),
               'bounds_lps_mm':loaded.bounds.tolist(),'label_ids':[s.id for s,_ in prepared],
               'source_triangle_ranges':ranges,'individual_bones_closed':all(x['watertight'] for x in manifest['segments']),
               'preserves_relative_positions':True,'boolean_union_applied':False}
            manifest['combined']=q;manifest['files'].append(q)
        manifest['complete'] = True
    except Exception as exc:
        manifest['complete'] = False
        manifest['error'] = str(exc)
        raise
    finally:
        (destination / 'export_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2),encoding='utf-8')
    (destination / '导入说明.txt').write_text('坐标：DICOM 患者 LPS；数值单位：毫米（导入 STL 时选择 mm）。\n独立文件对应单个骨标签；组合文件包含本批次所选骨，保留相对位置和独立骨壳，未作布尔融合。\n具体标签、文件对应及网格检查见 export_manifest.json。\n范围边缘会封口，代表截断；网格检查不代表解剖精度。\n',encoding='utf-8')
    progress(100, 'STL 已按所选范围导出')
    return str(destination), manifest
