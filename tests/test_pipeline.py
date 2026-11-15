"""Small synthetic fixtures test geometry and editing contracts, not clinical accuracy."""
from pathlib import Path
import json

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import pytest
import trimesh

from joint2stl.dicom_io import CT_UID, scan_directory, load_series
from joint2stl.model import Seed, Session, Volume, count_labels, load_project, save_project
from joint2stl.segmentation import (initial_segmentation, install_initial, paint,
    assign_component, split_from_seeds, clean_segment)
from joint2stl.meshing import mesh_from_mask, segment_meshes, export_stl


def volume(shape=(24,32,40)):
    affine=np.diag([0.6,0.9,1.2,1.]);affine[:3,3]=[-30,17,500]
    return Volume(np.zeros(shape,dtype=np.int16),affine,'test fixture')


def write_ct(path,uid,study,z,*,slope=1,intercept=-1024,orientation=(1,0,0,0,1,0),image_type=('ORIGINAL','PRIMARY','AXIAL'),kernel=None,description='fixture',manufacturer='',thickness=1):
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=CT_UID;meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID=CT_UID;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID=uid;ds.StudyInstanceUID=study;ds.SeriesNumber=3;ds.SeriesDescription=description
    if kernel is not None:ds.ConvolutionKernel=kernel
    if manufacturer:ds.Manufacturer=manufacturer
    ds.Modality='CT';ds.ImageType=list(image_type);ds.Rows=16;ds.Columns=20
    ds.PixelSpacing=[0.6,0.8];ds.ImageOrientationPatient=list(orientation);ds.ImagePositionPatient=[-8,-4,z]
    ds.SliceThickness=thickness;ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2'
    ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0
    ds.RescaleSlope=slope;ds.RescaleIntercept=intercept
    ds.PixelData=np.full((16,20),101,dtype=np.uint16).tobytes()
    ds.save_as(path,enforce_file_format=True)


def test_dicom_uses_positions_not_filenames_and_applies_per_slice_rescale(tmp_path):
    uid,study=generate_uid(),generate_uid()
    write_ct(tmp_path/'c.dcm',uid,study,10)
    write_ct(tmp_path/'a.dcm',uid,study,11,slope=0.5,intercept=0)
    write_ct(tmp_path/'b.dcm',uid,study,12)
    series,errors=scan_directory(tmp_path)
    assert errors==0 and len(series)==1 and series[0].usable
    vol=load_series(series[0])
    assert vol.hu.dtype==np.float32
    assert list(vol.hu[:,0,0])==[-923,50.5,-923]
    np.testing.assert_allclose(vol.world([2,1,1]),[-7.2,-3.4,12])


@pytest.mark.parametrize('positions',[[0,1,3],[0,1,1]])
def test_dicom_rejects_missing_or_duplicate_layers(tmp_path,positions):
    uid,study=generate_uid(),generate_uid()
    for i,z in enumerate(positions):write_ct(tmp_path/f'{i}.dcm',uid,study,z)
    series,_=scan_directory(tmp_path)
    assert not series[0].usable


def test_dicom_separates_scout_from_source(tmp_path):
    study=generate_uid();uid=generate_uid()
    for z in range(3):write_ct(tmp_path/f'{z}.dcm',uid,study,z)
    write_ct(tmp_path/'scout.dcm',generate_uid(),study,0,image_type=('ORIGINAL','PRIMARY','LOCALIZER'))
    series,_=scan_directory(tmp_path)
    assert len(series)==2 and sum(s.usable for s in series)==1


def test_recommendation_prefers_soft_thin_even_when_bone_is_thinner(tmp_path):
    from joint2stl.dicom_io import recommended_series
    study=generate_uid()
    for prefix,kernel,dz,thickness in [('bone','B_VSHARP_C',.4,.4),('soft','B_SOFT_C',1,1)]:
        uid=generate_uid()
        for z in range(3):write_ct(tmp_path/f'{prefix}_{z}.dcm',uid,study,z*dz,kernel=kernel,description=prefix,thickness=thickness)
    series,_=scan_directory(tmp_path);preferred=recommended_series(series)
    assert preferred is series[0] and preferred.kernel=='B_SOFT_C' and preferred.recommended
    assert not next(s for s in series if s.kernel=='B_VSHARP_C').recommended
    volume=load_series(preferred)
    assert volume.reconstruction['kernel']=='B_SOFT_C' and volume.reconstruction['kind']=='soft'
    session=Session(volume);path=tmp_path/'source.j2stl';save_project(session,path)
    assert load_project(path).volume.reconstruction==volume.reconstruction


@pytest.mark.parametrize('kernel,manufacturer,description,kind',[
    ('BONE','GE','soft tissue','sharp'),('STANDARD','GE','thin','soft'),
    (['Br40s','3'],'SIEMENS','Extremity','soft'),(['Br59s','3'],'SIEMENS','Extremity','sharp'),
    ('B','Philips','iDose (5)','soft'),('B_SOFT_C','UIH','1.0 x 1.0','soft'),
    ('B_VSHARP_C','UIH','1.0 x 1.0','sharp'),('CUSTOM','Other','Axial','unknown'),
    ('STANDARD','GE','Recon 3: 5mm bone','sharp')])
def test_reconstruction_family_comes_from_kernel_and_description(tmp_path,kernel,manufacturer,description,kind):
    uid,study=generate_uid(),generate_uid()
    for z in range(3):write_ct(tmp_path/f'{z}.dcm',uid,study,z,kernel=kernel,manufacturer=manufacturer,description=description)
    series,_=scan_directory(tmp_path)
    assert series[0].reconstruction_kind==kind
    assert series[0].recommended==(kind=='soft')


def test_overlapping_thick_soft_slices_are_not_automatically_called_thin(tmp_path):
    from joint2stl.dicom_io import recommended_series
    uid,study=generate_uid(),generate_uid()
    for z in range(3):write_ct(tmp_path/f'{z}.dcm',uid,study,z*.5,kernel='STANDARD',thickness=5)
    series,_=scan_directory(tmp_path)
    assert series[0].usable and series[0].effective_section_mm==5
    assert recommended_series(series) is None


def test_mixed_kernel_series_requires_manual_choice(tmp_path):
    uid,study=generate_uid(),generate_uid()
    for z in range(3):write_ct(tmp_path/f'{z}.dcm',uid,study,z,kernel='BONE' if z==1 else 'STANDARD')
    series,_=scan_directory(tmp_path)
    assert series[0].usable and series[0].reconstruction_kind=='mixed' and not series[0].recommended


def test_dicom_padding_is_not_mistaken_for_dense_bone(tmp_path):
    uid,study=generate_uid(),generate_uid()
    for z in range(3):
        path=tmp_path/f'{z}.dcm';write_ct(path,uid,study,z)
        ds=pydicom.dcmread(path);pixels=np.full((16,20),101,np.uint16);pixels[0,0]=4095
        ds.PixelData=pixels.tobytes();ds.add_new((0x0028,0x0120),'US',4095);ds.save_as(path,enforce_file_format=True)
    series,_=scan_directory(tmp_path);v=load_series(series[0])
    assert v.hu[0,0,0]==-4096 and v.hu[0,0,1]==-923


def test_dicom_rejects_non_hu_rescale_units(tmp_path):
    uid,study=generate_uid(),generate_uid()
    for z in range(3):
        path=tmp_path/f'{z}.dcm';write_ct(path,uid,study,z)
        ds=pydicom.dcmread(path);ds.RescaleType='US';ds.save_as(path,enforce_file_format=True)
    series,_=scan_directory(tmp_path)
    assert not series[0].usable and 'HU' in series[0].reason


def test_initial_segmentation_crops_filters_and_is_undoable():
    v=volume();v.hu.fill(-1000)
    v.hu[5:15,8:20,8:18]=600;v.hu[7,25,25]=600
    v.hu[5:15,8:20,30:38]=600  # outside ROI
    s=Session(v);s.set_roi((slice(0,24),slice(0,30),slice(0,25)))
    result=initial_segmentation(v,s.roi,min_mm3=20)
    assert result[1]==1
    install_initial(s,result)
    assert s.labels[10,10,12]>0 and s.labels[10,10,32]==0 and s.labels[7,25,25]==0
    saved=s.labels.copy();s.history.travel(s.labels)
    assert not s.labels.any()
    s.history.travel(s.labels,redo=True);np.testing.assert_array_equal(s.labels,saved)


def test_brush_respects_physical_radius_roi_and_other_labels():
    s=Session(volume());a=s.add_segment().id;b=s.add_segment().id
    s.set_roi((slice(4,18),slice(5,25),slice(5,30)))
    s.labels[10,10,12]=b
    paint(s,'axial',(10,10,10),1.3,a)
    assert s.labels[10,10,12]==a and s.labels[10,12,10]==0
    assert not s.labels[9].any()  # 2D brush edits only one plane
    s.labels[10,10,12]=b
    paint(s,'axial',(10,10,10),1.3,a,erase=True)
    assert s.labels[10,10,10]==0 and s.labels[10,10,12]==b
    paint(s,'axial',(2,10,10),10,a)
    assert not s.labels[2].any()


def test_connected_assignment_does_not_rename_disconnected_islands():
    s=Session(volume());a=s.add_segment().id;b=s.add_segment().id
    s.labels[3:8,3:8,3:8]=a;s.labels[15:20,20:25,20:25]=a
    result=assign_component(s,(5,5,5),b)
    assert result[5,5,5]==b and result[17,22,22]==a


@pytest.mark.parametrize('plane,axis',[('axial',0),('coronal',1),('sagittal',2)])
def test_slice_assignment_preserves_neighboring_slices(plane,axis):
    s=Session(volume());a=s.add_segment().id;b=s.add_segment().id
    box=(slice(3,19),slice(4,24),slice(5,30));s.apply(box,np.full(s.labels[box].shape,a,np.uint16),'initial')
    point=(10,10,10);result=assign_component(s,point,b,plane)
    assert result[point]==b
    neighbor=list(point);neighbor[axis]+=1
    assert result[tuple(neighbor)]==a
    np.testing.assert_array_equal(result>0,s.labels>0)


def test_seed_watershed_splits_bridge_preserves_unseeded_component():
    s=Session(volume());a=s.add_segment().id;b=s.add_segment().id;c=s.add_segment().id
    s.labels[4:18,5:18,4:15]=a;s.labels[4:18,5:18,17:28]=a
    s.labels[10:12,10:12,14:18]=a;s.labels[19:22,27:30,33:37]=a
    original=s.labels>0
    s.seeds=[Seed((10,10,8),b),Seed((10,10,23),c)]
    split=split_from_seeds(s)
    assert split[10,10,8]==b and split[10,10,23]==c and split[20,28,34]==a
    np.testing.assert_array_equal(split>0,original)


def test_project_roundtrip_contains_ct_geometry_names_and_seeds(tmp_path):
    s=Session(volume());bone=s.add_segment('左侧_肱骨');bone.reviewed=True
    s.labels[2:8,4:9,7:13]=bone.id;s.joint='shoulder';s.side='左侧'
    s.seeds=[Seed((5,6,9),bone.id)];s.set_roi((slice(1,20),slice(3,25),slice(4,30)))
    path=tmp_path/'test.j2stl';save_project(s,path);r=load_project(path)
    assert not (getattr(path.stat(),'st_flags',0)&0x8000), 'Saved projects must remain visible in file dialogs'
    np.testing.assert_array_equal(r.labels,s.labels);np.testing.assert_array_equal(r.volume.hu,s.volume.hu)
    np.testing.assert_array_equal(r.volume.affine_lps,s.volume.affine_lps)
    assert r.joint=='shoulder' and r.side=='左侧' and r.segments[0].reviewed
    assert tuple(r.seeds[0].zyx)==(5,6,9) and r.roi==s.roi
    assert not r.dirty


def test_mesh_preserves_lps_mm_with_reflected_direction():
    v=volume();v.affine_lps[0,0]*=-1
    mask=np.ones((8,10,12),dtype=bool);offset=(3,4,5)
    m=mesh_from_mask(mask,v,offset,sigma_mm=0)
    expected=v.world(np.array([[2.5,3.5,4.5],[10.5,13.5,16.5]]))
    np.testing.assert_allclose(m.bounds,np.sort(expected,axis=0),atol=1e-5)
    assert m.is_watertight and m.is_winding_consistent and m.volume>0


def test_hollow_bone_keeps_inner_cavity_orientation_in_stl(tmp_path):
    v=Volume(np.zeros((30,30,30),np.int16),np.eye(4))
    mask=np.zeros(v.hu.shape,bool);mask[3:27,3:27,3:27]=True;mask[9:21,9:21,9:21]=False
    mesh=mesh_from_mask(mask,v,sigma_mm=0)
    assert mesh.is_watertight and mesh.is_winding_consistent
    volumes=[part.volume for part in mesh.split(only_watertight=False)]
    assert any(value<0 for value in volumes)
    assert abs(mesh.volume-mask.sum())/mask.sum()<.01
    s=Session(v);bone=s.add_segment('骨皮质').id;s.parameters['surface_mm']=0
    s.parameters['mesh_smoothing']='off'  # This regression isolates cavity orientation and file conversion.
    s.apply(v.full_box,mask.astype(np.uint16)*bone,'cortical shell')
    directory,manifest=export_stl(s,tmp_path)
    loaded=trimesh.load_mesh(Path(directory)/manifest['segments'][0]['filename'],process=True,validate=False)
    assert loaded.is_watertight and abs(loaded.volume-mesh.volume)<.01


def test_export_stl_separate_files_no_overwrite_and_roundtrip(tmp_path):
    s=Session(volume());a=s.add_segment('左股骨');b=s.add_segment('左髌骨')
    s.labels[2:13,3:12,4:14]=a.id;s.labels[15:20,21:27,27:33]=b.id
    directory,manifest=export_stl(s,tmp_path)
    assert len(manifest['segments'])==2 and manifest['complete']
    for item in manifest['segments']:
        assert item['watertight'] and item['winding_consistent'] and item['positive_volume']
        assert trimesh.load_mesh(Path(directory)/item['filename']).is_watertight
    second,_=export_stl(s,tmp_path)
    assert second!=directory and (Path(directory)/'export_manifest.json').exists()


def test_preview_matches_native_export_surface_and_updates_after_edit(tmp_path):
    s=Session(volume());bone=s.add_segment('股骨').id
    box=(slice(4,15),slice(6,19),slice(7,21));mask=np.ones(s.labels[box].shape,bool)
    s.apply(box,np.full(mask.shape,bone,np.uint16),'initial')
    expected=mesh_from_mask(mask,s.volume,[b.start for b in box],surface_profile=s.parameters['mesh_smoothing'])
    preview=segment_meshes(s,preview=True)[0][1]
    np.testing.assert_array_equal(preview.vertices,expected.vertices)
    np.testing.assert_array_equal(preview.faces,expected.faces)
    export_stl(s,tmp_path)
    np.testing.assert_array_equal(preview.faces,expected.faces)
    old=preview.bounds.copy()
    extension=(slice(6,12),slice(8,16),slice(21,26))
    s.apply(extension,np.full(s.labels[extension].shape,bone,np.uint16),'extend')
    updated=segment_meshes(s,preview=True)[0][1]
    assert updated.bounds[1,0]>old[1,0]+1


def test_cleanup_fill_preserves_other_bones():
    s=Session(volume());a=s.add_segment().id;b=s.add_segment().id
    s.labels[3:15,4:17,5:20]=a;s.labels[6:10,7:12,8:15]=0;s.labels[8,9,10]=b
    result=clean_segment(s,a,'fill')
    assert result[7,8,9]==a and result[8,9,10]==b


def test_fill_recovers_scan_end_open_canal_without_expanding_outer_surface():
    from joint2stl.segmentation import fill_bone_cavities
    mask=np.zeros((32,36,36),bool);mask[:,6:30,6:30]=True;mask[:,10:26,10:26]=False
    from scipy import ndimage as ndi
    assert not (ndi.binary_fill_holes(mask)&~mask).any()
    result=fill_bone_cavities(mask,(1,.7,.7))
    assert result[:,6:30,6:30].all() and not result[:,:6].any()


def test_fill_repairs_orthogonally_enclosed_slots_and_keeps_joint_notch_open():
    from joint2stl.segmentation import fill_bone_cavities
    mask=np.zeros((40,40,40),bool);mask[3:37,8:32,8:32]=True
    mask[20:22,8:32,10:30]=False
    available=np.ones(mask.shape,bool);available[20,20,20]=False
    result=fill_bone_cavities(mask,(1,.7,.7),available)
    assert result[20,20,19] and not result[20,20,20]
    notch=np.zeros_like(mask);notch[20:37,8:32,8:32]=True;notch[20:27,8:32,17:23]=False
    np.testing.assert_array_equal(fill_bone_cavities(notch,(1,.7,.7)),notch)


def test_incremental_counts_stay_exact_across_brush_and_history():
    from joint2stl.segmentation import plane_box
    s=Session(volume());a=s.add_segment().id;b=s.add_segment().id
    box=(slice(4,14),slice(5,18),slice(6,20))
    s.apply(box,np.full(s.labels[box].shape,a,np.uint16),'initial')
    plane=plane_box(s.labels.shape,'axial',10);before=s.labels[plane].copy()
    paint(s,'axial',(10,10,10),2,b)
    s.record_preapplied(plane,before,'paint')
    for operation in [None,False,True,False]:
        if operation is not None:s.travel_history(operation)
        expected=count_labels(s.labels,3)
        np.testing.assert_array_equal(s.counts(),expected)
        assert s.counts().sum()==s.labels.size
    # Direct history travel also invalidates the lazy cache correctly.
    s.history.travel(s.labels)
    np.testing.assert_array_equal(s.counts(),count_labels(s.labels,3))
    s.history.travel(s.labels,redo=True)
    s.travel_history()
    np.testing.assert_array_equal(s.counts(),count_labels(s.labels,3))
