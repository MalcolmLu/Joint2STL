import importlib.util
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import numpy as np
import pytest
from scipy import ndimage as ndi

from joint2stl.ai import screen_proposal
from joint2stl.compute import available_backends, gaussian_filter


def test_ai_keeps_bright_thin_cortex_near_model_contour():
    prior=np.zeros((40,40),bool);prior[10:30,10:30]=True
    probability=np.zeros((40,40),np.float32);probability[11:29,11:29]=0.9
    image=np.zeros((40,40),np.float32);image[prior]=600
    result,quality=screen_proposal(prior,probability,image,(0.6,0.6))
    assert quality['accepted']
    np.testing.assert_array_equal(result,prior)


def test_ai_cannot_cut_away_existing_dense_core_with_an_inward_proposal():
    prior=np.zeros((50,50),bool);prior[10:40,10:40]=True
    probability=np.zeros(prior.shape,np.float32);probability[16:34,16:34]=.9
    image=np.zeros(prior.shape,np.float32);image[prior]=700
    result,quality=screen_proposal(prior,probability,image,(.6,.6))
    assert quality['accepted']
    np.testing.assert_array_equal(result,prior)


def test_ai_rejects_disappearance_of_separate_bone_island():
    prior=np.zeros((70,70),bool);prior[10:35,10:35]=True;prior[40:52,40:52]=True
    p=np.zeros(prior.shape,np.float32);p[10:35,10:35]=0.9
    result,quality=screen_proposal(prior,p,np.zeros(prior.shape),(0.6,0.6))
    assert not quality['accepted'] and '可能漏掉独立骨片' in quality['flags']


def test_ai_does_not_overwrite_another_bone():
    prior=np.zeros((40,40),bool);prior[10:30,10:30]=True
    p=prior.astype(np.float32)*0.9;foreign=np.zeros(prior.shape,bool);foreign[20:23,20:23]=True
    result,quality=screen_proposal(prior,p,np.ones(prior.shape)*800,(0.6,0.6),foreign)
    assert not result[foreign].any()


def test_ct_boundary_rejects_old_soft_tissue_overlabel_at_articular_edge():
    bone=np.zeros((64,64),bool);bone[18:46,18:46]=True
    prior=np.zeros_like(bone);prior[15:49,15:49]=True
    image=np.ones(prior.shape,np.float32)*40;image[bone]=700
    prediction,quality=screen_proposal(prior,prior.astype(np.float32)*.9,image,(1,1))
    assert quality['accepted'] and quality['ct_edge_constraint_applied']
    np.testing.assert_array_equal(prediction,bone)


def test_ct_supported_solid_marrow_does_not_get_new_internal_model_voids():
    from scipy import ndimage as ndi
    prior=np.zeros((64,64),bool);prior[15:49,15:49]=True
    cortex=prior&~ndi.binary_erosion(prior,iterations=2)
    image=np.ones(prior.shape,np.float32)*40;image[prior]=0;image[cortex]=700
    probability=prior.astype(np.float32)*.9;probability[25:28,15:49]=0
    result,quality=screen_proposal(prior,probability,image,(.7,.7))
    assert quality['accepted'] and result[26,32]
    assert not result[~prior].any()


def test_weak_cortical_patch_does_not_open_reference_and_remove_trabecular_interior():
    from joint2stl.ai import ct_bone_reference
    from scipy import ndimage as ndi
    prior=np.zeros((34,34,34),bool);prior[5:29,5:29,5:29]=True
    cortex=prior&~ndi.binary_erosion(prior)
    image=np.zeros(prior.shape,np.float32);image[cortex]=700
    image[11:24,11:24,28]=170  # Weak, still continuous cortex beside marrow.
    reference=ct_bone_reference(prior,image,(1,1,1))
    assert reference[17,17,24] and reference[17,17,28]
    np.testing.assert_array_equal(reference,prior)


def test_ai_rejects_nonfinite_or_large_leak():
    prior=np.zeros((60,60),bool);prior[20:40,20:40]=True
    result,q=screen_proposal(prior,np.full(prior.shape,np.nan),np.zeros(prior.shape),(1,1))
    assert not q['accepted'];np.testing.assert_array_equal(result,prior)
    result,q=screen_proposal(prior,np.ones(prior.shape),np.zeros(prior.shape),(1,1))
    assert not q['accepted']


def test_cpu_and_auto_filter_keep_float32_and_anisotropic_geometry():
    field=np.pad(np.ones((12,13,14),np.float32),4)
    sigma=(0.5,0.7,1.1)
    expected=ndi.gaussian_filter(field,sigma)
    np.testing.assert_array_equal(gaussian_filter(field,sigma,'cpu'),expected)
    np.testing.assert_allclose(gaussian_filter(field,sigma,'auto'),expected,rtol=3e-6,atol=3e-7)


@pytest.mark.skipif('metal' not in available_backends(),reason='Apple Metal GPU unavailable')
def test_metal_matches_cpu_at_borders_and_multiple_reflections():
    field=np.random.default_rng(9).random((4,7,12),dtype=np.float32)
    sigma=(2.1,0.6,1.4)
    expected=ndi.gaussian_filter(field,sigma)
    actual=gaussian_filter(field,sigma,'metal')
    np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=3e-7)


@pytest.mark.skipif('cuda' not in available_backends(),reason='NVIDIA CUDA GPU unavailable')
def test_cuda_matches_cpu_at_borders():
    field=np.random.default_rng(9).random((41,47,52),dtype=np.float32)
    expected=ndi.gaussian_filter(field,(0.4,0.6,1.4))
    actual=gaussian_filter(field,(0.4,0.6,1.4),'cuda')
    np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=3e-7)


@pytest.mark.skipif(os.name!='nt' or 'cuda' not in available_backends(),reason='Windows NVIDIA CUDA required')
def test_windows_cuda_filter_then_model_in_fresh_process():
    import subprocess
    import sys
    from joint2stl.ai import model_ready

    if not model_ready():
        pytest.skip('LiteMedSAM and optional model dependencies required')
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA PyTorch required')
    # Keep this in a new process so the CUDA test setup cannot pre-load PyTorch.
    code = r'''
import numpy as np
from joint2stl.compute import gaussian_filter, last_backend
from joint2stl.ai import infer_slice

yy, xx = np.ogrid[:64, :80]
image = np.where(((xx-40)/19)**2+((yy-32)/24)**2<1,650,45).astype(np.float32)
field = np.pad(np.ones((7,10,6),np.float32),3)
sigma = (.5,.6,.7)
cpu = gaussian_filter(field,sigma,'cpu')
np.testing.assert_allclose(gaussian_filter(field,sigma,'cuda'),cpu,rtol=3e-6,atol=3e-7)
assert last_backend() == 'cuda'
result, score, device = infer_slice(image,[19,5,61,59],'cuda')
assert device == 'cuda' and result.shape == image.shape
assert np.isfinite(result).all() and np.isfinite(score)
np.testing.assert_allclose(gaussian_filter(field,sigma,'cuda'),cpu,rtol=3e-6,atol=3e-7)
assert last_backend() == 'cuda'
'''
    run = subprocess.run([sys.executable,'-c',code],capture_output=True,timeout=120,
                         encoding='utf-8',errors='replace',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    assert run.returncode==0, run.stdout+'\n'+run.stderr


def test_unavailable_backend_is_explicit_not_silent():
    with pytest.raises(ValueError):gaussian_filter(np.zeros((4,4,4)),(1,1,1),'not-a-backend')
