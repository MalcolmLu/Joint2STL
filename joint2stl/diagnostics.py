"""Hardware report contains no patient identity or source file paths."""
from importlib.metadata import version,PackageNotFoundError
import platform
from . import __version__
from .compute import available_backends
from .resources import detect_memory,memory_policy

def hardware_report():
    policy=memory_policy(detect_memory())
    result={'platform':platform.platform(),'machine':platform.machine(),
            'surface_backends':available_backends(),'inference_device':None,'packages':{'joint2stl':__version__},
            'memory_policy':policy,
            'processing_limits_voxels':{key:entry['limit_voxels'] for key,entry in policy['operations'].items()},
            'gpu_memory_budgets_bytes':{'surface_filter':2*1024**3,'apple_ai_fraction':.35,'cuda_ai_fraction':.5}}
    for package in ['numpy','scipy','pydicom','scikit-image','trimesh','PySide6-Essentials','vtk','psutil','mlx','torch','torchvision','cupy-cuda12x']:
        try:result['packages'][package]=version(package)
        except PackageNotFoundError:pass
    try:
        from .mr_bone import mri_bone_model_ready,CHECKPOINT_SHA256
        result['mri_bone_model']={'name':'SegmentAnyBone','ready':mri_bone_model_ready(),
            'checkpoint_sha256':CHECKPOINT_SHA256,'license':'CC BY-NC 4.0','task':'binary MRI bone candidates; anatomical names require review'}
        from .mr_correction import correction_model_ready,CHECKPOINT_SHA256 as CORRECTION_SHA256
        result['mri_correction_model']={'name':'ScribblePrompt-UNet v1','ready':correction_model_ready(),
            'checkpoint_sha256':CORRECTION_SHA256,'license':'Apache-2.0','task':'local MRI positive/negative prompt corrections; anatomy and completeness require review'}
        from .ai import torch_device,model_ready
        result['inference_device']=torch_device()
        result['model_ready']=model_ready()
        if result['inference_device']=='cuda':
            import torch
            result['inference_gpu_name']=torch.cuda.get_device_name()
        elif result['inference_device']=='mps':result['inference_gpu_name']='Apple GPU / MPS'
    except (ImportError,RuntimeError) as exc:result['inference_status']=str(exc)
    return result
