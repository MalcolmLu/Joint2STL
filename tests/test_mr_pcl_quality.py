"""A generic bright soft-tissue blob must not silently become a sagittal PCL."""
import numpy as np
import pytest
from joint2stl.model import Session,Volume
from joint2stl.mr_tools import ensure_mr_segments,predict_mr_prompt
import joint2stl.mr_tools as tools


@pytest.mark.parametrize('scale,offset',[(1,0),(3,500),(.1,-200)])
def test_new_sagittal_pcl_dark_band_vs_bright_blob(monkeypatch,scale,offset):
    shape=(5,60,60);image=np.full(shape,120.,np.float32)
    # Relative signal has a broad range, rather than using any CT/HU threshold.
    image[:,10:50,10:30]=80;image[:,10:50,30:50]=160
    dark=np.zeros((60,60),bool);dark[15:45,27:31]=True
    blob=np.zeros_like(dark);blob[20:40,32:48]=True
    image[:,dark]=10;image[:,blob]=180
    affine=np.array([[0,0,4,0],[.5,0,0,0],[0,-.5,0,0],[0,0,0,1]],float)
    s=Session(Volume(image*scale+offset,affine,modality='MR'));ensure_mr_segments(s,acl_absent=True)
    for prediction,accepted in [(dark,1),(blob,0)]:
        monkeypatch.setattr(tools.ai,'infer_slice',lambda *a,**kw:(prediction.astype(float),.8,'cpu'))
        candidate,report=predict_mr_prompt(s,6,'axial',(2,10,10),(2,50,50),device='cpu')
        assert report['accepted_slices']==accepted
        assert bool(np.any(candidate==6))==bool(accepted)
        assert not np.any(candidate==5) and s.segment(5).presence=='absent'
        if not accepted:assert 'pcl_signal_or_shape_uncertain' in report['slices'][0]['flags']
