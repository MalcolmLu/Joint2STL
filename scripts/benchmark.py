"""Compare the Gaussian filter and mesh construction on one existing bone."""
from pathlib import Path
import argparse,json,platform,time
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from joint2stl.model import load_project
from joint2stl.compute import available_backends,gaussian_filter
from joint2stl.meshing import mesh_from_mask

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project');parser.add_argument('--output');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    session=load_project(args.project or root/'validation/膝关节_可编辑示例.j2stl')
    box=ndi.find_objects(session.labels)[0];mask=session.labels[box]==1
    field=np.pad(mask,3).astype(np.float32);sigma=.35/session.volume.spacing_zyx
    result={'platform':platform.platform(),'available_backends':available_backends(),'field_shape':list(field.shape),
            'scope':'Timings include CPU/GPU array conversion. The first filter call is reported separately from the median of three subsequent calls.'}
    fields={}
    for backend in available_backends():
        times=[]
        for _ in range(4):
            start=time.perf_counter();output=gaussian_filter(field,sigma,backend);times.append(time.perf_counter()-start)
        fields[backend]=output
        result[backend]={'first_filter_seconds':times[0],'warm_filter_median_seconds':float(np.median(times[1:])),
                         'max_abs_error_against_cpu':float(np.max(np.abs(output-fields['cpu'])))}
    cpu_mesh=None
    for backend in available_backends():
        start=time.perf_counter();mesh=mesh_from_mask(mask,session.volume,[s.start for s in box],backend=backend)
        result[backend].update(mesh_seconds=time.perf_counter()-start,mesh_faces=len(mesh.faces),mesh_closed=bool(mesh.is_watertight))
        if cpu_mesh is None:cpu_mesh=mesh
        else:
            points=mesh.vertices[::max(1,len(mesh.vertices)//20000)]
            distance=cKDTree(cpu_mesh.vertices).query(points)[0]
            result[backend]['sampled_max_vertex_distance_to_cpu_mm']=float(distance.max())
    output=Path(args.output) if args.output else root/'validation/gpu_benchmark.json'
    output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
