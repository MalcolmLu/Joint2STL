"""VTK offscreen GPU draw benchmark; does not claim to measure native Qt interaction."""
from pathlib import Path
import argparse,json,time
import numpy as np
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkRenderingCore import vtkActor,vtkPolyDataMapper,vtkRenderer,vtkRenderWindow
from vtkmodules.vtkIOGeometry import vtkSTLReader

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--stl-dir');parser.add_argument('--output');args=parser.parse_args()
if args.stl_dir:directory=Path(args.stl_dir)
else:
    path=root/'validation/real_ct_validation.json'
    if not path.exists():parser.error('Specify --stl-dir with a Joint2STL export folder.')
    project=json.loads(path.read_text(encoding='utf-8'));directory=Path(project['cases'][3]['export_directory'])
if not list(directory.glob('*.stl')):parser.error('No STL files in the selected folder.')
renderer=vtkRenderer();renderer.SetBackground(.05,.08,.12)
window=vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(1024,768);window.SetMultiSamples(0);window.AddRenderer(renderer)
faces=0
for path in sorted(directory.glob('*.stl')):
    reader=vtkSTLReader();reader.SetFileName(str(path));reader.Update();faces+=reader.GetOutput().GetNumberOfCells()
    mapper=vtkPolyDataMapper();mapper.SetInputConnection(reader.GetOutputPort());actor=vtkActor();actor.SetMapper(mapper);renderer.AddActor(actor)
renderer.ResetCamera();window.Render()
capabilities=window.ReportCapabilities()
gpu=[line.strip() for line in capabilities.splitlines() if any(k in line.lower() for k in ['opengl vendor string','opengl renderer string','opengl version string'])]
times=[]
for _ in range(20):
    renderer.GetActiveCamera().Azimuth(2)
    start=time.perf_counter();window.Render()
    if hasattr(window,'WaitForCompletion'):window.WaitForCompletion()
    times.append(time.perf_counter()-start)
result={'gpu':gpu,'faces':faces,'frame_size':[1024,768],'median_render_ms':float(np.median(times)*1000),
        'p95_render_ms':float(np.percentile(times,95)*1000),'gpu_completion_waited':hasattr(window,'WaitForCompletion'),
        'scope':'Offscreen VTK draw with GPU completion where supported. Native mouse interaction not included.'}
output=Path(args.output) if args.output else root/'validation/render_benchmark.json'
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));window.Finalize()
