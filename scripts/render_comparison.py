"""Render source-image contour comparisons without patient text or UI screenshots."""
from pathlib import Path
import argparse
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joint2stl.model import load_project

parser=argparse.ArgumentParser();parser.add_argument('--before',required=True);parser.add_argument('--after',required=True);parser.add_argument('--output',required=True)
args=parser.parse_args();before=load_project(args.before);after=load_project(args.after)
if before.labels.shape!=after.labels.shape or not np.allclose(before.volume.affine_lps,after.volume.affine_lps):
    parser.error('Projects must use the same source CT geometry.')
if before.volume.series_key and after.volume.series_key:
    if before.volume.series_key!=after.volume.series_key:parser.error('Projects have different source CT series.')
elif not np.array_equal(before.volume.hu,after.volume.hu):parser.error('Projects have different source CT pixels.')
boxes_before=ndi.find_objects(before.labels);boxes_after=ndi.find_objects(after.labels)
ids=[s.id for s in before.segments if s.id<=len(boxes_before) and boxes_before[s.id-1] is not None][:4]
if not ids:parser.error('The initial project has no nonempty labels.')
fig,axes=plt.subplots(len(ids),3,figsize=(12,3.8*len(ids)),layout='constrained',facecolor='#101824',squeeze=False)
for row,ident in enumerate(ids):
    a=boxes_before[ident-1];b=boxes_after[ident-1] if ident<=len(boxes_after) else None;b=b or a
    box=tuple(slice(min(x.start,y.start),max(x.stop,y.stop)) for x,y in zip(a,b))
    mask_a=before.labels[box]==ident;mask_b=after.labels[box]==ident
    differences=np.count_nonzero(mask_a^mask_b,axis=(1,2))
    if differences.max():z=int(differences.argmax())+box[0].start
    else:
        occupied=np.flatnonzero(mask_a.any(axis=(1,2)));z=int(occupied[len(occupied)//2])+box[0].start
    xy=tuple(slice(max(0,s.start-12),min(n,s.stop+12)) for s,n in zip(box[1:],before.labels.shape[1:]))
    image=before.volume.hu[(z,*xy)]
    for col,title in enumerate(['Source CT','Initial contour','Model candidate contour']):
        ax=axes[row,col];ax.imshow(image,cmap='gray',vmin=-250,vmax=1750,aspect=before.volume.spacing_zyx[1]/before.volume.spacing_zyx[2]);ax.axis('off')
        if col:
            target=(before.labels if col==1 else after.labels)[(z,*xy)]==ident
            if target.any():ax.contour(target.astype(float),levels=[.5],colors=['#ffc96c' if col==1 else '#6be0c1'],linewidths=1)
        ax.set_title(f'{title} | label {ident}, slice {z+1}',color='white',fontsize=10)
fig.suptitle('Real CT | largest changed slice for each label\nSource-image review; no expert reference segmentation is available',color='white',fontsize=14)
output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(output,dpi=130,facecolor=fig.get_facecolor());plt.close(fig);print(output)
