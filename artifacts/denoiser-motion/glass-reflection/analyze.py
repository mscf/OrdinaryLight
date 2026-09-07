from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
out=Path('/tmp/glass-comparison')
f={k:np.load(out/f'guides-{k}.npy') for k in ['False','True']};b=np.load(out/'reference-batches.npy');ref=b.mean(0)
# Interior patches avoid silhouettes and direct foreground occlusion.
rois={'direct':(100,205,160,235),'reflection':(145,140,166,162)}
def lum(a):return a@np.array([.2126,.7152,.0722])
def mapped(a):return np.log1p(np.maximum(lum(a),0))
metrics={}
for name,(x0,y0,x1,y1) in rois.items():
 sl=np.s_[y0:y1,x0:x1];target=mapped(ref[sl]);row={}
 row['reference_split_rmse']=float(np.sqrt(np.mean((mapped(b[::2].mean(0)[sl])-mapped(b[1::2].mean(0)[sl]))**2)))
 for k in f:
  a=mapped(f[k][:,y0:y1,x0:x1]);row[k]={'moving_endpoint_rmse':float(np.sqrt(np.mean((a[15]-target)**2))),'settled_rmse':float(np.sqrt(np.mean((a[-1]-target)**2))),'stationary_temporal_std':float(np.std(a[-16:],axis=0).mean())}
 row['settled_toggle_rmse']=float(np.sqrt(np.mean((mapped(f['True'][-1][sl])-mapped(f['False'][-1][sl]))**2)))
 metrics[name]=row
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
def display(a):
 a=np.maximum(a,0);return (np.clip(a/(1+a),0,1)**(1/2.2)*255).astype('uint8')
canvas=Image.new('RGB',(5*210,2*234));d=ImageDraw.Draw(canvas)
for r,(name,box) in enumerate(rois.items()):
 for c,(label,a) in enumerate([('Off: moving',f['False'][15]),('On: moving',f['True'][15]),('Off: settled',f['False'][-1]),('On: settled',f['True'][-1]),('512-sample reference',ref)]):
  crop=Image.fromarray(display(a)).crop(box).resize((200,200),Image.Resampling.NEAREST)
  canvas.paste(crop,(c*210,r*234+30));d.text((c*210+3,r*234+3),name+' / '+label,fill='white')
canvas.save(out/'crops.png')
