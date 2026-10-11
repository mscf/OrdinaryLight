"""CPU geometry conformance for portable surface clipping (requires Node)."""
from pathlib import Path
import json
import shutil
import subprocess
import pytest


def test_clip_preserves_plane_attributes_area_and_inputs():
    if not shutil.which('node'):
        pytest.skip('Node is required')
    runtime=Path(__file__).resolve().parents[1]/'ordinarylight/portable/web/surface-runtime.js'
    script='''
    import fs from 'node:fs';
    const source=fs.readFileSync(PATH,'utf8');
    const {OrdinaryLightSurface:R}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
    const positions=[-1,-1,0,1,-1,0,1,1,0,-1,1,0],scalars=[-3,-1,3,1],indices=[0,1,2,0,2,3];
    const before=JSON.stringify({positions,scalars,indices});
    const mesh=R.clipMesh(positions,scalars,indices,[[-.25,.25],[-.25,.25],[-.5,.5]]);
    let area=0;
    for(let i=0;i<mesh.indices.length;i+=3){
      const [a,b,c]=mesh.indices.slice(i,i+3).map(id=>mesh.positions.slice(id*3,id*3+3));
      area+=Math.abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2;
    }
    const slopePositions=[-2,-2,-4,2,-2,0,0,2,2];
    const slope=R.clipMesh(slopePositions,[-18,-2,10],[0,1,2],[[-.5,.5],[-.5,.5],[-.5,.5]]);
    const empty=R.clipMesh(positions,scalars,indices,[[2,3],[-1,1],[-1,1]]);
    console.log(JSON.stringify({mesh,area,slope,empty,unchanged:before===JSON.stringify({positions,scalars,indices})}));
    '''.replace('PATH',json.dumps(str(runtime)))
    result=json.loads(subprocess.check_output(['node','--input-type=module','-e',script],text=True))
    assert result['unchanged']
    assert result['area']==pytest.approx(.25)
    assert result['empty']['indices']==[]
    for name,limit in [('mesh',.25),('slope',.5)]:
        mesh=result[name]
        assert mesh['indices']
        for i,scalar in enumerate(mesh['scalars']):
            x,y,z=mesh['positions'][i*3:i*3+3]
            assert abs(x)<=limit+1e-12 and abs(y)<=limit+1e-12
            if name=='slope':
                assert abs(z)<=.5+1e-12
                assert z==pytest.approx(x+y)
                assert scalar==pytest.approx(x+2*y+3*z)
            else:
                assert scalar==pytest.approx(x+2*y)
