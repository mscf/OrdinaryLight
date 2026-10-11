from pathlib import Path
import subprocess
s=(Path(__file__).parents[1]/'ordinarylight/portable/web/surface-runtime.js').read_text().replace('export class','class')
def test_mesh_visibility():
    subprocess.run(['node','-e',s+r'''
const assert=require('node:assert/strict');
const r=Object.create(OrdinaryLightSurface.prototype);
r.canvas={width:500,height:500,getBoundingClientRect:()=>({width:500,height:500})};
r.camera={eye:{x:0,y:0,z:3},center:{x:0,y:0,z:0},up:{x:0,y:1,z:0}};
r.positions=[-1,-1,0,1,-1,0,0,1,0];r.indices=[0,1,2];
assert.ok(r.isVisible([0,0,0]));assert.ok(!r.isVisible([0,0,-.2]));assert.ok(r.isVisible([0,0,.2]));
assert.deepEqual(r.hitTest(250,250).position,[0,0,0]);assert.notEqual(r.pick(250,250),null);
r.clipBounds=[[-1,1],[-1,1],[-1,-.1]];
assert.equal(r.hitTest(250,250),null);assert.ok(!r.isVisible([0,0,0]));assert.ok(r.isVisible([0,0,-.2]));
'''],check=True)
