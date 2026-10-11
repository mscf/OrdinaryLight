from pathlib import Path
import subprocess


def test_wheel_policy_and_delta_units():
    source=(Path(__file__).parents[1]/"ordinarylight/portable/web/surface-runtime.js").read_text()
    subprocess.run(["node","--input-type=module","-e",source+r"""
import assert from 'node:assert/strict';
globalThis.ResizeObserver=class{observe(){}};
function create(policy){const listeners={};const canvas={style:{},clientHeight:500,addEventListener:(n,f)=>listeners[n]=f};const runtime=new OrdinaryLightSurface(canvas,'',{wheelZoom:policy});runtime.changed=()=>{};return {runtime,wheel:listeners.wheel};}
function event(extra={}){return {deltaY:1,deltaMode:0,preventDefault(){this.prevented=true;},...extra};}
for(const policy of ['modifier','disabled']){const {runtime,wheel}=create(policy),before=structuredClone(runtime.camera),e=event();wheel(e);assert.deepEqual(runtime.camera,before);assert.ok(!e.prevented);}
const a=create('modifier'),b=create('always'),e=event({altKey:true,deltaMode:1});a.wheel(e);b.wheel(event({deltaY:16}));assert.deepEqual(a.runtime.camera,b.runtime.camera);assert.ok(e.prevented);
const camera=create('always').runtime;camera.camera.center={x:.2,y:.1,z:.3};const center=structuredClone(camera.camera.center),radius=()=>Math.hypot(...['x','y','z'].map(k=>camera.camera.eye[k]-center[k])),beforeRadius=radius();camera.rotate(.2,.1);assert.ok(Math.abs(radius()-beforeRadius)<1e-12);assert.deepEqual(camera.camera.center,center);camera.rotate(0,100);assert.ok(Number.isFinite(radius()));assert.throws(()=>camera.rotate(NaN,0));
assert.throws(()=>create('unknown'));assert.throws(()=>a.runtime.zoom(0));assert.throws(()=>a.runtime.zoom(NaN));
"""],check=True,capture_output=True,text=True)
