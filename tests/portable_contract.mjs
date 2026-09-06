import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {webcrypto} from 'node:crypto';
import {pathToFileURL} from 'node:url';
globalThis.crypto ??= webcrypto;
const root=process.argv[2];
const {validatePackage,packParameters,checkCapabilities,PortableRuntime}=await import(pathToFileURL(`${root}/runtime.mjs`));
const m=JSON.parse(await readFile(`${root}/manifest.json`));
const data=await readFile(`${root}/payload.bin`);
const payload=data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength);
await validatePackage(m,payload);
assert.deepEqual(Array.from(new Float32Array(packParameters(m,m.state.parameters))),[1,2,4,2,2,4,1,3,4,2,3,4]);
for(const mutate of [
  m=>m.schema='future',
  m=>m.passes[0].bindings[0].access='read',
  m=>m.render.uniform_layout.camera=32,
  m=>m.parameters.fields[0].offset=4,
  m=>m.science.x.samples=[1],
  m=>m.state.parameters.unknown=1,
]) {const invalid=structuredClone(m);mutate(invalid);await assert.rejects(validatePackage(invalid,payload));}
assert.throws(()=>checkCapabilities(m,{features:new Set(),limits:{}}),/GPU limit/);
const runtime=Object.create(PortableRuntime.prototype);
runtime.manifest=m;runtime.values=structuredClone(m.state.parameters);runtime.presentation=structuredClone(m.state.presentation);
runtime.closed=false;runtime.buffers=new Map([['parameters',{}]]);
let writes=0,computes=0;
runtime.device={queue:{writeBuffer:()=>writes++}};runtime.compute=()=>computes++;
const snapshot=runtime.snapshot();
assert.throws(()=>runtime.setParameters({n:5}),e=>e.code==='reprepare-required');
assert.equal(writes,0);
assert.throws(()=>runtime.restore({...snapshot,presentation:{...snapshot.presentation,opacity:-1}}),/out of range/);
assert.deepEqual(runtime.snapshot(),snapshot);assert.equal(writes,0);
runtime.restore(JSON.parse(JSON.stringify(snapshot)));
assert.equal(writes,1);assert.equal(computes,1);
console.log('Browser contract checks passed');
