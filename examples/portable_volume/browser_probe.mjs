/** Run with Node 22+, or Node 20 --experimental-websocket.
 * node browser_probe.mjs <CDP URL> <demo URL> <output.json>
 * Uses a dedicated, already-running Chrome debugging instance.
 */
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const [endpoint, url, output] = process.argv.slice(2);
assert(endpoint && url && output, 'Expected CDP URL, demo URL, output.json');
const tabs = await (await fetch(`${endpoint}/json`)).json();
const tab = tabs.find(t => t.type === 'page');
assert(tab, 'No browser page found');
const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {ws.onopen=resolve;ws.onerror=reject;});
let id=0;
const pending=new Map();
ws.onmessage=event=>{
  const m=JSON.parse(event.data);
  if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(new Error(m.error.message)):p.resolve(m.result);}
};
const send=(method,params={})=>new Promise((resolve,reject)=>{
  const next=++id;pending.set(next,{resolve,reject});ws.send(JSON.stringify({id:next,method,params}));
});
async function evaluate(expression){
  const result=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(result.exceptionDetails)throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
  return result.result.value;
}
const timeout=setTimeout(()=>{console.error('Browser verification timed out');process.exit(1);},60000);
try {
  await send('Network.enable');
  await send('Network.setCacheDisabled',{cacheDisabled:true});
  await send('Page.enable');
  const loaded=new Promise(resolve=>{
    const listener=event=>{
      if(JSON.parse(event.data).method==='Page.loadEventFired'){
        ws.removeEventListener('message',listener);resolve();
      }
    };
    ws.addEventListener('message',listener);
  });
  await send('Page.navigate',{url});
  await loaded;
  let state;
  for(let i=0;i<100;i++){
    state=await evaluate('({ready:globalThis.ready,error:globalThis.startError})');
    if(state.ready||state.error)break;
    await new Promise(resolve=>setTimeout(resolve,200));
  }
  assert(!state.error,state.error);assert(state.ready,'Demo did not become ready');
  const result=await evaluate(`(async()=>{
    const {PortableRuntime,fetchPackage}=await import('./runtime.js');
    const check=(test,message)=>{if(!test)throw new Error(message);};
    const bytes=buffer=>Array.from(new Uint8Array(buffer));
    const read=async r=>Object.fromEntries(await Promise.all(Object.keys(r.manifest.outputs).map(async name=>[name,bytes(await r.read(name))])));
    const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
    const r=viewer.runtime, device=r.device;
    const errors=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
    device.pushErrorScope('validation');
    const initial=await read(r), snapshot=r.snapshot(), pixels=Array.from(await r.readPixels());
    check(new Set(pixels.filter((_,i)=>i%4!==3)).size>8,'Volume image is blank');
    await viewer.update({parameters:{seed:9}});
    const updated=await read(r);
    check(!equal(initial.rts,updated.rts),'Live seed update did not change output');
    check(viewer.runtime===r && r.device===device,'Live edit replaced runtime/device');
    await viewer.update({presentation:{yaw:1.2,mode:'slice'}});
    check(equal(updated,await read(r)),'Presentation edit changed computation');
    r.restore(snapshot);r.render();
    check(equal(initial,await read(r)),'Snapshot did not restore numerical results');
    check(equal(pixels,Array.from(await r.readPixels())),'Snapshot did not restore pixels');
    const bad=structuredClone(snapshot);bad.presentation.opacity=-1;
    let rejected=false;try{r.restore(bad);}catch{rejected=true;}
    check(rejected && equal(snapshot,r.snapshot()),'Invalid restore was not atomic');
    rejected=false;try{r.setParameters({trial_count:24});}catch(e){rejected=e.code==='reprepare-required';}
    check(rejected && equal(snapshot,r.snapshot()),'Structural edit was accepted as live');
    const restored=await PortableRuntime.create(await fetchPackage('/initial/manifest.json'),{device});
    restored.restore(JSON.parse(JSON.stringify(snapshot)));restored.render(512,512);
    check(equal(initial,await read(restored)),'Fresh runtime restoration changed output');
    check(equal(pixels,Array.from(await restored.readPixels())),'Fresh runtime restoration changed pixels');
    restored.close();
    await viewer.update({parameters:{trial_count:24},preparation:{bins:20}});
    check(viewer.runtime!==r && r.closed,'Structural edit did not replace/close old runtime');
    check(viewer.runtime.device===device,'Replacement allocated a second device');
    check(viewer.runtime.manifest.science.samples_per_cell===24 && viewer.runtime.manifest.render.dimensions[2]===20,'Replacement shape is wrong');
    const replacement=await read(viewer.runtime), replacementSnapshot=viewer.runtime.snapshot();
    const keep=viewer.runtime;
    rejected=false;try{await viewer.update({preparation:{x:keep.manifest.science.y.name}});}catch{rejected=true;}
    check(rejected && viewer.runtime===keep && !keep.closed,'Failed preparation discarded active runtime');
    await device.queue.onSubmittedWorkDone();
    const gpuError=await device.popErrorScope();check(!gpuError,gpuError?.message);check(!errors.length,errors.join('; '));
    const info=device.adapterInfo;
    return {initial,updated,replacement,snapshot,replacementSnapshot,pixels,width:512,height:512,
      browserAdapter:{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description},
      checks:['render','live update','presentation independence','same-runtime restore','fresh-runtime restore','invalid restore atomicity','structural guard','same-device replacement','failed replacement recovery'],gpuErrors:errors};
  })()`);
  await writeFile(output,JSON.stringify(result));
  console.log(JSON.stringify({checks:result.checks,gpuErrors:result.gpuErrors,output},null,2));
} finally {clearTimeout(timeout);ws.close();}
