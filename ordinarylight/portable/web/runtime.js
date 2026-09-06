// OrdinaryLight portable-volume/v1. ES module, usable without a notebook or UI framework.
const schema = 'ordinarylight/portable-volume-v1';
const fail = message => { throw new Error(`Portable package: ${message}`); };
const clone = value => structuredClone(value);
const integer = n => Number.isSafeInteger(n) && n > 0;
const digest = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
const textDigest = text => digest(new TextEncoder().encode(text));

export async function validatePackage(manifest, payload) {
  const m=manifest;
  if (m.schema!==schema || m.byte_order!=='little') fail('unsupported schema/byte order');
  if(m.payload.file!=='payload.bin' || !m.resources.length || !m.passes.length) fail('resources, passes and payload.bin are required');
  if (!(payload instanceof ArrayBuffer)) fail('payload must be an ArrayBuffer');
  if (payload.byteLength!==m.payload.byte_length || await digest(payload)!==m.payload.sha256) fail('payload integrity failure');
  const resources=new Map(); let bytes=0;
  for (const r of m.resources) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(r.id) || resources.has(r.id)) fail('invalid/duplicate resource identity');
    if (r.kind!=='buffer' || !['f32','i32','u32'].includes(r.dtype) || !r.shape?.length || !r.shape.every(integer)) fail('unsupported resource layout');
    if (!integer(r.byte_length) || r.shape.reduce((a,b)=>a*b,4)!==r.byte_length || r.byte_length>128*1024**2) fail('invalid buffer byte length');
    if (JSON.stringify(r.usage)!==JSON.stringify(['storage','copy-src','copy-dst'])) fail('unsupported buffer usage');
    if (r.initial && (!Number.isSafeInteger(r.initial.offset) || r.initial.offset<0 || r.initial.offset%4 || r.initial.length!==r.byte_length || r.initial.offset+r.initial.length>payload.byteLength)) fail('invalid initial payload range');
    bytes+=r.byte_length; resources.set(r.id,r);
  }
  if (bytes>512*1024**2) fail('allocation budget exceeds v1 limit');
  const passIds=new Set();
  for (const p of m.passes) {
    if(p.kind!=='compute' || passIds.has(p.id) || p.after.some(x=>!passIds.has(x))) fail('invalid pass ordering');
    passIds.add(p.id);
    const s=m.shaders[p.shader];
    if(s?.schema!=='ordinaryshade/shader-v1' || s.target!=='wgsl' || await textDigest(s.source)!==s.sha256 || s.reflection.stage!=='compute') fail('unsupported/corrupt compute shader');
    if(s.reflection.workgroup_size.length!==3 || !s.reflection.workgroup_size.every(integer) || !p.workgroups.every(integer) || p.workgroups.length!==3) fail('invalid dispatch');
    const reflection=s.reflection.resources;
    if(reflection.length!==p.bindings.length) fail('binding reflection mismatch');
    const seen=new Set();
    for(const b of p.bindings) {
      const r=reflection.find(r=>r.set===b.group && r.binding===b.binding);
      if(b.group!==0 || !r || r.kind!=='storage_buffer' || r.access!==b.access || !resources.has(b.resource) || seen.has(b.binding)) fail('invalid binding contract');
      seen.add(b.binding);
    }
  }
  if(Object.values(m.outputs).some(id=>!resources.has(id))) fail('unknown output');
  const v=m.render;
  if(v.kind!=='volume-buffer-v1' || !resources.has(v.density) || !resources.has(v.transfer) || await textDigest(v.source)!==v.sha256) fail('invalid volume description');
  if(v.dimensions.length!==3 || !v.dimensions.every(integer) || v.dimensions.reduce((a,b)=>a*b,4)!==resources.get(v.density).byte_length) fail('volume dimensions mismatch');
  if(resources.get(v.density).dtype!=='f32' || resources.get(v.transfer).dtype!=='f32' || resources.get(v.transfer).byte_length%16) fail('invalid volume/transfer type');
  const layout={byte_length:64,viewport:0,camera:16,slice:32,dimensions:48};
  if(v.vertex_entry!=='vertex_main' || v.fragment_entry!=='fragment_main' || v.topology!=='triangle-list' || v.interpolation!=='nearest' || Object.keys(v.uniform_layout).length!==Object.keys(layout).length || Object.entries(layout).some(([k,n])=>v.uniform_layout[k]!==n)) fail('unsupported volume pipeline/layout');
  const params=m.parameters; const names=new Set(); const offsets=new Set();
  if(!integer(params.rows) || !integer(params.row_stride) || params.row_stride%4 || resources.get(params.resource)?.byte_length!==params.rows*params.row_stride || params.byte_order!=='little') fail('invalid parameter buffer');
  for(const f of params.fields) {
    if(names.has(f.name) || offsets.has(f.offset) || !Number.isInteger(f.offset) || f.offset<0 || f.offset%4 || f.offset+4>params.row_stride || !['f32','i32','u32','bool'].includes(f.type) || !['live','structural'].includes(f.update)) fail('invalid parameter field');
    names.add(f.name); offsets.add(f.offset);
  }
  if(m.science.kind!=='vector-histogram/v1' || m.science.layout!=='zyx' || m.science.x.samples.length!==v.dimensions[0] || m.science.y.samples.length!==v.dimensions[1] || m.science.x.samples.length*m.science.y.samples.length!==params.rows || m.science.edges.length!==v.dimensions[2]+1) fail('scientific grid mismatch');
  if(!names.has(m.science.x.name) || !names.has(m.science.y.name) || m.science.x.name===m.science.y.name) fail('invalid parameter axes');
  if(Object.keys(m.state.parameters).length!==names.size || Object.keys(m.state.parameters).some(n=>!names.has(n))) fail('state parameter set mismatch');
  packParameters(m,m.state.parameters);
  presentation(m.state.presentation);
  return m;
}

async function shaderModule(device,source,label) {
  const module=device.createShaderModule({code:source,label});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');
  if(errors.length) fail(`${label}: ${errors.map(e=>`line ${e.lineNum}: ${e.message}`).join('; ')}`);
  return module;
}

export async function fetchPackage(url) {
  const base=new URL(url,globalThis.location?.href);
  const response=await fetch(base); if(!response.ok) fail(`manifest HTTP ${response.status}`);
  const manifest=await response.json();
  if(manifest.payload?.file!=='payload.bin') fail('unsupported payload reference');
  const data=await fetch(new URL('payload.bin',base)); if(!data.ok) fail(`payload HTTP ${data.status}`);
  const payload=await data.arrayBuffer(); await validatePackage(manifest,payload);
  return {manifest,payload};
}

export function checkCapabilities(m, capabilities) {
  for(const f of m.required_features) if(!capabilities.features.has(f)) fail(`missing GPU feature ${f}`);
  for(const [name,value] of Object.entries(m.required_limits)) if(!integer(value) || capabilities.limits[name]===undefined || capabilities.limits[name]<value) fail(`GPU limit ${name}: requires ${value}, available ${capabilities.limits[name]}`);
  for(const p of m.passes) {
    const w=m.shaders[p.shader].reflection.workgroup_size;
    for(let i=0;i<3;i++) if(w[i]>capabilities.limits[['maxComputeWorkgroupSizeX','maxComputeWorkgroupSizeY','maxComputeWorkgroupSizeZ'][i]]) fail('compute workgroup size unsupported');
    if(w.reduce((a,b)=>a*b,1)>capabilities.limits.maxComputeInvocationsPerWorkgroup) fail('workgroup invocation limit exceeded');
    if(p.workgroups.some(n=>n>capabilities.limits.maxComputeWorkgroupsPerDimension)) fail('dispatch exceeds device limit');
  }
  for(const r of m.resources) if(r.byte_length>capabilities.limits.maxBufferSize || r.byte_length>capabilities.limits.maxStorageBufferBindingSize) fail(`buffer ${r.id} exceeds actual device limits`);
}

function validateValue(f,value) {
  if(f.type==='bool') { if(typeof value!=='boolean') fail(`${f.name} requires a boolean`); return value; }
  if(typeof value!=='number' || !Number.isFinite(value)) fail(`${f.name} requires a finite number`);
  if(f.range && (value<f.range[0] || value>f.range[1])) fail(`${f.name} outside declared range`);
  if(f.type==='f32' && !Number.isFinite(Math.fround(value))) fail(`${f.name} exceeds f32`);
  if(f.type!=='f32' && (!Number.isInteger(value) || value<(f.type==='u32'?0:-2147483648) || value>(f.type==='u32'?4294967295:2147483647))) fail(`${f.name} exceeds ${f.type}`);
  return value;
}

export function packParameters(m,values) {
  const p=m.parameters; const bytes=new ArrayBuffer(p.rows*p.row_stride); const view=new DataView(bytes);
  const nx=m.science.x.samples.length;
  for(let row=0;row<p.rows;row++) for(const f of p.fields) {
    let value=values[f.name];
    if(f.name===m.science.x.name) value=m.science.x.samples[row%nx];
    if(f.name===m.science.y.name) value=m.science.y.samples[Math.floor(row/nx)];
    validateValue(f,value); const offset=row*p.row_stride+f.offset;
    if(f.type==='f32') view.setFloat32(offset,value,true);
    else if(f.type==='i32') view.setInt32(offset,value,true);
    else view.setUint32(offset,f.type==='bool'?Number(value):value,true);
  }
  return bytes;
}

function presentation(value) {
  const keys=['yaw','pitch','radius','opacity','maximum','mode','slice_z'];
  if(Object.keys(value).some(k=>!keys.includes(k))) fail('unknown presentation field');
  for(const k of keys.filter(k=>k!=='mode')) if(!Number.isFinite(value[k])) fail(`invalid presentation ${k}`);
  if(Math.abs(value.pitch)>1.5 || value.radius<2 || value.radius>100 || value.opacity<0 || value.opacity>10 || value.maximum<=0 || value.slice_z<0 || value.slice_z>1 || !['volume','slice'].includes(value.mode)) fail('presentation settings out of range');
  return value;
}

export class PortableRuntime {
  static async create(data,{device,canvas=null,format=canvas?navigator.gpu.getPreferredCanvasFormat():'rgba8unorm'}={}) {
    await validatePackage(data.manifest,data.payload);
    if(!['rgba8unorm','bgra8unorm'].includes(format)) fail('unsupported presentation format');
    let ownsDevice=false;
    if(!device) {
      if(!navigator.gpu) fail('WebGPU is unavailable in this host');
      const adapter=await navigator.gpu.requestAdapter();
      if(!adapter) fail('This browser or embedded preview is not exposing a usable WebGPU adapter. The page cannot enable GPU support itself. Try an external browser with WebGPU enabled; in Chrome, inspect chrome://gpu for the reason.');
      checkCapabilities(data.manifest,adapter);
      device=await adapter.requestDevice({requiredFeatures:data.manifest.required_features,requiredLimits:data.manifest.required_limits}); ownsDevice=true;
    }
    checkCapabilities(data.manifest,device);
    const runtime=new PortableRuntime(data,device,canvas,ownsDevice);
    runtime.format=format;
    device.pushErrorScope('validation');
    let failure=null;
    try { await runtime.initialize(); } catch(error) { failure=error; }
    const gpuError=await device.popErrorScope();
    if(failure || gpuError) { runtime.close(); throw failure || new Error(gpuError.message); }
    return runtime;
  }

  constructor(data,device,canvas,ownsDevice) {
    this.manifest=clone(data.manifest); this.payload=data.payload; this.device=device; this.canvas=canvas;
    this.ownsDevice=ownsDevice; this.buffers=new Map(); this.pipelines=[]; this.closed=false;
    this.values=clone(this.manifest.state.parameters); this.presentation=presentation(clone(this.manifest.state.presentation));
    this.lost=null; device.lost.then(info=>{this.lost=info.message;});
  }
  requireOpen() { if(this.closed || this.lost) fail(this.lost || 'runtime closed'); }
  async initialize() {
    const d=this.device,m=this.manifest;
    // Validate default state before any allocation.
    packParameters(m,this.values);
    for(const r of m.resources) {
      const buffer=d.createBuffer({label:r.id,size:r.byte_length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
      this.buffers.set(r.id,buffer);
      if(r.initial) d.queue.writeBuffer(buffer,0,this.payload,r.initial.offset,r.initial.length);
    }
    for(const p of m.passes) {
      const s=m.shaders[p.shader]; const module=await shaderModule(d,s.source,p.id);
      const pipeline=await d.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:s.reflection.entry_point}});
      const bindings=d.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:p.bindings.map(b=>({binding:b.binding,resource:{buffer:this.buffers.get(b.resource)}}))});
      this.pipelines.push({pipeline,bindings,workgroups:p.workgroups});
    }
    this.uniform=d.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const module=await shaderModule(d,m.render.source,'volume');
    this.renderPipeline=await d.createRenderPipelineAsync({layout:'auto',vertex:{module,entryPoint:m.render.vertex_entry},fragment:{module,entryPoint:m.render.fragment_entry,targets:[{format:this.format}]},primitive:{topology:m.render.topology}});
    this.renderBindings=d.createBindGroup({layout:this.renderPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.buffers.get(m.render.density)}},
      {binding:1,resource:{buffer:this.uniform}},
      {binding:2,resource:{buffer:this.buffers.get(m.render.transfer)}}]});
    if(this.canvas) { this.context=this.canvas.getContext('webgpu'); if(!this.context) fail('canvas WebGPU context unavailable'); this.context.configure({device:d,format:this.format,alphaMode:'opaque',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_DST}); }
  }
  compute() {
    this.requireOpen(); const e=this.device.createCommandEncoder();
    for(const step of this.pipelines) { const p=e.beginComputePass(); p.setPipeline(step.pipeline); p.setBindGroup(0,step.bindings); p.dispatchWorkgroups(...step.workgroups); p.end(); }
    this.device.queue.submit([e.finish()]);
  }
  setParameters(updates) {
    this.requireOpen(); const fields=new Map(this.manifest.parameters.fields.map(f=>[f.name,f]));
    const values={...this.values};
    for(const [name,value] of Object.entries(updates)) {
      const f=fields.get(name); if(!f) fail(`unknown parameter ${name}`); validateValue(f,value);
      if(f.update==='structural' && value!==values[name]) { const error=new Error(`Structural edit requires preparation: ${name}`); error.code='reprepare-required'; throw error; }
      if([this.manifest.science.x.name,this.manifest.science.y.name].includes(name) && value!==values[name]) fail('axis values come from the exported grid; replace preparation to change axes');
      values[name]=value;
    }
    const bytes=packParameters(this.manifest,values);
    this.device.queue.writeBuffer(this.buffers.get(this.manifest.parameters.resource),0,bytes); this.values=values; this.compute();
  }
  setPresentation(updates) { this.requireOpen(); this.presentation=presentation({...this.presentation,...updates}); }
  render(width=this.canvas?.width||256,height=this.canvas?.height||256) {
    this.requireOpen(); if(!integer(width)||!integer(height)||Math.max(width,height)>this.device.limits.maxTextureDimension2D) fail('invalid viewport');
    const d=this.device;
    if(width!==this.width || height!==this.height) { this.output?.destroy(); this.output=d.createTexture({size:[width,height],format:this.format,usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_SRC}); this.width=width; this.height=height; }
    const p=this.presentation, dims=this.manifest.render.dimensions;
    d.queue.writeBuffer(this.uniform,0,new Float32Array([width,height,p.opacity,p.maximum,p.yaw,p.pitch,p.radius,.0125,p.mode==='slice'?1:0,p.slice_z,0,0,...dims,0]));
    const e=d.createCommandEncoder(); const pass=e.beginRenderPass({colorAttachments:[{view:this.output.createView(),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});
    pass.setPipeline(this.renderPipeline); pass.setBindGroup(0,this.renderBindings); pass.draw(3); pass.end();
    if(this.context) e.copyTextureToTexture({texture:this.output},{texture:this.context.getCurrentTexture()},[width,height]);
    d.queue.submit([e.finish()]);
  }
  async read(name) {
    this.requireOpen(); const id=this.manifest.outputs[name]||name; const source=this.buffers.get(id); if(!source) fail(`unknown read resource ${name}`);
    const size=this.manifest.resources.find(r=>r.id===id).byte_length;
    const staging=this.device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try { const e=this.device.createCommandEncoder();e.copyBufferToBuffer(source,0,staging,0,size);this.device.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);return staging.getMappedRange().slice(0); }
    finally { staging.destroy(); }
  }
  async readPixels() {
    this.requireOpen(); if(!this.output) fail('render before capture');
    const stride=Math.ceil(this.width*4/256)*256;
    const staging=this.device.createBuffer({size:stride*this.height,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try { const e=this.device.createCommandEncoder(); e.copyTextureToBuffer({texture:this.output},{buffer:staging,bytesPerRow:stride},[this.width,this.height]);this.device.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);const mapped=new Uint8Array(staging.getMappedRange()), pixels=new Uint8Array(this.width*this.height*4);for(let y=0;y<this.height;y++) pixels.set(mapped.subarray(y*stride,y*stride+this.width*4),y*this.width*4);if(this.format==='bgra8unorm')for(let i=0;i<pixels.length;i+=4)[pixels[i],pixels[i+2]]=[pixels[i+2],pixels[i]];return pixels; }
    finally {staging.destroy();}
  }
  snapshot() { this.requireOpen(); return {schema:'ordinarylight/view-state-v1',package_id:this.manifest.id,parameters:clone(this.values),presentation:clone(this.presentation)}; }
  restore(snapshot) {
    this.requireOpen(); if(snapshot.schema!=='ordinarylight/view-state-v1'||snapshot.package_id!==this.manifest.id) fail('snapshot package/version mismatch');
    if(Object.keys(snapshot.parameters).sort().join('\0')!==Object.keys(this.values).sort().join('\0')) fail('snapshot parameter set mismatch');
    const view=presentation(clone(snapshot.presentation)); // validate everything before writes
    this.setParameters(snapshot.parameters); this.presentation=view;
  }
  close() { if(this.closed)return; this.closed=true;for(const b of this.buffers.values())b.destroy();this.uniform?.destroy();this.output?.destroy();if(this.ownsDevice)this.device.destroy(); }
}

export class PortableViewer {
  static async create(data,options={}) { return new PortableViewer(await PortableRuntime.create(data,options),options.onReprepare); }
  constructor(runtime,onReprepare) { this.runtime=runtime;this.onReprepare=onReprepare;this.busy=false; }
  async update(changes) {
    if(this.busy) fail('another replacement is in progress');
    if(changes.preparation || Object.entries(changes.parameters||{}).some(([n,v])=>this.runtime.manifest.parameters.fields.find(f=>f.name===n)?.update==='structural' && v!==this.runtime.values[n])) {
      if(!this.onReprepare) fail('structural edit requires a host preparation callback');
      this.busy=true;
      try { const data=await this.onReprepare(this.runtime.snapshot(),clone(changes));await this.replace(data); }
      finally {this.busy=false;}
    } else { if(changes.parameters)this.runtime.setParameters(changes.parameters); }
    if(changes.presentation)this.runtime.setPresentation(changes.presentation);
    this.runtime.render();
  }
  async replace(data) {
    const old=this.runtime;
    const next=await PortableRuntime.create(data,{device:old.device,format:old.format});
    next.canvas=old.canvas; next.context=old.context;
    try {next.compute();next.render();await next.device.queue.onSubmittedWorkDone();}
    catch(error){next.close();throw error;}
    next.ownsDevice=old.ownsDevice;old.ownsDevice=false;this.runtime=next;old.close();
  }
  close() {this.runtime.close();}
}
