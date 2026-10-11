"""Optional Chrome/SwiftShader conformance test: python -m pytest this_file.

Requires playwright and Chrome. Shader code is the packaged OrdinaryShade artifact.
"""
from pathlib import Path
import json
import pytest


def test_portable_surface_pixels_picking_and_lifecycle(tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    root = Path(__file__).resolve().parents[1] / 'ordinarylight' / 'portable'
    runtime = (root / 'web/surface-runtime.js').read_text().replace('export class OrdinaryLightSurface', 'class OrdinaryLightSurface')
    shader = json.dumps((root / 'surface.wgsl').read_text())
    document = tmp_path / 'surface.html'
    document.write_text('<canvas style="width:256px;height:256px"></canvas><script type="module">' + runtime + '\nconst shader=' + shader + ''';
    window.errors=[];
    const r=window.renderer=new OrdinaryLightSurface(document.querySelector('canvas'),shader,{onError:e=>errors.push(e.message)});
    r.camera={eye:{x:0,y:0,z:3},center:{x:0,y:0,z:0},up:{x:0,y:1,z:0}};
    window.mesh={positions:[-.5,-.5,0,.5,-.5,0,.5,.5,0,-.5,.5,0],scalars:[.25,.25,.25,.25],indices:[0,1,2,0,2,3],stops:[0,0,0,0,1,1,1,1]};
    window.ready=r.setMesh(mesh);
    window.pixel=async(x=128,y=128,width=1)=>{
      await r.device.queue.onSubmittedWorkDone();
      const b=r.device.createBuffer({size:1024,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const e=r.device.createCommandEncoder();
      e.copyTextureToBuffer({texture:r.color,origin:[x,y]},{buffer:b,bytesPerRow:1024},[width,1]);
      r.device.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
      const p=Array.from(new Uint8Array(b.getMappedRange()).slice(0,4*width));b.unmap();b.destroy();return p;
    };
    </script>''')
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--enable-features=Vulkan', '--use-vulkan=swiftshader'])
        page = browser.new_page(device_scale_factor=1)
        page.goto(document.as_uri())
        page.evaluate('window.ready')
        assert page.evaluate('pixel()') == [64, 64, 64, 255]
        row=page.evaluate('pixel(0,128,256)')
        assert any(64<row[i]<255 and row[i]==row[i+1]==row[i+2] for i in range(0,len(row),4)), 'Expected partially covered surface edge pixels'
        assert page.evaluate('renderer.pick(128,128)') in (0, 1, 2, 3)
        assert page.evaluate('renderer.pick(0,0)') is None
        assert page.evaluate('async()=>{try{await renderer.setMesh({...mesh,scalars:[NaN,0,0,0]});return false;}catch(e){return /Invalid/.test(e.message);}}')
        before = page.evaluate('renderer.project([.3,.3,0])')
        page.evaluate("renderer.canvas.style.height='512px';renderer.draw()")
        after = page.evaluate('renderer.project([.3,.3,0])')
        assert abs(before['x']-after['x']) < 1e-6
        assert page.evaluate('(p)=>renderer.pick(p.x,p.y)', after) == 2
        page.evaluate("renderer.canvas.style.height='256px';renderer.draw()")
        page.evaluate('renderer.setLines([-.5,0,-.2,.5,0,-.2],[1,0,0]);renderer.draw()')
        assert page.evaluate('pixel()') == [64, 64, 64, 255]
        assert page.evaluate('pixel(128,127)') == [64, 64, 64, 255]
        page.evaluate('renderer.setLines([-.5,0,.2,.5,0,.2],[1,0,0]);renderer.draw()')
        samples = page.evaluate('Promise.all([pixel(),pixel(128,127)])')
        # MSAA may blend a thin red line with the gray surface at its edge.
        assert any(max(pixel[0],pixel[2])>64 and pixel[1]<64 for pixel in samples)
        page.evaluate('renderer.setLines([]);renderer.setMesh({...mesh,clipBounds:[[-.2,.2],[-.2,.2],[-.5,.5]]})')
        assert page.evaluate('pixel()') == [64,64,64,255]
        # All original vertices are outside this crop. Rendering intersections are not observations.
        assert page.evaluate('renderer.pick(128,128)') is None
        outside=page.evaluate('renderer.project([.35,0,0])')
        assert page.evaluate('(p)=>pixel(Math.round(p.x),Math.round(p.y))',outside)==[255,255,255,255]
        page.evaluate('renderer.setMesh({...mesh,clipBounds:[[-.5,0],[-.5,.5],[-.5,.5]]})')
        inside=page.evaluate('renderer.project([-.3,-.3,0])')
        assert page.evaluate('(p)=>renderer.pick(p.x,p.y)',inside) in (0,3)
        assert page.evaluate('(p)=>renderer.pick(p.x,p.y)',outside) is None
        page.evaluate('renderer.setMesh({...mesh,clipBounds:[[2,3],[-1,1],[-1,1]]})')
        assert page.evaluate('pixel()') == [255,255,255,255]
        page.evaluate('renderer.setLines([]);renderer.setMesh({...mesh,indices:[]})')
        assert page.evaluate('pixel()') == [255, 255, 255, 255]
        assert page.evaluate('renderer.pick(128,128)') is None
        assert page.evaluate('errors') == []
        page.evaluate("renderer.canvas.style.width='257px'")
        page.evaluate('renderer.setMesh({...mesh,stops:[1,0,0,0,1,0,0,1]})')
        capture=page.evaluate('async()=>{const p=await renderer.capture(2);const i=(Math.floor(p.height/2)*p.width+Math.floor(p.width/2))*4;return {width:p.width,height:p.height,pixel:Array.from(p.data.slice(i,i+4))}}')
        assert capture=={'width':514,'height':512,'pixel':[255,0,0,255]}
        assert page.evaluate('renderer.width')==257
        assert page.evaluate('async()=>{try{await renderer.capture(5);return false}catch(e){return /scale/.test(e.message)}}')
        assert page.evaluate('errors')==[]
        page.evaluate('renderer.setAppearance({background:[16/255,32/255,48/255,1]})')
        assert page.evaluate('async()=>Array.from((await renderer.capture(1)).data.slice(0,4))')==[16,32,48,255]
        page.evaluate('renderer.device.destroy()')
        page.wait_for_function('errors.length===1')
        assert 'device lost' in page.evaluate('errors[0]')
        page.evaluate('renderer.dispose();renderer.dispose()')
        assert page.evaluate('renderer.closed')
        browser.close()
