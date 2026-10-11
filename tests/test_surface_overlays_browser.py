"""Real GPU checks for generic overlay blending, occlusion and cleanup."""
import os
import subprocess
from pathlib import Path
import pytest


def test_surface_overlays(tmp_path):
    playwright=pytest.importorskip('playwright.sync_api')
    python=os.environ.get('ORDINARYLIGHT_PYTHON')
    if not python:pytest.skip('Set ORDINARYLIGHT_PYTHON')
    target=tmp_path/'overlays.html'
    subprocess.run([python,'-c',r'''
import json,sys
from pathlib import Path
from ordinarylight.portable.surface import surface_assets
assets=surface_assets();source=assets['runtime'].replace('export class OrdinaryLightSurface','class OrdinaryLightSurface')
Path(sys.argv[1]).write_text('<canvas style="width:400px;height:200px"></canvas><script type="module">'+source+'\nwindow.renderer=new OrdinaryLightSurface(document.querySelector("canvas"),'+json.dumps(assets['shader'])+');renderer.projection=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];</script>')
''',str(target)],check=True,capture_output=True,text=True)
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--enable-unsafe-webgpu','--enable-unsafe-swiftshader','--use-angle=swiftshader','--enable-features=Vulkan','--use-vulkan=swiftshader'])
        page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(target.as_uri());page.wait_for_function('!!window.renderer')
        page.evaluate('renderer.setMesh({positions:[-1,-1,.4,1,-1,.4,-1,1,.4,1,1,.4],indices:[0,1,2,1,3,2],scalars:[0,0,0,0],stops:[0,0,1,0,0,0,1,1]})')
        before=page.evaluate('JSON.stringify([renderer.positions,renderer.indices])')
        page.evaluate('renderer.setOverlays([{positions:[-1,-1,.4,0,-1,.4,-1,1,.4,0,1,.4],indices:[0,1,2,1,3,2],lines:[-.5,-1,.4,-.5,1,.4],color:[1,0,0],opacity:.5},{positions:[0,-1,.6,1,-1,.6,0,1,.6,1,1,.6],indices:[0,1,2,1,3,2],lines:[],color:[0,1,0],opacity:1}])')
        result=page.evaluate('''async()=>{const p=await renderer.capture(1),at=x=>Array.from(p.data.slice((100*p.width+x)*4,(100*p.width+x)*4+4));return {left:at(60),right:at(300)};}''')
        assert 125<=result['left'][0]<=130 and 125<=result['left'][2]<=130
        assert result['right']==[0,0,255,255]
        assert page.evaluate('JSON.stringify([renderer.positions,renderer.indices])')==before
        page.evaluate('renderer.setOverlays([])');assert page.evaluate('renderer.overlayLayers.length')==0
        page.evaluate('renderer.dispose()');assert page.evaluate('renderer.closed')
        assert not errors,errors
        browser.close()
