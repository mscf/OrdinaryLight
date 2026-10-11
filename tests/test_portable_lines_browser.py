"""Optional real WebGPU pixel test for the generic stroke renderer."""
import os
import subprocess
from pathlib import Path
import pytest


def test_portable_lines(tmp_path):
    playwright=pytest.importorskip('playwright.sync_api')
    python=os.environ.get('ORDINARYLIGHT_PYTHON')
    if not python:pytest.skip('Set ORDINARYLIGHT_PYTHON to an interpreter with the updated package')
    target=tmp_path/'lines.html'
    subprocess.run([python,'-c',r'''
import json,sys
from pathlib import Path
from ordinarylight.portable.surface import line_assets
assets=line_assets()
source=assets['runtime'].replace('export class OrdinaryLightSurface','class OrdinaryLightSurface')+'\n'+assets['lines'].replace('export class OrdinaryLightLines','class OrdinaryLightLines')
Path(sys.argv[1]).write_text('<canvas style="width:400px;height:200px"></canvas><script type="module">'+source+'\nwindow.renderer=new OrdinaryLightLines(document.querySelector("canvas"),'+json.dumps(assets['shader'])+');</script>')
''',str(target)],check=True,capture_output=True,text=True)
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--enable-unsafe-webgpu','--enable-unsafe-swiftshader','--use-angle=swiftshader','--enable-features=Vulkan','--use-vulkan=swiftshader'])
        page=browser.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(target.as_uri());page.wait_for_function('!!window.renderer')
        page.evaluate('renderer.setData([{segments:[[-1,.5,.3,.5],[.7,.5,2,.5]],color:[1,0,0],width:4,dash:0},{segments:[[.5,0,.5,1]],color:[0,0,1],width:3,dash:6}],[1,1,1,1])')
        pixels=page.evaluate('''async()=>{const p=await renderer.runtime.capture(1);const at=(x,y)=>Array.from(p.data.slice((y*p.width+x)*4,(y*p.width+x)*4+4));return {red:at(50,100),gap:at(160,100),blue:Array.from({length:200},(_,y)=>at(200,y)).filter(p=>p[2]>240&&p[0]<20).length};}''')
        assert pixels['red']==[255,0,0,255]
        assert pixels['gap']==[255,255,255,255]
        assert 60<pixels['blue']<140 # dashed line, not a solid stroke
        page.evaluate('document.querySelector("canvas").style.width="300px"')
        page.wait_for_function('renderer.runtime.width===300')
        page.evaluate('renderer.setData([],[0,0,0,1])')
        assert page.evaluate('renderer.runtime.indexCount')==0
        assert page.evaluate('''()=>{try{renderer.setData([{segments:[[0,NaN,1,1]],color:[1,0,0],width:2}],[0,0,0,1]);return false;}catch{return true;}}''')
        page.evaluate('renderer.dispose()');assert page.evaluate('renderer.closed && renderer.runtime.closed')
        assert not errors,errors
        browser.close()
