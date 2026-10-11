"""Projected handles reject occlusion and distinguish coincident targets."""
from pathlib import Path
import pytest


def test_visible_handles(tmp_path):
    playwright=pytest.importorskip('playwright.sync_api')
    root=Path(__file__).parents[1]/'ordinarylight/portable/web'
    source=(root/'surface-runtime.js').read_text()+(root/'drag-controller.js').read_text()
    target=tmp_path/'pick.html'
    target.write_text('<canvas width="500" height="500" style="width:500px;height:500px"></canvas><script type="module">'+source+r"""
const canvas=document.querySelector('canvas'),r=Object.create(OrdinaryLightSurface.prototype);
r.canvas=canvas;r.camera={eye:{x:0,y:0,z:3},center:{x:0,y:0,z:0},up:{x:0,y:1,z:0}};
r.positions=[-1,-1,0,1,-1,0,0,1,0];r.indices=[0,1,2];
window.events=[];window.hover=null;
window.controller=new OrdinaryLightDragController(canvas,{project:p=>r.project(p),isVisible:p=>r.isVisible(p),targets:()=>[
 {id:'hidden',direction:[0,1,0],lines:[[[-.3,0,-.2],[.3,0,-.2]]]},
 {id:'lower',direction:[0,1,0],lines:[[[-.3,0,0],[.3,0,0]]]},
 {id:'upper',direction:[0,1,0],lines:[[[-.3,0,0],[.3,0,0]]]}
],onHover:h=>window.hover=h?.target.id??null,onDrag:e=>events.push({phase:e.phase,id:e.target.id})});
</script>""")
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True)
        page=browser.new_page();page.goto(target.as_uri())
        handles=page.evaluate('controller.handles()')
        assert [h['target']['id'] for h in handles]==['lower','upper']
        assert ((handles[0]['x']-handles[1]['x'])**2+(handles[0]['y']-handles[1]['y'])**2)**.5>=22
        rect=page.locator('canvas').bounding_box()
        for h in handles:
            page.mouse.move(rect['x']+h['x'],rect['y']+h['y'])
            assert page.evaluate('hover')==h['target']['id']
            page.mouse.down();page.mouse.move(rect['x']+h['x'],rect['y']+h['y']+10);page.mouse.up()
            assert page.evaluate('events.at(-1)')=={'phase':'end','id':h['target']['id']}
        page.mouse.move(700,700);assert page.evaluate('hover') is None
        page.evaluate('controller.dispose()');browser.close()
