"""Compare volume-free secondary shading with unspecialized shading.

Writes timings to /tmp/secondary-surface-final-profile.json and HDR arrays to /tmp/secondary-surface-final-*.npy.
"""
from dataclasses import replace
import json
import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.showcases.catalog.raster import SHOWCASES


def main():
    item=next(s for s in SHOWCASES if s.id=='optical-screen-rough-reflection')
    scene=item.build(); camera=item.camera.camera(scene,angle=-0.45)
    glfw=load_glfw(); assert glfw.init()
    glfw.window_hint(glfw.CLIENT_API,glfw.NO_API)
    window=glfw.create_window(2052,1764,'Secondary GPU comparison',None,None)
    config=_gi_config(item,shared_primary=True,path_spp=2,restir_reservoirs=2,ray_batch_capacity=524288)
    results={}; images={}
    try:
     for name,changes in [('baseline',{'wavefront_scene_specialization':False}),('repeat',{'wavefront_scene_specialization':False}),('surface_only',{})]:
      print('START',name,flush=True)
      rows=[]
      with ol.VulkanGlfwPresenter(window,config=replace(config,wavefront_hdr_capture=True,**changes)) as renderer:
       for i in range(64):
        glfw.poll_events(); renderer.present_wavefront(scene,camera,2052,1764)
        if i>=32: rows.append(dict(renderer.last_timings))
       images[name]=renderer.capture_wavefront_hdr().copy(); np.save('/tmp/secondary-surface-final-'+name+'.npy',images[name])
      labels=rows[-1]['wavefront_stage_ms'].keys()
      results[name]={'gpu_ms':float(np.median([r['gpu_frame_ms'] for r in rows])), 'stages':{k:float(np.median([r['wavefront_stage_ms'].get(k,0) for r in rows])) for k in labels},'tiles':rows[-1]['wavefront_tiles'],'hdr_mean':float(images[name][...,:3].mean()), 'hdr_rmse':float(np.sqrt(np.mean((images[name]-images['baseline'])**2))), 'hdr_max_difference':float(np.max(np.abs(images[name]-images['baseline'])))}
      print(name,json.dumps(results[name]),flush=True)
    finally:
     glfw.destroy_window(window); glfw.terminate()
    assert np.array_equal(images['repeat'], images['surface_only'])
    assert all(np.isfinite(image).all() for image in images.values())
    open('/tmp/secondary-surface-final-profile.json','w').write(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
