/* Generic screen-space WebGPU strokes, using the existing OrdinaryShade scalar shader.
 * Segments use normalized [0,1] coordinates, with Y increasing upward. */
export class OrdinaryLightLines {
  constructor(canvas,shader,{onError=()=>{}}={}){
    this.canvas=canvas;this.closed=false;this.series=[];this.queue=Promise.resolve();
    this.runtime=new OrdinaryLightSurface(canvas,shader,{onError});
    this.runtime.events.abort();this.runtime.observer.disconnect();
    this.runtime.projection=()=>[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];
    this.observer=new ResizeObserver(()=>{this.update().catch(onError);});this.observer.observe(canvas);
  }
  setData(series,background){
    if(this.closed)throw new Error('Line renderer is disposed');
    for(const s of series){
      if(!Array.isArray(s.segments)||s.segments.some(p=>p.length!==4||!p.every(Number.isFinite))||
         !Array.isArray(s.color)||s.color.length!==3||s.color.some(v=>!Number.isFinite(v)||v<0||v>1)||
         !Number.isFinite(s.width)||s.width<=0||!Number.isFinite(s.dash??0)||(s.dash??0)<0)throw new Error('Invalid line series');
    }
    this.hasData=true;this.series=structuredClone(series);this.runtime.setAppearance({background});return this.update();
  }
  update(){
    this.queue=this.queue.catch(()=>{}).then(async()=>{
      if(this.closed||!this.hasData)return;
      const rect=this.canvas.getBoundingClientRect(),w=Math.max(1,rect.width),h=Math.max(1,rect.height);
      const positions=[],scalars=[],indices=[],stops=[];
      const count=Math.max(2,this.series.length);
      for(let i=0;i<count;i++)stops.push(...(this.series[i]?.color??this.series[0]?.color??[0,0,0]),i/(count-1));
      for(let k=0;k<this.series.length;k++){
        const s=this.series[k];
        for(const [x0,y0,x1,y1] of s.segments){
          // Clip the centerline before tessellation; never clamp endpoints into a fake line.
          const dx=x1-x0,dy=y1-y0;let lo=0,hi=1;
          for(const [p,q] of [[-dx,x0],[dx,1-x0],[-dy,y0],[dy,1-y0]]){
            if(p===0){if(q<0){hi=-1;break;}continue;}
            const t=q/p;if(p<0)lo=Math.max(lo,t);else hi=Math.min(hi,t);
          }
          if(lo>=hi)continue;
          const length=Math.hypot(dx*w,dy*h);if(!length)continue;
          const nx=-dy*h/length*s.width/2/w,ny=dx*w/length*s.width/2/h;
          const step=s.dash?s.dash/length:hi-lo;
          for(let t=lo;t<hi;t+=s.dash?step*2:hi-lo){
            const end=Math.min(hi,t+step),a=[x0+dx*t,y0+dy*t],b=[x0+dx*end,y0+dy*end],offset=scalars.length;
            for(const [x,y] of [[a[0]+nx,a[1]+ny],[a[0]-nx,a[1]-ny],[b[0]+nx,b[1]+ny],[b[0]-nx,b[1]-ny]]){positions.push(x*2-1,y*2-1,.5);scalars.push(k/(count-1));}
            indices.push(offset,offset+1,offset+2,offset+2,offset+1,offset+3);
          }
        }
      }
      await this.runtime.setMesh({positions,scalars,indices,stops,clipBounds:[[-1,1],[-1,1],[0,1]]});
    });return this.queue;
  }
  dispose(){this.closed=true;this.observer.disconnect();this.runtime.dispose();}
}
