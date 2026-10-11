/* Generic projected-segment picking and constrained dragging. Coordinates are caller-owned. */
export class OrdinaryLightDragController {
  constructor(element,{targets,project,onDrag,isVisible=()=>true,onHover=()=>{}}){
    this.abort=new AbortController();this.active=null;
    if(!element.hasAttribute("tabindex"))element.tabIndex=0;
    const options={capture:true,signal:this.abort.signal};
    const local=e=>{const r=element.getBoundingClientRect();return [e.clientX-r.left,e.clientY-r.top];};
    const stop=e=>{e.preventDefault();e.stopImmediatePropagation();};
    // Perspective-correct interpolation maps a projected segment back to physical space.
    const anchorAt=(line,a,b,t)=>{const u=a.w&&b.w?(t/b.w)/((1-t)/a.w+t/b.w):t;return line[0].map((v,i)=>v+u*(line[1][i]-v));};
    this.handles=()=>{
      const result=[],rect=element.getBoundingClientRect();
      for(const target of targets()){
        let best=null;
        const segments=target.lines.map(line=>{const a=project(line[0]),b=project(line[1]);return a&&b?{line,a,b,length:Math.hypot(b.x-a.x,b.y-a.y)}:null;}).filter(Boolean).sort((a,b)=>b.length-a.length);
        for(const {line,a,b,length} of segments){
          for(const t of [.5,.25,.75]){const anchor=anchorAt(line,a,b,t);if(!isVisible(anchor))continue;
            best={target,anchor,length,x:a.x+t*(b.x-a.x),y:a.y+t*(b.y-a.y)};break;
          }
          if(best)break;
        }
        if(!best)continue;
        const source={x:best.x,y:best.y};
        for(let k=0;k<100;k++){
          const radius=k?10+5*Math.sqrt(k):0,angle=k*2.399963;
          const x=source.x+radius*Math.cos(angle),y=source.y+radius*Math.sin(angle);
          if(x<9||y<9||x>rect.width-9||y>rect.height-9||result.some(h=>Math.hypot(h.x-x,h.y-y)<22))continue;
          result.push({...best,x,y,source});break;
        }
      }return result;
    };
    const pick=e=>{
      const [x,y]=local(e);let best=null,distance=8;
      const handle=this.handles().filter(h=>Math.hypot(h.x-x,h.y-y)<=9).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0];
      if(handle)return handle;
      for(const target of targets())for(const line of target.lines){
        const a=project(line[0]),b=project(line[1]);if(!a||!b)continue;
        const dx=b.x-a.x,dy=b.y-a.y,t=Math.max(0,Math.min(1,((x-a.x)*dx+(y-a.y)*dy)/(dx*dx+dy*dy||1)));
        const d=Math.hypot(x-a.x-t*dx,y-a.y-t*dy);
        const anchor=anchorAt(line,a,b,t);
        if(d<distance&&isVisible(anchor)){distance=d;best={target,anchor};}
      }return best;
    };
    const finish=(e,cancel=false,consume=true)=>{
      const a=this.active;if(!a||e.pointerId!==undefined&&e.pointerId!==a.id)return;
      if(consume)stop(e);this.active=null;element.style.cursor='';onHover(null);
      if(a.id!==null&&element.hasPointerCapture(a.id))element.releasePointerCapture(a.id);
      onDrag({phase:cancel?'cancel':'end',target:a.target,delta:a.delta});
    };
    this.cancel=()=>finish({},true,false);
    // Bind stable HTML controls; callers own labels, units and step semantics.
    this.bindKeyboard=(control,{target,step=.01,onFocus=()=>{}})=>{
      control.addEventListener('focus',()=>onFocus(target()),options);
      control.addEventListener('keydown',e=>{
        if(e.altKey||e.ctrlKey||e.metaKey)return;
        if(['Enter','Tab'].includes(e.key)){
          if(this.active?.control===control)finish(e,false,e.key!=='Tab');return;
        }
        const direction={ArrowLeft:-1,ArrowDown:-1,ArrowRight:1,ArrowUp:1}[e.key];
        if(!direction)return;
        if(this.active&&this.active.control!==control)return;
        const current=target();if(!current)return;stop(e);
        if(!this.active){this.active={control,target:current,id:null,delta:0};onDrag({phase:'start',target:current,delta:0});}
        this.active.delta+=direction*step*(e.shiftKey?.1:1);
        onDrag({phase:'move',target:this.active.target,delta:this.active.delta});
      },options);
      control.addEventListener('blur',e=>{if(this.active?.control===control)finish(e,false,false);onFocus(null);},options);
    };
    element.addEventListener('pointerdown',e=>{
      if(e.button!==0||this.active)return;const hit=pick(e);if(!hit)return;
      const a=project(hit.anchor),b=project(hit.anchor.map((v,i)=>v+hit.target.direction[i]));
      if(!a||!b||Math.hypot(b.x-a.x,b.y-a.y)<2)return; // edge-on: leave numeric editing available
      stop(e);element.focus({preventScroll:true});this.active={...hit,id:e.pointerId,start:local(e),delta:0};element.setPointerCapture(e.pointerId);element.style.cursor='grabbing';
      onDrag({phase:'start',target:hit.target,delta:0});
    },options);
    element.addEventListener('pointermove',e=>{
      const a=this.active;if(!a){const hit=pick(e);element.style.cursor=hit?'grab':'';onHover(hit);return;}if(e.pointerId!==a.id)return;stop(e);
      const xy=local(e),origin=project(a.anchor);let delta=a.delta;
      // Solve the one-dimensional projected movement; retain perspective scaling.
      for(let i=0;i<8;i++){
        const p=project(a.anchor.map((v,j)=>v+delta*a.target.direction[j]));
        const q=project(a.anchor.map((v,j)=>v+(delta+1e-4)*a.target.direction[j]));if(!p||!q)break;
        const dx=(q.x-p.x)/1e-4,dy=(q.y-p.y)/1e-4,n=dx*dx+dy*dy;if(n<4)break;
        const step=((origin.x+xy[0]-a.start[0]-p.x)*dx+(origin.y+xy[1]-a.start[1]-p.y)*dy)/n;
        delta+=Math.max(-.25,Math.min(.25,step));if(Math.abs(step)<1e-7)break;
      }
      if(Number.isFinite(delta)){a.delta=delta;onDrag({phase:'move',target:a.target,delta});}
    },options);
    element.addEventListener('pointerleave',()=>{if(!this.active){element.style.cursor='';onHover(null);}},options);
    element.addEventListener('pointerup',e=>finish(e),options);
    element.addEventListener('pointercancel',e=>finish(e,true),options);
    element.addEventListener('lostpointercapture',e=>finish(e,true),options);
    element.ownerDocument.addEventListener('keydown',e=>{if(e.key==='Escape')finish(e,true);},options);
    element.addEventListener('wheel',e=>{if(this.active)stop(e);},{...options,passive:false});
  }
  dispose(){this.active=null;this.abort.abort();}
}
