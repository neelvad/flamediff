"""Self-contained HTML report viewer: embed a Report dict as JSON and render it with vanilla JS +
inline SVG -- no server, no build step, no dependencies, works offline. It mirrors the TUI: the
per-(table, metric) trajectory sparklines with anomaly markers, the ranked event list, and a
drill-down into each event's *why* (attribution bars + movers, or a churn breakdown)."""
from __future__ import annotations

import html as _html
import json

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>flamediff — __TITLE__</title>
<style>
:root{--base:#1e1e2e;--mantle:#181825;--surface:#313244;--surface2:#45475a;--text:#cdd6f4;
--sub:#a6adc8;--overlay:#6c7086;--mauve:#cba6f7;--red:#f38ba8;--peach:#fab387;--yellow:#f9e2af;
--mauve2:#b4befe;--blue:#89b4fa;--teal:#94e2d5;}
*{box-sizing:border-box}
body{margin:0;background:var(--base);color:var(--text);
font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:14px 20px;background:var(--mantle);border-bottom:1px solid var(--surface);
display:flex;gap:22px;align-items:baseline;flex-wrap:wrap}
header h1{margin:0;font-size:16px;color:var(--mauve)}
.meta{color:var(--sub);font-size:12px}
.worst{color:var(--red);font-weight:bold}
main{padding:16px 20px;display:flex;flex-direction:column;gap:18px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--overlay);margin:0 0 8px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}
.cell{background:var(--mantle);border:1px solid var(--surface);border-radius:6px;padding:6px 8px;cursor:pointer}
.cell:hover{border-color:var(--surface2)}
.cell .lbl{font-size:11px;display:flex;justify-content:space-between}
.cell .metric{color:var(--sub)} .cell .tbl{color:var(--overlay)}
.cell.sev5 .metric{color:var(--red)} .cell.sev2 .metric{color:var(--peach)}
.cell svg{width:100%;height:44px;display:block;margin-top:3px}
svg .spark{fill:none;stroke:var(--blue);stroke-width:1.5}
.mk{stroke:var(--base);stroke-width:1}
.mk.s5{fill:var(--red)} .mk.s2{fill:var(--peach)} .mk.s1{fill:var(--yellow)}
.hl{stroke:var(--overlay);stroke-width:1;stroke-dasharray:2 2}
.hd{fill:var(--text);stroke:var(--base);stroke-width:1}
#tip{position:fixed;pointer-events:none;display:none;z-index:10;white-space:nowrap;font-size:11px;
padding:3px 7px;border-radius:4px;background:var(--surface2);color:var(--text);border:1px solid var(--overlay)}
#bottom{display:grid;grid-template-columns:minmax(300px,1fr) minmax(320px,1.2fr);gap:18px}
#evlist{display:flex;flex-direction:column;gap:2px;max-height:64vh;overflow:auto}
.ev{display:grid;grid-template-columns:52px 1fr auto;gap:8px;padding:6px 8px;background:var(--mantle);
border-left:3px solid var(--surface2);border-radius:4px;cursor:pointer;align-items:center}
.ev:hover,.ev.sel{background:var(--surface)}
.ev.s5{border-left-color:var(--red)} .ev.s2{border-left-color:var(--peach)} .ev.s1{border-left-color:var(--yellow)}
.ev .step{color:var(--overlay);font-size:12px}
.ev .m{color:var(--sub)} .ev .sev{font-weight:bold;color:var(--peach)}
.evgrp{padding:8px 8px 2px;color:var(--overlay);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
#detailbody{background:var(--mantle);border:1px solid var(--surface);border-radius:6px;padding:14px;min-height:220px}
.tag{font-size:11px;padding:1px 7px;border-radius:9px;background:var(--surface);color:var(--sub)}
.bar{margin:8px 0}
.bar .row{display:flex;justify-content:space-between;font-size:12px;color:var(--sub)}
.bar .track{height:8px;background:var(--surface);border-radius:4px;overflow:hidden;margin-top:3px}
.fill{height:100%}.fill.g{background:var(--overlay)}.fill.p{background:var(--peach)}.fill.i{background:var(--mauve)}
.movers{margin-top:12px;color:var(--sub);font-size:12px}
.movers code,.churn code{color:var(--teal)}
.churn{color:var(--sub);font-size:12px;margin-top:10px}
.dim{color:var(--overlay);font-size:12px;margin-top:12px}
</style></head><body>
<header>
  <h1>flamediff · <span id="run"></span></h1>
  <span class="meta" id="meta"></span>
  <span class="worst" id="worst"></span>
</header>
<main>
  <section><h2>trajectory</h2><div id="grid"></div></section>
  <section id="bottom">
    <div><h2>anomalies</h2><div id="evlist"></div></div>
    <div><h2>why</h2><div id="detailbody"></div></div>
  </section>
</main>
<script id="flamediff-data" type="application/json">__DATA__</script>
<script>
const tip=document.createElement('div'); tip.id='tip'; document.body.appendChild(tip);
const sc=s=>s>=5?'s5':s>=2?'s2':'s1', cc=s=>s>=5?'sev5':s>=2?'sev2':'', tn=t=>t.replace('_emb','');
const evkey=e=>e.table+'|'+e.metric+'|'+e.step+'|'+e.method;
let DATA=null, SEL=null;
const bar=(l,f,cl)=>`<div class="bar"><div class="row"><span>${l}</span><span>${(f*100).toFixed(0)}%</span></div>`
  +`<div class="track"><div class="fill ${cl}" style="width:${Math.max(0,Math.min(1,f))*100}%"></div></div></div>`;
function detail(e){
  SEL=evkey(e); const w=e.why||{}, b=document.getElementById('detailbody');
  let h=`<div><span class="tag">${w.kind||''}</span> <b>step ${e.step}</b> · ${e.table}.${e.metric} · `
    +`<b>${e.severity.toFixed(1)}×</b> <span style="color:var(--overlay)">[${e.method}]</span></div>`
    +`<p style="color:var(--sub)">${w.text||''}</p>`;
  if(w.aligned_residual!=null){
    h+=bar('global basis drift',w.global||0,'g')+bar('popularity (r²)',w.popularity_r2||0,'p')
      +bar('idiosyncratic',w.aligned_residual||0,'i');
    if(w.top_movers&&w.top_movers.length)
      h+=`<div class="movers">movers: ${w.top_movers.map(m=>'<code>'+m+'</code>').join(' ')}</div>`;
  }else if(w.churn){const c=w.churn;
    h+=`<div class="churn">inserted <code>${c.inserted}</code> · evicted <code>${c.evicted}</code>`
      +` · re-admitted <code>${c.readmitted}</code> · slot-moved <code>${c.slot_moved}</code>`
      +` · survivors <code>${c.survivors}</code></div>`;}
  h+=`<div class="dim">value ${e.value} vs baseline ${e.baseline} (${e.direction})</div>`;
  b.innerHTML=h;
}
function renderEvents(list,incs){
  const el=document.getElementById('evlist'); el.innerHTML='';
  if(!list.length){el.innerHTML='<div class="ev">no anomalies</div>';document.getElementById('detailbody').innerHTML='';return;}
  let sel=null,first=null;
  const addRow=e=>{
    const r=document.createElement('div'); r.className='ev '+sc(e.severity);
    r.innerHTML=`<span class="step">${e.step}</span>`
      +`<span>${tn(e.table)}<span class="m">.${e.metric}</span></span>`
      +`<span class="sev">${e.severity.toFixed(1)}×</span>`;
    r.onclick=()=>{document.querySelectorAll('.ev').forEach(x=>x.classList.remove('sel'));r.classList.add('sel');detail(e);};
    if(evkey(e)===SEL)sel=[r,e];
    if(!first)first=[r,e];
    el.appendChild(r);
  };
  if(incs&&incs.length){
    incs.forEach(inc=>{
      const h=document.createElement('div'); h.className='evgrp';
      const st=inc.steps.length>1?`steps ${inc.steps[0]}–${inc.steps[inc.steps.length-1]}`:`step ${inc.steps[0]}`;
      h.textContent=`${st} · ${inc.n_events} signal${inc.n_events>1?'s':''} · worst ${inc.severity.toFixed(1)}×`;
      el.appendChild(h);
      inc.events.forEach(i=>addRow(list[i]));
    });
  }else list.forEach(addRow);
  if(!sel)sel=first;
  sel[0].classList.add('sel'); detail(sel[1]);
}
function renderGrid(){
  const grid=document.getElementById('grid'); grid.innerHTML='';
  const evLook={};
  DATA.events.forEach(e=>{const k=e.table+'|'+e.metric+'|'+e.step;evLook[k]=Math.max(evLook[k]||0,e.severity);});
  DATA.series.forEach(s=>{
    const fin=s.values.filter(v=>v!=null); if(!fin.length)return;
    const W=200,H=46,pad=5,n=s.values.length,mn=Math.min(...fin),mx=Math.max(...fin),rng=(mx-mn)||1;
    const X=i=>pad+(n<=1?0:i/(n-1))*(W-2*pad), Y=v=>pad+(1-(v-mn)/rng)*(H-2*pad);
    let d='',st=false,mk='',ms=0;
    s.values.forEach((v,i)=>{ if(v==null){st=false;return;}
      d+=(st?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1)+' ';st=true;
      const sev=evLook[s.table+'|'+s.metric+'|'+s.steps[i]];
      if(sev){ms=Math.max(ms,sev);mk+=`<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="3" class="mk ${sc(sev)}"/>`;}
    });
    const c=document.createElement('div'); c.className='cell '+cc(ms);
    c.innerHTML=`<div class="lbl"><span class="metric">${s.metric}</span><span class="tbl">${tn(s.table)}</span></div>`
      +`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><path class="spark" d="${d}"/>${mk}`
      +`<line class="hl" y1="${pad}" y2="${H-pad}" style="display:none"/><circle class="hd" r="3.2" style="display:none"/></svg>`;
    const svg=c.querySelector('svg'),hl=svg.querySelector('.hl'),hd=svg.querySelector('.hd');
    svg.addEventListener('mousemove',ev=>{
      const r=svg.getBoundingClientRect();
      const t=Math.max(0,Math.min(1,((ev.clientX-r.left)/r.width*W-pad)/(W-2*pad)));
      let i=Math.round(t*(n-1)); if(i<0||i>=n)return;
      if(s.values[i]==null){let j=i;while(j<n&&s.values[j]==null)j++;if(j>=n){j=i;while(j>=0&&s.values[j]==null)j--;}i=j;}
      if(i<0||i>=n||s.values[i]==null)return;
      hl.setAttribute('x1',X(i));hl.setAttribute('x2',X(i));hl.style.display='';
      hd.setAttribute('cx',X(i));hd.setAttribute('cy',Y(s.values[i]));hd.style.display='';
      tip.innerHTML=`step <b>${s.steps[i]}</b> · ${(+s.values[i]).toPrecision(4)}`;
      tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+14)+'px';
    });
    svg.addEventListener('mouseleave',()=>{hl.style.display='none';hd.style.display='none';tip.style.display='none';});
    c.onclick=()=>renderEvents(DATA.events.filter(e=>e.table===s.table&&e.metric===s.metric));
    grid.appendChild(c);
  });
}
function apply(D){
  DATA=D;
  document.getElementById('run').textContent=D.run;
  document.getElementById('meta').textContent=`${D.n_checkpoints} checkpoints · ${D.tables.length} tables · cal: ${(D.calibration||'').split(':')[0]}`;
  document.getElementById('worst').textContent=`worst ${D.worst_severity}× · ${D.n_incidents!=null?D.n_incidents+' incidents ('+D.n_events+' signals)':D.n_events+' anomalies'}`;
  renderGrid(); renderEvents(D.events,D.incidents);
}
const POLL=__POLL_MS__;
apply(JSON.parse(document.getElementById('flamediff-data').textContent));
if(POLL>0)setInterval(()=>fetch('data.json',{cache:'no-store'}).then(r=>r.json()).then(apply).catch(()=>{}),POLL);
</script></body></html>
"""


def render_html(report: dict, live_poll_ms: int = 0) -> str:
    """Render a Report dict into a self-contained HTML document. If ``live_poll_ms`` > 0 the page
    re-fetches ``data.json`` on that interval (used by ``flamediff serve``); 0 is a static file."""
    data = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    title = _html.escape(str(report.get("run", "run")))
    return (_TEMPLATE.replace("__TITLE__", title)
            .replace("__POLL_MS__", str(int(live_poll_ms)))
            .replace("__DATA__", data))
