"""Self-contained HTML view for `flamediff rank` -- the factorization-advisory page.

Same contract as _html.py's report viewer: embed the rank payload as JSON, render with vanilla JS
+ inline SVG -- no server, no build step, no dependencies, works offline. Per table: the
energy-at-rank curve (how much variance the top-r factors keep) and the rank-at-energy trajectory
over the run (has the rank stabilized enough to size a factorization?).
"""
from __future__ import annotations

import html as _html
import json

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>flamediff rank — __TITLE__</title>
<style>
:root{--base:#1e1e2e;--mantle:#181825;--surface:#313244;--surface2:#45475a;--text:#cdd6f4;
--sub:#a6adc8;--overlay:#6c7086;--mauve:#cba6f7;--red:#f38ba8;--peach:#fab387;--yellow:#f9e2af;
--blue:#89b4fa;--teal:#94e2d5;--green:#a6e3a1;}
*{box-sizing:border-box}
body{margin:0;background:var(--base);color:var(--text);
font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:14px 20px;background:var(--mantle);border-bottom:1px solid var(--surface);
display:flex;gap:22px;align-items:baseline;flex-wrap:wrap}
header h1{margin:0;font-size:16px;color:var(--mauve)}
.meta{color:var(--sub);font-size:12px}
main{padding:16px 20px;display:flex;flex-direction:column;gap:18px}
.card{background:var(--mantle);border:1px solid var(--surface);border-radius:6px;padding:14px}
.card h2{margin:0 0 4px;font-size:14px;color:var(--text)}
.card .sub{color:var(--overlay);font-size:12px;margin-bottom:10px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.chart .ttl{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--overlay);margin-bottom:4px}
svg{width:100%;height:180px;display:block}
.axis{stroke:var(--surface2);stroke-width:1}
.lbl{fill:var(--overlay);font-size:10px}
.curve{fill:none;stroke:var(--blue);stroke-width:1.8}
.gline{stroke:var(--surface);stroke-width:1;stroke-dasharray:2 3}
.r90{stroke:var(--teal)} .r95{stroke:var(--yellow)} .r99{stroke:var(--peach)}
.rl{fill:none;stroke-width:1.6}
.mk{stroke:var(--base);stroke-width:1}
.stable{stroke:var(--green);stroke-width:1;stroke-dasharray:3 3}
.advice{margin-top:10px;font-size:13px;color:var(--sub)}
.advice b{color:var(--green)} .advice .warn{color:var(--peach)}
.legend{font-size:11px;color:var(--sub);display:flex;gap:14px;margin-top:6px}
.sw{display:inline-block;width:10px;height:3px;vertical-align:middle;margin-right:5px}
#tip{position:fixed;pointer-events:none;display:none;z-index:10;white-space:nowrap;font-size:11px;
padding:3px 7px;border-radius:4px;background:var(--surface2);color:var(--text);border:1px solid var(--overlay)}
</style></head><body>
<header>
  <h1>flamediff rank · <span id="run"></span></h1>
  <span class="meta" id="meta"></span>
</header>
<main id="main"></main>
<script id="flamediff-rank-data" type="application/json">__DATA__</script>
<script>
const tip=document.createElement('div'); tip.id='tip'; document.body.appendChild(tip);
const D=JSON.parse(document.getElementById('flamediff-rank-data').textContent);
document.getElementById('run').textContent=D.run;
document.getElementById('meta').textContent=`${D.tables.length} tables · rank needed at 90 / 95 / 99% variance`;
const W=360,H=180,P={l:34,r:8,t:8,b:20};
const EN=[['0.9','r90','90%'],['0.95','r95','95%'],['0.99','r99','99%']];
function svgEl(html){const div=document.createElement('div');div.innerHTML=
  `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${html}</svg>`;return div.firstChild;}
function hover(svg,fmt){
  svg.addEventListener('mousemove',ev=>{const v=fmt(ev);if(!v){tip.style.display='none';return;}
    tip.innerHTML=v;tip.style.display='block';
    tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+14)+'px';});
  svg.addEventListener('mouseleave',()=>{tip.style.display='none';});
}
function energyChart(t){
  const n=t.energy_curve.length;
  const X=r=>P.l+((r-1)/Math.max(1,n-1))*(W-P.l-P.r), Y=e=>P.t+(1-e)*(H-P.t-P.b);
  let h=`<line class="axis" x1="${P.l}" y1="${Y(0)}" x2="${W-P.r}" y2="${Y(0)}"/>`
    +`<line class="axis" x1="${P.l}" y1="${P.t}" x2="${P.l}" y2="${Y(0)}"/>`;
  for(const [e,cls,lab] of EN){
    const r=t.rank_at_energy[e][t.rank_at_energy[e].length-1];
    h+=`<line class="gline" x1="${P.l}" y1="${Y(+e)}" x2="${W-P.r}" y2="${Y(+e)}"/>`
      +`<circle class="mk ${cls}" fill="currentColor" style="color:var(--${cls==='r90'?'teal':cls==='r95'?'yellow':'peach'})" cx="${X(r)}" cy="${Y(t.energy_curve[r-1])}" r="3.5"/>`
      +`<text class="lbl" x="${X(r)+5}" y="${Y(t.energy_curve[r-1])-5}">r=${r}</text>`;
  }
  h+=`<path class="curve" d="${t.energy_curve.map((e,i)=>(i?'L':'M')+X(i+1).toFixed(1)+' '+Y(e).toFixed(1)).join(' ')}"/>`
    +`<text class="lbl" x="${P.l-4}" y="${Y(1)+4}" text-anchor="end">100%</text>`
    +`<text class="lbl" x="${P.l-4}" y="${Y(0)+4}" text-anchor="end">0%</text>`
    +`<text class="lbl" x="${P.l}" y="${H-6}">rank 1</text>`
    +`<text class="lbl" x="${W-P.r}" y="${H-6}" text-anchor="end">${n}</text>`;
  const svg=svgEl(h);
  hover(svg,ev=>{const b=svg.getBoundingClientRect();
    const r=Math.max(1,Math.min(n,Math.round(1+((ev.clientX-b.left)/b.width*W-P.l)/(W-P.l-P.r)*(n-1))));
    return `top-<b>${r}</b> factors keep <b>${(t.energy_curve[r-1]*100).toFixed(1)}%</b>`;});
  return svg;
}
function rankChart(t){
  const steps=t.steps,n=steps.length,dim=t.dim;
  const X=i=>P.l+(n<=1?0:i/(n-1))*(W-P.l-P.r), Y=r=>P.t+(1-r/dim)*(H-P.t-P.b);
  let h=`<line class="axis" x1="${P.l}" y1="${Y(0)}" x2="${W-P.r}" y2="${Y(0)}"/>`
    +`<line class="axis" x1="${P.l}" y1="${P.t}" x2="${P.l}" y2="${Y(0)}"/>`
    +`<line class="gline" x1="${P.l}" y1="${Y(dim)}" x2="${W-P.r}" y2="${Y(dim)}"/>`;
  if(t.stable_since!=null){const i=steps.indexOf(t.stable_since);
    if(i>=0)h+=`<line class="stable" x1="${X(i)}" y1="${P.t}" x2="${X(i)}" y2="${Y(0)}"/>`
      +`<text class="lbl" x="${X(i)+4}" y="${P.t+10}" style="fill:var(--green)">stable</text>`;}
  for(const [e,cls] of EN)
    h+=`<path class="rl ${cls}" d="${t.rank_at_energy[e].map((r,i)=>(i?'L':'M')+X(i).toFixed(1)+' '+Y(r).toFixed(1)).join(' ')}"/>`;
  h+=`<text class="lbl" x="${P.l-4}" y="${Y(dim)+4}" text-anchor="end">${dim}</text>`
    +`<text class="lbl" x="${P.l-4}" y="${Y(0)+4}" text-anchor="end">0</text>`
    +`<text class="lbl" x="${P.l}" y="${H-6}">step ${steps[0]}</text>`
    +`<text class="lbl" x="${W-P.r}" y="${H-6}" text-anchor="end">${steps[n-1]}</text>`;
  const svg=svgEl(h);
  hover(svg,ev=>{const b=svg.getBoundingClientRect();
    const i=Math.max(0,Math.min(n-1,Math.round(((ev.clientX-b.left)/b.width*W-P.l)/(W-P.l-P.r)*(n-1))));
    return `step <b>${steps[i]}</b> · r@90 ${t.rank_at_energy['0.9'][i]} · r@95 ${t.rank_at_energy['0.95'][i]} · r@99 ${t.rank_at_energy['0.99'][i]}`;});
  return svg;
}
const main=document.getElementById('main');
D.tables.forEach(t=>{
  const r95=t.rank_at_energy['0.95'][t.rank_at_energy['0.95'].length-1];
  const card=document.createElement('div'); card.className='card';
  card.innerHTML=`<h2>${t.table}</h2><div class="sub">dim=${t.dim} · ${t.n.toLocaleString()} resident ids</div>`;
  const charts=document.createElement('div'); charts.className='charts';
  for(const [ttl,el] of [['energy kept by top-r factors',energyChart(t)],
                         ['rank needed over the run',rankChart(t)]]){
    const c=document.createElement('div'); c.className='chart';
    c.innerHTML=`<div class="ttl">${ttl}</div>`; c.appendChild(el); charts.appendChild(c);
  }
  card.appendChild(charts);
  const legend=document.createElement('div'); legend.className='legend';
  legend.innerHTML=`<span><span class="sw" style="background:var(--teal)"></span>90%</span>
    <span><span class="sw" style="background:var(--yellow)"></span>95%</span>
    <span><span class="sw" style="background:var(--peach)"></span>99%</span>
    <span><span class="sw" style="background:var(--green)"></span>rank95 stable</span>`;
  card.appendChild(legend);
  const adv=document.createElement('div'); adv.className='advice';
  const frac=(r95/t.dim*100).toFixed(0);
  adv.innerHTML=(t.stable_since!=null
      ?`rank95 stable since step ${t.stable_since} — <b>safe to size a factorization</b>. `
      :`<span class="warn">rank95 still moving — sizing a factorization now would bake in a dimensionality the table hasn't settled into.</span> `)
    +(r95<=t.dim/2
      ?`Top-<b>${r95}</b> factors keep 95% of the variance (${frac}% of the parameters).`
      :`95% of the variance needs rank ${r95} of ${t.dim} — little to gain from factorizing.`);
  card.appendChild(adv);
  main.appendChild(card);
});
</script></body></html>
"""


def render_rank_html(payload: dict) -> str:
    """Render a `flamediff rank` payload (spectral.render_json's dict) into a static page."""
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = _html.escape(str(payload.get("run", "run")))
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)
