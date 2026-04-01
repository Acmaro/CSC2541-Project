"""Generate method-comparison visualization HTML page."""
import csv, json

BASE = 'D:/AI4DD Project/CSC2541-Project/data/generation_stratified'
METHODS = ['baseline', 'crem', 'jtvae', 'libinvent', 'mmpdb']
FILES = {
    'baseline': 'baseline_scores.csv',
    'crem': 'crem_scores.csv',
    'jtvae': 'jtvae_scores.csv',
    'libinvent': 'libinvent_scores.csv',
    'mmpdb': 'mmpdb_scores.csv',
}
METHOD_COLORS = {
    'baseline': '#94a3b8',
    'crem':     '#60a5fa',
    'jtvae':    '#f59e0b',
    'libinvent':'#a78bfa',
    'mmpdb':    '#34d399',
}

def parse_bool(v):
    if v in ('True', 'true', '1'): return True
    if v in ('False', 'false', '0'): return False
    return None

def load(method):
    path = f'{BASE}/{method}/{FILES[method]}'
    rows = []
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                'smiles':    r['variant_smiles'],
                'tanimoto':  float(r['tanimoto']),
                'des':       float(r['desirability']),
                'qed':       float(r['qed']) if r.get('qed') else None,
                'mw_s':      float(r['mw_score']),
                'clogp_s':   float(r['clogp_score']),
                'tpsa_s':    float(r['tpsa_score']),
                'hbd_s':     float(r['hbd_score']),
                'hba_s':     float(r['hba_score']),
                'rot_s':     float(r['rot_bonds_score']),
                'bq':        r['quality_bin'],
                'synth':     parse_bool(r.get('is_synth')),
                'synth1':    parse_bool(r.get('is_synth_depth1')),
                'synth3':    parse_bool(r.get('is_synth_depth3')),
                'synth6':    parse_bool(r.get('is_synth_depth6')),
                'method':    method,
            })
    return rows

all_rows = []
for m in METHODS:
    rows = load(m)
    print(f'{m}: {len(rows)} rows')
    all_rows.extend(rows)

print(f'Total: {len(all_rows)} rows')

# Build columnar arrays
arrays = {k: [] for k in ['smiles','tanimoto','des','qed','mw_s','clogp_s','tpsa_s','hbd_s','hba_s','rot_s','bq','synth','synth1','synth3','synth6','method']}
for r in all_rows:
    for k in arrays:
        arrays[k].append(r[k])

DATA_JSON = json.dumps(arrays, separators=(',', ':'))
COLORS_JSON = json.dumps(METHOD_COLORS)

print(f'Data JSON: {len(DATA_JSON)/1024:.1f} KB')

# ---- Build HTML ----
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Segoe UI", system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }
h1 { font-size: 1.15rem; font-weight: 600; color: #f1f5f9; }
h2 { font-size: 0.72rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
#app { display: grid; grid-template-columns: 230px 1fr; gap: 12px; padding: 14px; min-height: 100vh; }
#sidebar { display: flex; flex-direction: column; gap: 14px; }
#content { display: flex; flex-direction: column; gap: 12px; }
.card { background: #1e2130; border-radius: 8px; border: 1px solid #2d3348; padding: 14px; }
.card-plot { background: #1e2130; border-radius: 8px; border: 1px solid #2d3348; overflow: hidden; }
.row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.ctrl-group { margin-bottom: 12px; }
.ctrl-group:last-child { margin-bottom: 0; }
.ctrl-group label { display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px; }
.ctrl-group label span { color: #e2e8f0; font-weight: 600; }
input[type=range] { width: 100%; accent-color: #6366f1; cursor: pointer; }
.sep { border: none; border-top: 1px solid #2d3348; margin: 12px 0; }
.method-btn { display: flex; align-items: center; gap: 8px; padding: 5px 8px; border-radius: 5px; cursor: pointer; border: 1px solid transparent; font-size: 0.75rem; transition: all 0.15s; margin-bottom: 4px; }
.method-btn.active { border-color: currentColor; background: #1e293b; }
.method-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.toggle-row { display: flex; gap: 4px; flex-wrap: wrap; }
.toggle-btn { flex: 1; padding: 5px 4px; font-size: 0.68rem; border-radius: 4px; border: 1px solid #2d3348; background: #161b2e; color: #94a3b8; cursor: pointer; text-align: center; white-space: nowrap; }
.toggle-btn.active { background: #312e81; border-color: #6366f1; color: #c7d2fe; }
.badge-row { display: flex; gap: 5px; flex-wrap: wrap; }
.badge { padding: 3px 7px; border-radius: 4px; font-size: 0.68rem; font-weight: 600; cursor: pointer; border: 1px solid transparent; }
.badge.active { border-color: currentColor; }
.stat-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
.stat-box { background: #161b2e; border-radius: 6px; padding: 8px 10px; border: 1px solid #2d3348; text-align: center; }
.stat-box .val { font-size: 1.15rem; font-weight: 700; line-height: 1; }
.stat-box .lbl { font-size: 0.62rem; color: #64748b; margin-top: 3px; }
.note { font-size: 0.65rem; color: #475569; margin-top: 6px; }
"""

METHODS_JS = json.dumps(METHODS)
COLORS_JS = COLORS_JSON

# Build bin badges
BIN_COLORS = {'<0.5':'#f59e0b','0.5-0.7':'#34d399','0.7-0.85':'#a78bfa','0.85-1.0':'#fb7185'}
bin_badges = ''
for b, c in BIN_COLORS.items():
    label = b.replace('<', '&lt;')
    bin_badges += f'<div class="badge active" data-qb="{b}" onclick="toggleBin(this)" style="color:{c};">{label}</div>\n'

method_btns = ''
for m, c in METHOD_COLORS.items():
    method_btns += f'<div class="method-btn active" data-method="{m}" onclick="toggleMethod(this)" style="color:{c};"><span class="method-dot" style="background:{c}"></span>{m}</div>\n'

stat_boxes = ''
for m, c in METHOD_COLORS.items():
    stat_boxes += f'<div class="stat-box" id="stat-{m}"><div class="val" id="stat-{m}-hits" style="color:{c}">-</div><div class="lbl">{m} hits</div></div>\n'

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Method Comparison Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{CSS}</style>
</head>
<body>
<div id="app">
  <!-- SIDEBAR -->
  <div id="sidebar">
    <div class="card">
      <h1>Method Comparison</h1>
      <div style="font-size:0.68rem;color:#64748b;margin-top:3px;">CSC2541 &middot; Desirability &times; Synthesizability</div>
    </div>

    <div class="card">
      <h2>Methods</h2>
      {method_btns}
    </div>

    <div class="card">
      <h2>Thresholds</h2>
      <div class="ctrl-group">
        <label>Tanimoto &tau;_t <span id="tau-t-val">0.30</span></label>
        <input type="range" id="tau-t" min="0" max="1" step="0.01" value="0.30">
      </div>
      <div class="ctrl-group">
        <label>Desirability &tau;_d <span id="tau-d-val">0.20</span></label>
        <input type="range" id="tau-d" min="0" max="1" step="0.01" value="0.20">
      </div>
      <p class="note">Affects hit counts &amp; scatter quadrant.</p>
    </div>

    <div class="card">
      <h2>Synth Depth</h2>
      <div class="toggle-row">
        <div class="toggle-btn active" data-depth="synth" onclick="toggleDepth(this)">Default</div>
        <div class="toggle-btn" data-depth="synth1" onclick="toggleDepth(this)">Depth 1</div>
        <div class="toggle-btn" data-depth="synth3" onclick="toggleDepth(this)">Depth 3</div>
        <div class="toggle-btn" data-depth="synth6" onclick="toggleDepth(this)">Depth 6</div>
      </div>
    </div>

    <div class="card">
      <h2>Quality Bin</h2>
      <div class="badge-row">
        {bin_badges}
      </div>
    </div>

    <div class="card">
      <h2>Scatter Color</h2>
      <div class="toggle-row">
        <div class="toggle-btn active" data-sc="method" onclick="toggleScatterColor(this)">Method</div>
        <div class="toggle-btn" data-sc="synth" onclick="toggleScatterColor(this)">Synth.</div>
        <div class="toggle-btn" data-sc="des" onclick="toggleScatterColor(this)">Des.</div>
      </div>
      <div style="margin-top:10px">
        <h2 style="margin-bottom:6px">Scatter Synth Filter</h2>
        <div class="toggle-row">
          <div class="toggle-btn active" data-ss="all" onclick="toggleScatterSynth(this)">All</div>
          <div class="toggle-btn" data-ss="true" onclick="toggleScatterSynth(this)" style="color:#4ade80">&#10003; Synth</div>
          <div class="toggle-btn" data-ss="false" onclick="toggleScatterSynth(this)" style="color:#f87171">&#10007; Not</div>
        </div>
      </div>
    </div>
  </div>

  <!-- MAIN CONTENT -->
  <div id="content">
    <!-- Stats row -->
    <div class="card" style="padding:10px 14px">
      <div class="stat-grid">
        {stat_boxes}
      </div>
    </div>

    <!-- Row 1: Violin distributions -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
      <div class="card-plot"><div id="violin-tan" style="height:360px;"></div></div>
      <div class="card-plot"><div id="violin-des" style="height:360px;"></div></div>
    </div>

    <!-- Row 2: Scatter + Synth bar -->
    <div class="row2">
      <div class="card-plot"><div id="scatter-main" style="height:320px;"></div></div>
      <div class="card-plot"><div id="bar-synth" style="height:320px;"></div></div>
    </div>

    <!-- Row 3: Hit rate + Component radar + QED -->
    <div class="row3">
      <div class="card-plot"><div id="bar-hits" style="height:260px;"></div></div>
      <div class="card-plot"><div id="radar-comp" style="height:260px;"></div></div>
      <div class="card-plot"><div id="violin-qed" style="height:260px;"></div></div>
    </div>
  </div>
</div>

<script>
const RAW = {DATA_JSON};
const N = RAW.tanimoto.length;
const METHODS = {METHODS_JS};
const COLORS = {COLORS_JS};
const BIN_COLORS = {{"<0.5":"#f59e0b","0.5-0.7":"#34d399","0.7-0.85":"#a78bfa","0.85-1.0":"#fb7185"}};

let state = {{
  activeMethods: new Set(METHODS),
  activeBins: new Set(["<0.5","0.5-0.7","0.7-0.85","0.85-1.0"]),
  tauT: 0.30, tauD: 0.20,
  depth: "synth",
  scatterColor: "method",
  scatterSynth: "all",
}};

function getIdx(method) {{
  const idx = [];
  for (let i = 0; i < N; i++) {{
    if (RAW.method[i] !== method) continue;
    if (!state.activeBins.has(RAW.bq[i])) continue;
    idx.push(i);
  }}
  return idx;
}}

function cfg() {{ return {{responsive:true,displaylogo:false,modeBarButtonsToRemove:["lasso2d","select2d"]}}; }}

function darkLayout(extra) {{
  return Object.assign({{
    paper_bgcolor:"transparent", plot_bgcolor:"#161b2e",
    font:{{color:"#94a3b8",size:10}},
    margin:{{t:38,b:44,l:50,r:14}},
    legend:{{bgcolor:"transparent",bordercolor:"#2d3348",borderwidth:1,font:{{size:9}}}},
    xaxis:{{gridcolor:"#1e293b",zerolinecolor:"#2d3348"}},
    yaxis:{{gridcolor:"#1e293b",zerolinecolor:"#2d3348"}},
  }}, extra||{{}});
}}

// ---- Violin: tanimoto & desirability ----
function plotViolin(divId, field, title, xtitle) {{
  const traces = [];
  const annotations = [];
  let mIdx = 0;
  for (const m of METHODS) {{
    if (!state.activeMethods.has(m)) continue;
    let idx = getIdx(m);
    // For desirability: filter to non-zero values only (geometric mean collapses to 0 when any component fails)
    if (field === "des") {{
      const all = idx.length;
      idx = idx.filter(i => RAW[field][i] > 0);
      const pct = all > 0 ? (idx.length / all * 100).toFixed(0) : 0;
      annotations.push({{
        x: mIdx, y: 1.06, xref:"x", yref:"y",
        text: pct + "%", showarrow: false,
        font: {{size:9, color: COLORS[m]}},
      }});
    }}
    if (idx.length < 2) {{ mIdx++; continue; }}
    traces.push({{
      type:"violin", name:m,
      y: idx.map(i => RAW[field][i]),
      box:{{visible:true}}, meanline:{{visible:true}},
      line:{{color:COLORS[m]}}, fillcolor:COLORS[m]+"33",
      points:false, opacity:0.85,
    }});
    mIdx++;
  }}
  const desNote = field === "des" ? " (non-zero only, % shown)" : "";
  const layout = darkLayout({{
    title:{{text: title + desNote, font:{{size:11,color:"#94a3b8"}},x:0.02}},
    yaxis:{{title:{{text:xtitle,font:{{size:10}}}},gridcolor:"#1e293b",range:[-0.05,1.08]}},
    violingap:0, violingroupgap:0.1, violinmode:"group",
    margin:{{t:38,b:30,l:50,r:10}},
    showlegend:false,
    annotations: field === "des" ? annotations : [],
  }});
  if (field === "tanimoto") {{
    layout.shapes = [{{type:"line",x0:-0.5,x1:METHODS.length-0.5,y0:state.tauT,y1:state.tauT,line:{{color:"#818cf8",width:1.5,dash:"dot"}}}}];
  }} else if (field === "des") {{
    layout.shapes = [{{type:"line",x0:-0.5,x1:METHODS.length-0.5,y0:state.tauD,y1:state.tauD,line:{{color:"#fbbf24",width:1.5,dash:"dot"}}}}];
  }}
  Plotly.newPlot(divId, traces, layout, cfg());
}}

// ---- Scatter: tanimoto vs desirability ----
function plotScatter() {{
  const traces = [];
  const addedMethods = [];
  for (const m of METHODS) {{
    if (!state.activeMethods.has(m)) continue;
    let idx = getIdx(m);
    // Apply synthesizability filter
    if (state.scatterSynth === "true")  idx = idx.filter(i => RAW[state.depth][i] === true);
    if (state.scatterSynth === "false") idx = idx.filter(i => RAW[state.depth][i] === false);
    let colors;
    if (state.scatterColor === "method") {{
      colors = COLORS[m];
    }} else if (state.scatterColor === "synth") {{
      colors = idx.map(i => RAW[state.depth][i] === true ? "#4ade80" : RAW[state.depth][i] === false ? "#f87171" : "#64748b");
    }} else {{
      colors = idx.map(i => RAW.des[i]);
    }}
    traces.push({{
      x: idx.map(i => RAW.tanimoto[i]),
      y: idx.map(i => RAW.des[i]),
      mode:"markers", type:"scattergl", name:m,
      marker:{{
        color:colors, size:3, opacity:0.65,
        colorscale: state.scatterColor === "des" ? "Viridis" : undefined,
        showscale: state.scatterColor === "des" && addedMethods.length === 0,
        colorbar: state.scatterColor === "des" && addedMethods.length === 0 ? {{thickness:10,tickfont:{{color:"#94a3b8",size:9}},bgcolor:"transparent",bordercolor:"#2d3348"}} : undefined,
      }},
      hovertemplate: "<b>"+m+"</b><br>Tan: %{{x:.3f}}<br>Des: %{{y:.3f}}<extra></extra>",
    }});
    addedMethods.push(m);
  }}
  const shapes = [
    {{type:"line",x0:state.tauT,x1:state.tauT,y0:0,y1:1,line:{{color:"#818cf8",width:1.5,dash:"dot"}}}},
    {{type:"line",x0:0,x1:1,y0:state.tauD,y1:state.tauD,line:{{color:"#fbbf24",width:1.5,dash:"dot"}}}},
    {{type:"rect",x0:state.tauT,x1:1.01,y0:state.tauD,y1:1.01,fillcolor:"#22c55e08",line:{{width:0}}}},
  ];
  const layout = darkLayout({{
    title:{{text:"Tanimoto vs Desirability",font:{{size:11,color:"#94a3b8"}},x:0.02}},
    xaxis:{{title:{{text:"Tanimoto (ECFP4)"}},range:[-0.02,1.05],gridcolor:"#1e293b",zerolinecolor:"#2d3348"}},
    yaxis:{{title:{{text:"Desirability"}},range:[-0.02,1.08],gridcolor:"#1e293b",zerolinecolor:"#2d3348"}},
    shapes, hovermode:"closest", showlegend: state.scatterColor === "method",
    legend:{{bgcolor:"transparent",bordercolor:"#2d3348",borderwidth:1,font:{{size:9}}}},
    margin:{{t:38,b:44,l:50,r:10}},
  }});
  Plotly.newPlot("scatter-main", traces, layout, cfg());
}}

// ---- Bar: synthesizability rate per method ----
function plotSynthBar() {{
  const methodNames = [], rates = [], counts = [], colors = [];
  for (const m of METHODS) {{
    if (!state.activeMethods.has(m)) continue;
    const idx = getIdx(m);
    const known = idx.filter(i => RAW[state.depth][i] !== null);
    const synth = idx.filter(i => RAW[state.depth][i] === true).length;
    const rate = known.length > 0 ? synth / known.length * 100 : 0;
    methodNames.push(m); rates.push(+rate.toFixed(2)); counts.push(synth); colors.push(COLORS[m]);
  }}
  const layout = darkLayout({{
    title:{{text:"Synthesizability Rate (" + state.depth + ")",font:{{size:11,color:"#94a3b8"}},x:0.02}},
    xaxis:{{gridcolor:"#1e293b"}},
    yaxis:{{title:{{text:"% Synthesizable"}},range:[0,100],gridcolor:"#1e293b"}},
    margin:{{t:38,b:44,l:50,r:10}},
    showlegend:false,
  }});
  Plotly.newPlot("bar-synth", [{{
    x: methodNames, y: rates, type:"bar",
    marker:{{color:colors, opacity:0.85}},
    text: rates.map(r => r.toFixed(1)+"%"),
    textposition:"outside", textfont:{{size:10,color:"#94a3b8"}},
    hovertemplate: "<b>%{{x}}</b><br>Rate: %{{y:.1f}}%<extra></extra>",
  }}], layout, cfg());
}}

// ---- Bar: hit rate per method ----
function plotHitBar() {{
  const methodNames = [], rates = [], colors = [];
  for (const m of METHODS) {{
    if (!state.activeMethods.has(m)) continue;
    const idx = getIdx(m);
    const hits = idx.filter(i => RAW.tanimoto[i] >= state.tauT && RAW.des[i] >= state.tauD).length;
    const rate = idx.length > 0 ? hits / idx.length * 100 : 0;
    methodNames.push(m); rates.push(+rate.toFixed(2)); colors.push(COLORS[m]);
    document.getElementById("stat-"+m+"-hits").textContent = hits.toLocaleString();
  }}
  const layout = darkLayout({{
    title:{{text:`Hit Rate (\u03c4_t=${{state.tauT.toFixed(2)}}, \u03c4_d=${{state.tauD.toFixed(2)}})`,font:{{size:11,color:"#94a3b8"}},x:0.02}},
    xaxis:{{gridcolor:"#1e293b"}},
    yaxis:{{title:{{text:"Hit Rate (%)"}},gridcolor:"#1e293b"}},
    margin:{{t:38,b:40,l:50,r:10}},
    showlegend:false,
  }});
  Plotly.newPlot("bar-hits", [{{
    x: methodNames, y: rates, type:"bar",
    marker:{{color:colors, opacity:0.85}},
    text: rates.map(r => r.toFixed(1)+"%"),
    textposition:"outside", textfont:{{size:10,color:"#94a3b8"}},
    hovertemplate: "<b>%{{x}}</b><br>Hit Rate: %{{y:.1f}}%<extra></extra>",
  }}], layout, cfg());
}}

// ---- Radar: mean component scores ----
function plotRadar() {{
  const theta = ["MW","cLogP","TPSA","HBD","HBA","RotB","MW"];
  const fields = ["mw_s","clogp_s","tpsa_s","hbd_s","hba_s","rot_s"];
  const traces = [];
  for (const m of METHODS) {{
    if (!state.activeMethods.has(m)) continue;
    const idx = getIdx(m);
    if (idx.length === 0) continue;
    const means = fields.map(f => {{
      const vals = idx.map(i => RAW[f][i]).filter(v => v != null);
      return vals.reduce((a,b) => a+b, 0) / vals.length;
    }});
    traces.push({{
      type:"scatterpolar", name:m,
      r:[...means, means[0]], theta,
      fill:"toself", fillcolor:COLORS[m]+"33",
      line:{{color:COLORS[m],width:1.5}},
    }});
  }}
  Plotly.newPlot("radar-comp", traces, {{
    paper_bgcolor:"transparent", plot_bgcolor:"transparent",
    font:{{color:"#94a3b8",size:9}},
    margin:{{t:38,b:10,l:10,r:10}},
    title:{{text:"Mean Component Scores",font:{{size:11,color:"#94a3b8"}},x:0.02}},
    polar:{{
      bgcolor:"#161b2e",
      radialaxis:{{range:[0,1],gridcolor:"#2d3348",tickfont:{{size:8}}}},
      angularaxis:{{gridcolor:"#2d3348"}},
    }},
    legend:{{bgcolor:"transparent",bordercolor:"#2d3348",borderwidth:1,font:{{size:9}}}},
  }}, cfg());
}}

// ---- Violin: QED ----
function plotQEDViolin() {{
  const traces = [];
  for (const m of METHODS) {{
    if (!state.activeMethods.has(m)) continue;
    const idx = getIdx(m).filter(i => RAW.qed[i] != null);
    if (idx.length === 0) continue;
    traces.push({{
      type:"violin", name:m,
      y: idx.map(i => RAW.qed[i]),
      box:{{visible:true}}, meanline:{{visible:true}},
      line:{{color:COLORS[m]}}, fillcolor:COLORS[m]+"33",
      points:false, opacity:0.85,
    }});
  }}
  Plotly.newPlot("violin-qed", traces, darkLayout({{
    title:{{text:"QED Distribution",font:{{size:11,color:"#94a3b8"}},x:0.02}},
    yaxis:{{title:{{text:"QED"}},range:[-0.05,1.05],gridcolor:"#1e293b"}},
    violingap:0, violingroupgap:0.1, violinmode:"group",
    margin:{{t:38,b:30,l:50,r:10}},
    showlegend:false,
  }}), cfg());
}}

function replotAll() {{
  plotViolin("violin-tan","tanimoto","Tanimoto Distribution","Tanimoto (ECFP4)");
  plotViolin("violin-des","des","Desirability Distribution","Desirability");
  plotScatter();
  plotSynthBar();
  plotHitBar();
  plotRadar();
  plotQEDViolin();
}}

// Event handlers
document.getElementById("tau-t").addEventListener("input", function() {{
  state.tauT = +this.value;
  document.getElementById("tau-t-val").textContent = state.tauT.toFixed(2);
  replotAll();
}});
document.getElementById("tau-d").addEventListener("input", function() {{
  state.tauD = +this.value;
  document.getElementById("tau-d-val").textContent = state.tauD.toFixed(2);
  replotAll();
}});

function toggleMethod(el) {{
  const m = el.dataset.method;
  if (state.activeMethods.has(m)) {{
    if (state.activeMethods.size > 1) {{ state.activeMethods.delete(m); el.classList.remove("active"); }}
  }} else {{ state.activeMethods.add(m); el.classList.add("active"); }}
  replotAll();
}}
function toggleBin(el) {{
  const qb = el.dataset.qb;
  if (state.activeBins.has(qb)) {{
    if (state.activeBins.size > 1) {{ state.activeBins.delete(qb); el.classList.remove("active"); }}
  }} else {{ state.activeBins.add(qb); el.classList.add("active"); }}
  replotAll();
}}
function toggleDepth(el) {{
  document.querySelectorAll("[data-depth]").forEach(e => e.classList.remove("active"));
  el.classList.add("active"); state.depth = el.dataset.depth; replotAll();
}}
function toggleScatterColor(el) {{
  document.querySelectorAll("[data-sc]").forEach(e => e.classList.remove("active"));
  el.classList.add("active"); state.scatterColor = el.dataset.sc; replotAll();
}}
function toggleScatterSynth(el) {{
  document.querySelectorAll("[data-ss]").forEach(e => e.classList.remove("active"));
  el.classList.add("active"); state.scatterSynth = el.dataset.ss; replotAll();
}}

replotAll();
</script>
</body>
</html>
"""

out = 'D:/AI4DD Project/CSC2541-Project/docs/method_comparison.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f'Written {len(HTML)/1024:.1f} KB to {out}')
