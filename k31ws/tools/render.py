#!/usr/bin/env python3
"""render.py: build k31ws/tracker.html from k31ws/status/*.json, k31ws/pool.json and k31ws/campaign.json.

The tracker is the single source of truth of the K3.1 VR200 kernel campaign (NVFP4 checkpoint, DEP16 prefill,
DCP4 x DP8 EP32 decode): one row per workstream with current kernel time, share of the step, model achievable,
target, gap closed, state, and a detail card with the gap analysis, inefficiencies, Rubin features, plan and work log.
"""
import html, json, glob, os, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = lambda s: html.escape(str(s if s is not None else ""))


def load(p, default):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def fnum(v, nd=2):
    if v is None or v == "":
        return "–"
    try:
        v = float(v)
    except Exception:
        return E(v)
    return f"{v:,.{nd}f}" if abs(v) < 100 else f"{v:,.0f}"


def closed(s):
    if s.get("ckpt") != "nvfp4":  # user rule: gap closed only when baseline, current, achievable are all NVFP4-basis
        return None
    b, c, a = s.get("baseline_ms"), s.get("current_ms"), s.get("achievable_ms")
    try:
        b, c, a = float(b), float(c), float(a)
        return max(0.0, min(1.0, (b - c) / (b - a))) if b > a else None
    except Exception:
        return None


ORDER = {"infra": 0, "both": 1, "decode": 2, "prefill": 3}
STATE_CLS = {"analysis": "st-an", "implementing": "st-im", "measuring": "st-me", "blocked": "st-bl",
             "landed": "st-la", "done": "st-do", "queued": "st-q"}


def main():
    camp = load(f"{ROOT}/campaign.json", {})
    pool = load(f"{ROOT}/pool.json", {})
    ss = [load(p, None) for p in sorted(glob.glob(f"{ROOT}/status/*.json"))]
    ss = [s for s in ss if s]
    ss.sort(key=lambda s: (ORDER.get(s.get("side"), 9), -(float(s.get("baseline_ms") or 0) - float(s.get("achievable_ms") or 0)), s.get("id", "")))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def side_rows(side):
        out = []
        for s in ss:
            if s.get("side") != side:
                continue
            cl = closed(s)
            share = None
            try:
                share = 100 * float(s["current_ms"]) / float(s["step_ms"])
            except Exception:
                pass
            ratio = None
            try:
                ratio = float(s["current_ms"]) / float(s["achievable_ms"])
            except Exception:
                pass
            bar = ("NVFP4 baseline pending" if s.get("ckpt") != "nvfp4" else "–") if cl is None else f'<span class="bar"><i style="width:{cl*100:.0f}%"></i></span> {cl*100:.0f}%'
            out.append(f"""<tr>
<td><a href="#{E(s['id'])}" class="wid">{E(s['id'])}</a><div class="wt">{E(s.get('title'))}</div></td>
<td class="n">{fnum(s.get('current_ms'))}</td><td class="n">{'–' if share is None else f'{share:.0f}%'}</td>
<td class="n">{fnum(s.get('achievable_ms'))}</td><td class="n">{fnum(s.get('target_ms'))}</td>
<td class="n">{'–' if ratio is None else f'{ratio:.1f}×'}</td><td class="gc">{bar}</td>
<td><span class="st {STATE_CLS.get(s.get('state'), '')}">{E(s.get('state'))}</span></td>
<td class="nx">{E(s.get('next'))}</td></tr>""")
        return "\n".join(out)

    def table(side, label):
        rows = side_rows(side)
        if not rows:
            return ""
        return f"""<h3>{label}</h3><div class="tw"><table>
<thead><tr><th>Workstream</th><th>Now (ms)</th><th>Share of step</th><th>Model achievable</th><th>Target</th>
<th>Now / achievable</th><th>Gap closed</th><th>State</th><th>Next</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    def card(s):
        li = lambda xs: "".join(f"<li>{E(x)}</li>" for x in (xs or []))
        work = "".join(f"<tr><td class='t'>{E(w.get('t'))}</td><td>{E(w.get('what'))}</td><td>{E(w.get('result'))}</td></tr>"
                       for w in reversed(s.get("work") or []))
        ev = "".join(f"<li><code>{E(x)}</code></li>" for x in (s.get("evidence") or []))
        return f"""<section class="card" id="{E(s['id'])}">
<header><span class="wid">{E(s['id'])}</span><h4>{E(s.get('title'))}</h4>
<span class="st {STATE_CLS.get(s.get('state'), '')}">{E(s.get('state'))}</span></header>
<div class="meta"><span>Shape: {E(s.get('shape'))}</span><span>Basis: {E(s.get('basis'))}</span>
<span>Branch: <code>{E(s.get('branch'))}</code></span><span>Owner: {E(s.get('owner'))}</span><span>Updated: {E(s.get('updated'))}</span></div>
<div class="nums"><div><b>{fnum(s.get('baseline_ms'))}</b><small>baseline ms</small></div><div><b>{fnum(s.get('current_ms'))}</b><small>now ms</small></div>
<div><b>{fnum(s.get('achievable_ms'))}</b><small>model achievable ms</small></div><div><b>{fnum(s.get('target_ms'))}</b><small>target ms</small></div>
<div><b>{fnum(s.get('step_ms'))}</b><small>step ms</small></div></div>
<h5>Gap analysis</h5><p>{E(s.get('gap_analysis'))}</p>
<div class="two"><div><h5>Current inefficiencies</h5><ul>{li(s.get('inefficiencies'))}</ul></div>
<div><h5>Vera Rubin features used</h5><ul>{li(s.get('rubin_features'))}</ul></div></div>
<h5>Plan</h5><ol>{li(s.get('plan'))}</ol>
{('<p class="blk"><b>Blocked:</b> ' + E(s.get('blockers')) + '</p>') if s.get('blockers') else ''}
<h5>Work log</h5><div class="tw"><table class="log"><tbody>{work or '<tr><td>No entries yet.</td></tr>'}</tbody></table></div>
{('<h5>Evidence</h5><ul class="ev">' + ev + '</ul>') if ev else ''}
</section>"""

    steps = "".join(f"<div class='kpi'><small>{E(k)}</small><b>{E(v.get('now'))}</b><span>model achievable {E(v.get('ach'))} · {E(v.get('note',''))}</span></div>"
                    for k, v in (camp.get("steps") or {}).items())
    holds = "".join(f"<tr><td>{E(h.get('job'))}</td><td>{E(h.get('name'))}</td><td>{E(h.get('state'))}</td><td>{E(h.get('nodes'))}</td><td>{E(h.get('start'))}</td><td>{E(h.get('left'))}</td></tr>"
                    for h in pool.get("holds", []))
    findings = "".join(f"<li><b>{E(f.get('t'))}</b> {E(f.get('text'))}</li>" for f in reversed(camp.get("findings") or []))
    page = f"""<title>K3.1 VR200 Kernel Campaign</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap">
<style>
/* layout: status board. Summary first (steps, pool, findings), then one table per side, then one card per workstream */
:root {{ --bg:#f5f7f6; --panel:#ffffff; --ink:#15201c; --mute:#5b6a64; --line:#d9e1dd; --acc:#0f7b5f; --acc2:#e3f2ec;
  --warn:#a35b00; --bad:#b3261e; --good:#0f7b5f; --info:#2f5fa7;
  --f-d:'Space Grotesk', 'IBM Plex Sans', system-ui, sans-serif; --f-b:'IBM Plex Sans', system-ui, sans-serif; --f-m:'IBM Plex Mono', ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f1513; --panel:#161e1b; --ink:#e3ece8; --mute:#93a39c; --line:#2a3632; --acc:#4cc59d; --acc2:#17322a;
  --warn:#e3a14a; --bad:#ef7a72; --good:#4cc59d; --info:#86aef0; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#0f1513; --panel:#161e1b; --ink:#e3ece8; --mute:#93a39c; --line:#2a3632; --acc:#4cc59d; --acc2:#17322a;
  --warn:#e3a14a; --bad:#ef7a72; --good:#4cc59d; --info:#86aef0; color-scheme:dark }}
body {{ background:var(--bg); color:var(--ink); font:14px/1.5 var(--f-b); }}
.wrap {{ max-width:1280px; margin:0 auto; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); padding-block:24px 64px; }}
h1,h2,h3,h4 {{ font-family:var(--f-d); text-wrap:balance; margin:0; }}
h1 {{ font-size:28px; }} h2 {{ font-size:20px; margin-top:32px; }} h3 {{ font-size:16px; margin:22px 0 8px; color:var(--mute); text-transform:uppercase; letter-spacing:.06em; }}
.sub {{ color:var(--mute); max-width:90ch; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin-top:16px; }}
.kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; display:flex; flex-direction:column; gap:2px; min-width:0; }}
.kpi small {{ color:var(--mute); text-transform:uppercase; letter-spacing:.05em; font-size:11px; }} .kpi b {{ font:600 22px var(--f-d); font-variant-numeric:tabular-nums; }} .kpi span {{ color:var(--mute); font-size:12px; }}
.tw {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }} th {{ font-size:12px; color:var(--mute); font-weight:500; white-space:nowrap; }}
td.n {{ text-align:right; font-family:var(--f-m); font-variant-numeric:tabular-nums; white-space:nowrap; }}
.wid {{ font-family:var(--f-m); font-weight:500; color:var(--acc); text-decoration:none; }} .wt {{ color:var(--mute); font-size:12px; }}
td.nx {{ color:var(--mute); font-size:12px; min-width:220px; }} td.gc {{ white-space:nowrap; font-family:var(--f-m); font-size:12px; }}
.bar {{ display:inline-block; width:70px; height:6px; background:var(--line); border-radius:3px; vertical-align:middle; }} .bar i {{ display:block; height:100%; background:var(--acc); border-radius:3px; }}
.st {{ font-size:11px; padding:2px 8px; border-radius:10px; border:1px solid var(--line); white-space:nowrap; }}
.st-an {{ color:var(--info); }} .st-im {{ color:var(--acc); }} .st-me {{ color:var(--acc); background:var(--acc2); }} .st-bl {{ color:var(--bad); border-color:var(--bad); }}
.st-la, .st-do {{ color:var(--good); background:var(--acc2); border-color:var(--good); }} .st-q {{ color:var(--mute); }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin-top:16px; }}
.card header {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }} .card h4 {{ font-size:17px; }}
.meta {{ display:flex; flex-wrap:wrap; gap:6px 16px; color:var(--mute); font-size:12px; margin:6px 0 10px; }}
.nums {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(110px, 1fr)); gap:8px; }} .nums div {{ border:1px solid var(--line); border-radius:6px; padding:6px 10px; }}
.nums b {{ font:600 18px var(--f-d); font-variant-numeric:tabular-nums; display:block; }} .nums small {{ color:var(--mute); font-size:11px; }}
h5 {{ font:600 13px var(--f-b); margin:14px 0 4px; }} .card p {{ margin:0; max-width:110ch; }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:12px; }} ul, ol {{ margin:4px 0; padding-left:20px; }}
table.log td {{ font-size:12px; }} td.t {{ font-family:var(--f-m); white-space:nowrap; color:var(--mute); }}
.blk {{ color:var(--bad); margin-top:10px !important; }} code {{ font-family:var(--f-m); font-size:12px; word-break:break-all; }}
.ev {{ font-size:12px; }} .find li {{ margin:4px 0; }}
</style>
<div class="wrap">
<h1>K3.1 VR200 Kernel Campaign</h1>
<p class="sub">{E(camp.get('goal'))}</p>
<p class="sub">Rendered {E(now)} from the workstream status files. Gap closed = (baseline − now) / (baseline − model achievable). Targets are at most 1.1× model achievable.</p>
<div class="kpis">{steps}</div>
<h2>Key findings</h2><ul class="find">{findings or '<li>None yet.</li>'}</ul>
<h2>Workstreams</h2>
{table('infra', 'Infrastructure and enablement')}{table('both', 'Shared (prefill and decode)')}{table('decode', 'Decode: DCP4 × DP8, EP32')}{table('prefill', 'Prefill: DEP16')}
<h2>Node pool (Rhea)</h2><p class="sub">{E(pool.get('note'))}</p>
<div class="tw"><table><thead><tr><th>Job</th><th>Name</th><th>State</th><th>Nodes</th><th>Start</th><th>Time left</th></tr></thead><tbody>{holds or '<tr><td colspan=6>No holds.</td></tr>'}</tbody></table></div>
<h2>Workstream details</h2>
{''.join(card(s) for s in ss)}
</div>"""
    with open(f"{ROOT}/tracker.html", "w") as f:
        f.write(page)
    print(f"wrote {ROOT}/tracker.html ({len(ss)} workstreams)")


if __name__ == "__main__":
    main()
