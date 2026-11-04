"""Build a self-contained HTML review page for the SAM auto-labels.

Inputs : labels_sam/overlays, labels_sam/flags, labels_sam/audit.csv,
         review_notes.json  ({image: {"gt": int, "note": str, "flag_notes": {crop: str}}})
Output : review page HTML (images embedded as data URIs).
"""
import base64
import csv
import html
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "labels_sam")


def b64(img, long_side, q):
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


CSS = """
:root{
  /* layout: one scan column - summary table, then one card per image; flagged crops in a strip */
  --bg:#f3f4f2;--panel:#ffffff;--ink:#1d2321;--muted:#5d6763;--line:#d9ddd9;
  --accent:#2f6b5a;--ok:#2f7a4a;--warn:#a5640c;--bad:#b23a2e;--chip:#e9eeeb;
  --display:"IBM Plex Sans Condensed",system-ui,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#141817;--panel:#1c2120;--ink:#e6ebe8;--muted:#9aa5a0;--line:#2e3532;
  --accent:#7cc3ab;--ok:#6fcf8f;--warn:#e0a54b;--bad:#ef7a6c;--chip:#253029;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141817;--panel:#1c2120;--ink:#e6ebe8;--muted:#9aa5a0;--line:#2e3532;
  --accent:#7cc3ab;--ok:#6fcf8f;--warn:#e0a54b;--bad:#ef7a6c;--chip:#253029;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-block:28px 64px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  display:flex;flex-direction:column;gap:28px}
h1{font-family:var(--display);font-weight:600;font-size:30px;margin:0;text-wrap:balance}
h2{font-family:var(--display);font-weight:600;font-size:20px;margin:0}
.lede{color:var(--muted);max-width:68ch;margin:6px 0 0}
.how{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.how div{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px;font-size:14px}
.how b{display:block;font-family:var(--display);font-size:15px}
.tbl{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:7px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-family:var(--display);font-weight:600;color:var(--muted);font-size:12px;letter-spacing:.06em;text-transform:uppercase}
td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
td.gt{font-weight:700}
a{color:var(--accent)}
.code{font-family:var(--mono);font-weight:600}
.pill{display:inline-block;font-size:12px;padding:1px 9px;border-radius:999px;background:var(--chip);font-family:var(--display);letter-spacing:.03em}
.pill.ok{color:var(--ok)}.pill.fix{color:var(--warn)}.pill.check{color:var(--bad)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px;display:flex;flex-direction:column;gap:12px;scroll-margin-top:12px}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px}
.nums{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--muted);font-size:14px;font-variant-numeric:tabular-nums}
.nums b{color:var(--ink)}
.note{margin:0;max-width:80ch}
.ov{width:100%;height:auto;border-radius:4px;border:1px solid var(--line);cursor:zoom-in}
.ov.big{cursor:zoom-out}
.ovbox{overflow:auto;max-height:80vh}
.ovbox.big .ov{width:auto;max-width:none}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}
figure{margin:0;display:flex;flex-direction:column;gap:4px}
figure img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;border:1px solid var(--line)}
figcaption{font-size:12.5px;color:var(--muted);line-height:1.35}
figcaption b{color:var(--ink)}
.legend{font-size:13px;color:var(--muted)}
button:focus-visible,img:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "review.html")
    notes = {}
    np_ = os.path.join(HERE, "review_notes.json")
    if os.path.exists(np_):
        notes = json.load(open(np_))
    rows = list(csv.DictReader(open(os.path.join(LAB, "audit.csv"))))
    order = sorted(rows, key=lambda r: (not r["image"].startswith("img"), r["image"]))
    codes = {r["image"]: f"{'S' if r['image'].startswith('img') else 'B'}{i + 1:02d}" for i, r in enumerate(order)}
    sys.path.insert(0, HERE)
    from prepare_dataset import TEST, VAL
    split = lambda n: "test" if n in TEST else "val" if n in VAL else "train"

    trs, cards = [], []
    for r in order:
        n = r["image"]
        nt = notes.get(n, {})
        gt = nt.get("gt", int(r["sam"]))
        sam = nt.get("auto", gt)  # SAM count before my corrections
        state = "check" if nt.get("unsure") else ("ok" if gt == sam else "fix")
        label = {"ok": "auto-labels correct", "fix": f"corrected {gt - sam:+d}", "check": "please check"}[state]
        holes = int(r["holes"]) if r["holes"] else 0
        trs.append(
            f'<tr><td><a class="code" href="#{codes[n]}">{codes[n]}</a></td><td>{html.escape(n)}</td><td>{split(n)}</td>'
            f'<td>{sam}</td><td>{r["non_ai"]}</td><td>{holes or "–"}</td><td class="gt">{gt}</td>'
            f'<td><span class="pill {state}">{label}</span></td></tr>')
        ov = cv2.imread(os.path.join(LAB, "overlays", n + ".jpg"))
        figs = []
        for c in [c for c in r["crops"].split(";") if c]:
            kind = c.rsplit("_", 1)[1].split(".")[0]
            fn = nt.get("flag_notes", {}).get(c, "")
            cap = f'<b>{"possible miss" if kind == "miss" else "possible merge"}</b>' + (f" – {html.escape(fn)}" if fn else "")
            figs.append(f'<figure><img src="{b64(cv2.imread(os.path.join(LAB, "flags", c)), 360, 80)}" alt="flagged spot in {codes[n]}"><figcaption>{cap}</figcaption></figure>')
        strip = f'<div class="strip">{"".join(figs)}</div>' if figs else '<p class="legend">No flagged spots.</p>'
        cards.append(f'''
<section class="card" id="{codes[n]}">
  <div class="head"><h2><span class="code">{codes[n]}</span> · {html.escape(n)}</h2><span class="pill {state}">{label}</span></div>
  <div class="nums"><span>SAM automatic <b>{sam}</b></span><span>Non-AI <b>{r["non_ai"]}</b></span>{f"<span>nut holes <b>{holes}</b></span>" if holes else ""}<span>ground truth after review <b>{gt}</b></span><span>split <b>{split(n)}</b></span></div>
  {f'<p class="note">{html.escape(nt["note"])}</p>' if nt.get("note") else ""}
  <div class="ovbox"><img class="ov" loading="lazy" src="{b64(ov, 1600, 72)}" alt="SAM masks for {codes[n]}" tabindex="0"></div>
  {strip}
</section>''')

    page = f"""<title>Counting Label Review</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600;700&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header>
    <h1>Counting Label Review</h1>
    <p class="lede">SAM 2.1 auto-labels for all 25 images, checked against the classical (Non-AI) count, nut-hole counts and an automatic audit. I reviewed every flagged spot and fixed 15 objects in 8 images with one SAM point prompt each (adds and splits). Images show the corrected masks. Only one spot needs your eyes (B14, mask 159). Reply with any image codes where you disagree; otherwise these counts become the ground truth for both solutions.</p>
  </header>
  <div class="how">
    <div><b>Possible miss (red circle)</b>Dark foreground with no mask. Often a black screw SAM skipped; sometimes only an unmasked shank of a screw whose head is masked (count unaffected).</div>
    <div><b>Possible merge (blue circle)</b>A mask much bigger than one object; may hold two touching objects.</div>
    <div><b>Status</b><span class="pill ok">auto-labels correct</span> <span class="pill fix">corrected ±n</span> <span class="pill check">please check</span><br>Click an image to view it at full size.</div>
  </div>
  <div class="tbl"><table>
    <thead><tr><th>Code</th><th>Image</th><th>Split</th><th>SAM auto</th><th>Non-AI</th><th>Holes</th><th>Ground truth</th><th>Status</th></tr></thead>
    <tbody>{"".join(trs)}</tbody></table></div>
  {"".join(cards)}
</div>
<script>
document.querySelectorAll('.ov').forEach(function(img){{
  function t(){{img.classList.toggle('big');img.parentElement.classList.toggle('big');}}
  img.addEventListener('click',t);
  img.addEventListener('keydown',function(e){{if(e.key==='Enter'||e.key===' '){{e.preventDefault();t();}}}});
}});
</script>
"""
    with open(out, "w") as f:
        f.write(page)
    print(out, round(os.path.getsize(out) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
