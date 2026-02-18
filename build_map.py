"""
Generate an interactive HTML map of professor embeddings.

Usage:
    uv run python build_map.py
    uv run python build_map.py --output map.html
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import hdbscan
from collections import Counter

from db import get_all_professors, DEFAULT_DB_PATH


def normalize_keywords(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item is not None]
            return [str(parsed)] if parsed is not None else []
        except json.JSONDecodeError:
            return [value]
    return [str(value)]


def main():
    parser = argparse.ArgumentParser(description="Build interactive professor map")
    parser.add_argument("--output", default="docs/index.html", help="Output HTML file")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    args = parser.parse_args()

    # Load data
    print("Loading professors from DB...")
    profs = get_all_professors(Path(args.db))
    df = pd.DataFrame(profs)

    # Filter to those with embeddings AND non-empty expertise text.
    # Stub profiles (no expertise_raw) get an identical embedding of "" which
    # causes them to cluster into a meaningless corner of the UMAP.
    has_embedding = df["embedding"].notna()
    has_expertise = df["expertise_raw"].notna() & (df["expertise_raw"].str.strip() != "")
    mask = has_embedding & has_expertise
    embeddings = np.array([
        np.frombuffer(row["embedding"], dtype=np.float32)
        for _, row in df[mask].iterrows()
    ])
    df = df[mask].reset_index(drop=True)
    print(f"  {len(df)} professors with embeddings and expertise text")

    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)

    # Build JSON data for the page
    df["expertise_keywords"] = df["expertise_keywords"].apply(normalize_keywords)
    df["keywords_str"] = df["expertise_keywords"].apply(lambda v: ", ".join(v))

    points = []
    for i, row in df.iterrows():
        # Truncate expertise_raw to ~2 sentences for the tooltip
        raw = row.get("expertise_raw")
        raw = str(raw) if raw and not (isinstance(raw, float)) else ""
        if len(raw) > 250:
            # Cut at last sentence boundary before 250 chars
            cut = raw[:250].rfind(".")
            raw = raw[: cut + 1] if cut > 50 else raw[:250] + "…"
        points.append({
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "name": row["name"],
            "school": row["school"] or "",
            "faculty": row["faculty"] or "",
            "department": row["department"] or "",
            "bio": raw,
            "keywords": row["keywords_str"],
            "email": row["email"] or "",
            "url": row["profile_url"] or "",
        })

    data_json = json.dumps(points)

    # Cluster the UMAP coordinates and generate labels
    print("Clustering...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
    cluster_labels = clusterer.fit_predict(coords)

    # For each cluster, find the most distinctive keywords
    # First, build global keyword frequency
    all_kw_counts = Counter()
    for kws in df["expertise_keywords"]:
        for kw in kws:
            if isinstance(kw, str) and kw.strip():
                all_kw_counts[kw.lower().strip()] += 1

    cluster_annotations = []
    for cid in sorted(set(cluster_labels)):
        if cid == -1:  # noise
            continue
        mask = cluster_labels == cid
        cluster_coords = coords[mask]
        cx = float(np.median(cluster_coords[:, 0]))
        cy = float(np.median(cluster_coords[:, 1]))

        # Count keywords within this cluster
        cluster_kw = Counter()
        for idx in np.where(mask)[0]:
            for kw in df.iloc[idx]["expertise_keywords"]:
                if isinstance(kw, str) and kw.strip():
                    cluster_kw[kw.lower().strip()] += 1

        # Score keywords by tf-idf-like ratio: cluster frequency / global frequency
        # Prefer keywords that are common in this cluster but not everywhere
        cluster_size = mask.sum()
        total = len(df)
        scored = []
        for kw, count in cluster_kw.items():
            if count < 3:  # need at least 3 occurrences in cluster
                continue
            tf = count / cluster_size  # how common in cluster
            df_ratio = all_kw_counts[kw] / total  # how common globally
            score = tf / (df_ratio + 0.001)  # distinctiveness
            scored.append((kw, score, count))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Pick top 2-3 keywords for the label
        label_parts = [str(kw).title() for kw, _, _ in scored[:3]]
        if not label_parts:
            # Fallback: most common department in cluster
            dept_counts = Counter()
            for idx in np.where(mask)[0]:
                dept = df.iloc[idx]["department"]
                if isinstance(dept, str) and dept.strip():
                    dept_counts[dept] += 1
            if dept_counts:
                label_parts = [dept_counts.most_common(1)[0][0]]
            else:
                continue

        label = " / ".join(label_parts)
        cluster_annotations.append({
            "x": round(cx, 4),
            "y": round(cy, 4),
            "label": label,
            "size": int(cluster_size),
        })

    annotations_json = json.dumps(cluster_annotations)
    print(f"  {len(cluster_annotations)} cluster labels generated")

    # Get unique schools for color mapping
    schools = sorted(df["school"].dropna().unique().tolist())
    school_colors = {}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
               "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]
    for i, s in enumerate(schools):
        school_colors[s] = palette[i % len(palette)]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Expert Finder Map</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f0f0f; color: #e0e0e0; }}
  #header {{ padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }}
  #header h1 {{ font-size: 18px; font-weight: 600; }}
  #header .count {{ font-size: 14px; color: #888; }}
  #controls {{ padding: 0 24px 12px; display: flex; gap: 12px; align-items: center; }}
  #controls label {{ font-size: 13px; color: #aaa; }}
  #controls select {{ background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 4px 8px; border-radius: 4px; font-size: 13px; }}
  #search {{ background: #1a1a1a; color: #e0e0e0; border: 1px solid #333; padding: 4px 10px; border-radius: 4px; font-size: 13px; width: 300px; margin-left: 16px; }}
  #search::placeholder {{ color: #555; }}
  #searchStatus {{ font-size: 12px; color: #888; min-width: 70px; }}
  #plot {{ width: 100vw; height: calc(100vh - 90px); }}
</style>
</head>
<body>
<div id="header">
  <h1>Expert Finder Map</h1>
  <span class="count" id="count"></span>
</div>
<div id="controls">
  <label>Color by:</label>
  <select id="colorBy">
    <option value="school">School</option>
    <option value="faculty">Faculty</option>
  </select>
  <input type="text" id="search" placeholder="Search by name, keyword or department…" autocomplete="off" spellcheck="false">
  <span id="searchStatus"></span>
</div>
<div id="plot"></div>
<script>
const DATA = {data_json};
const SCHOOL_COLORS = {json.dumps(school_colors)};
const CLUSTERS = {annotations_json};

function wrapText(str, maxLen) {{
  if (!str) return str;
  const words = str.split(" ");
  const lines = [];
  let line = "";
  for (const word of words) {{
    if ((line + " " + word).trim().length > maxLen && line) {{
      lines.push(line);
      line = word;
    }} else {{
      line = (line + " " + word).trim();
    }}
  }}
  if (line) lines.push(line);
  return lines.join("<br>");
}}

function getTraces(colorField) {{
  const groups = {{}};
  DATA.forEach(p => {{
    const key = p[colorField] || "Unknown";
    if (!groups[key]) groups[key] = {{ x: [], y: [], text: [], urls: [] }};
    groups[key].x.push(p.x);
    groups[key].y.push(p.y);
    const bio = p.bio ? `<br><i>${{wrapText(p.bio, 55)}}</i>` : "";
    groups[key].text.push(
      `<b>${{p.name}}</b><br>${{p.school}} · ${{p.department}}${{bio}}`
    );
    groups[key].urls.push(p.url);
  }});

  return Object.entries(groups).map(([name, g], i) => ({{
    x: g.x, y: g.y,
    text: g.text,
    customdata: g.urls,
    type: "scattergl",
    mode: "markers",
    name: name,
    marker: {{
      size: 5,
      opacity: 0.7,
      color: colorField === "school" ? (SCHOOL_COLORS[name] || "#888") : undefined,
    }},
    hovertemplate: "%{{text}}<extra></extra>",
  }}));
}}

const layout = {{
  paper_bgcolor: "#0f0f0f",
  plot_bgcolor: "#0f0f0f",
  font: {{ color: "#ccc", size: 11 }},
  margin: {{ l: 0, r: 0, t: 0, b: 0 }},
  xaxis: {{ visible: false }},
  yaxis: {{ visible: false }},
  legend: {{ bgcolor: "rgba(0,0,0,0.6)", x: 0.01, y: 0.99 }},
  hoverlabel: {{
    namelength: -1,
    bgcolor: "#1e1e2e",
    bordercolor: "#555",
    align: "left",
    font: {{ size: 12, color: "#e0e0e0",
             family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" }},
  }},
  hovermode: "closest",
  dragmode: "pan",
  annotations: CLUSTERS.map(c => ({{
    x: c.x, y: c.y,
    text: c.label,
    showarrow: false,
    font: {{ color: "rgba(255,255,255,0.75)", size: 11 }},
    bgcolor: "rgba(0,0,0,0.5)",
    borderpad: 3,
  }})),
}};

const config = {{
  scrollZoom: true,
  displayModeBar: false,
  responsive: true,
}};

document.getElementById("count").textContent = DATA.length.toLocaleString() + " professors";

// Highlight trace — always the last trace, targeted via index -1
const hlTrace = {{
  x: [], y: [], text: [],
  type: "scattergl", mode: "markers", name: "",
  marker: {{ size: 14, color: "rgba(0,0,0,0)", symbol: "circle-open",
             line: {{ width: 3, color: "#FFD700" }} }},
  hovertemplate: "%{{text}}<extra></extra>",
  showlegend: false,
}};

function getAllTraces(colorField) {{
  return [...getTraces(colorField), hlTrace];
}}

Plotly.newPlot("plot", getAllTraces("school"), layout, config);

document.getElementById("colorBy").addEventListener("change", (e) => {{
  Plotly.react("plot", getAllTraces(e.target.value), layout);
}});

// Search
let searchTimer = null;
function doSearch(query) {{
  const status = document.getElementById("searchStatus");
  const q = query.trim().toLowerCase();
  if (!q) {{
    hlTrace.x = []; hlTrace.y = []; hlTrace.text = [];
    Plotly.restyle("plot", {{ x: [[]], y: [[]], text: [[]] }}, [-1]);
    status.textContent = "";
    return;
  }}
  const matches = DATA.filter(p =>
    p.name.toLowerCase().includes(q) ||
    p.keywords.toLowerCase().includes(q) ||
    p.department.toLowerCase().includes(q)
  );
  status.textContent = matches.length
    ? `${{matches.length}} match${{matches.length !== 1 ? "es" : ""}}`
    : "No matches";
  hlTrace.x = matches.map(m => m.x);
  hlTrace.y = matches.map(m => m.y);
  hlTrace.text = matches.map(m => `<b>${{m.name}}</b><br>${{m.school}} · ${{m.department}}`);
  Plotly.restyle("plot", {{ x: [hlTrace.x], y: [hlTrace.y], text: [hlTrace.text] }}, [-1]);
  if (!matches.length) return;
  // Zoom to fit all matches with padding
  const xs = matches.map(m => m.x), ys = matches.map(m => m.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const xpad = Math.max((xmax - xmin) * 0.4, 2.5);
  const ypad = Math.max((ymax - ymin) * 0.4, 2.5);
  Plotly.relayout("plot", {{
    "xaxis.range": [xmin - xpad, xmax + xpad],
    "yaxis.range": [ymin - ypad, ymax + ypad],
  }});
}}

document.getElementById("search").addEventListener("input", (e) => {{
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => doSearch(e.target.value), 150);
}});
document.getElementById("search").addEventListener("keydown", (e) => {{
  if (e.key === "Escape") {{ e.target.value = ""; doSearch(""); }}
}});

// Click to open profile
document.getElementById("plot").on("plotly_click", (data) => {{
  const url = data.points[0].customdata;
  if (url) window.open(url, "_blank");
}});
</script>
</body>
</html>"""

    output_path = Path(args.output)
    output_path.write_text(html)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
