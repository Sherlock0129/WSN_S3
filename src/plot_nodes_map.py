import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import re

EARTH_RADIUS = 6371000  # meters

# --- Parsers ---
coord_re = re.compile(r'(-?[\d\.]+ -?[\d\.]+ -?[\d\.]+)')

def parse_wkt_points(wkt: str):
    return [list(map(float, s.split())) for s in coord_re.findall(wkt)]

def load_points(file_path: str, id_field_first: str = 'name', id_field_second: str = 'id'):
    df = pd.read_csv(file_path)
    pts = []
    for _, row in df.iterrows():
        pid = row.get(id_field_first, row.get(id_field_second))
        for p in parse_wkt_points(row['WKT']):
            pts.append({'id': str(pid), 'lon': p[0], 'lat': p[1], 'h': p[2]})
    return pts

# --- Geo -> Local meters (ENU) ---

def geodetic_to_enu(lon, lat, h, lon0, lat0, h0):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon0_rad = np.deg2rad(lon0)
    lat0_rad = np.deg2rad(lat0)
    x = (lon_rad - lon0_rad) * np.cos(lat0_rad) * EARTH_RADIUS
    y = (lat_rad - lat0_rad) * EARTH_RADIUS
    z = h - h0
    return np.array([x, y, z])

# --- Load data ---
s3_pts = load_points('src/data/S3.csv')
sink_pts = load_points('src/data/sink.csv')
lake_rows = pd.read_csv('src/data/LAKE.csv')

# Find origin node 0 from S3.csv
node0 = next((p for p in s3_pts if p['id'] == '0'), None)
if node0 is None:
    raise RuntimeError("Node '0' not found in S3.csv")

lon0, lat0, h0 = node0['lon'], node0['lat'], node0['h']

# Convert sink points to ENU
sink_pts_enu = []
for d in sink_pts:
    enu = geodetic_to_enu(d['lon'], d['lat'], d['h'], lon0, lat0, h0)
    sink_pts_enu.append({**d, 'enu': enu})

# Prepare categories
ris_pts = [p for p in sink_pts_enu if p['id'].upper().startswith('RIS') or p['id'].upper().startswith('PIS')]
rf_pts = [p for p in sink_pts_enu if p['id'].upper().startswith('RF')]
sink_node = [p for p in sink_pts_enu if p['id'].upper() == 'SINK']

# Convert lakes to ENU polygons
lake_traces = []
palette = ['#99d8c9','#a6bddb','#c7e9b4','#b3cde3','#cbe8f6','#d0f0c0']
for i, row in lake_rows.iterrows():
    name = str(row['name'])
    coords = parse_wkt_points(row['WKT'])
    if not coords:
        continue
    # lon,lat from polygon
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    enu = np.array([geodetic_to_enu(lon, lat, 0.0, lon0, lat0, h0) for lon, lat in zip(lons, lats)])
    x = enu[:,0]
    y = enu[:,1]
    lake_traces.append(go.Scatter(x=x, y=y, mode='lines', fill='toself', name=name,
                                  line=dict(color='rgba(50,120,180,0.6)', width=1),
                                  fillcolor=palette[i % len(palette)], opacity=0.35, hoverinfo='name'))

# Build figure
fig = go.Figure()

# Lakes first (filled polygons)
for tr in lake_traces:
    fig.add_trace(tr)

# RIS markers
if ris_pts:
    rx = [p['enu'][0] for p in ris_pts]
    ry = [p['enu'][1] for p in ris_pts]
    labels = [p['id'] for p in ris_pts]
    fig.add_trace(go.Scatter(x=rx, y=ry, mode='markers+text', name='RIS',
                             marker=dict(symbol='triangle-up', size=10, color='#1f77b4'),
                             text=labels, textposition='top center'))

# RF cluster heads
if rf_pts:
    rx = [p['enu'][0] for p in rf_pts]
    ry = [p['enu'][1] for p in rf_pts]
    labels = [p['id'] for p in rf_pts]
    fig.add_trace(go.Scatter(x=rx, y=ry, mode='markers+text', name='RF cluster head',
                             marker=dict(symbol='square', size=10, color='#d62728'),
                             text=labels, textposition='top center'))

# SINK
if sink_node:
    sx = [p['enu'][0] for p in sink_node]
    sy = [p['enu'][1] for p in sink_node]
    fig.add_trace(go.Scatter(x=sx, y=sy, mode='markers+text', name='SINK',
                             marker=dict(symbol='star', size=14, color='#2ca02c'),
                             text=['SINK'], textposition='bottom center'))

# Origin (node 0) for reference
fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', name='Origin (Node 0)',
                         marker=dict(symbol='cross', size=12, color='black'),
                         text=['Node 0'], textposition='bottom right'))

fig.update_layout(
    title='RIS / RF Cluster Heads / SINK with Lakes (meters, origin = Node 0)',
    xaxis=dict(title='X (m)', scaleanchor='y', scaleratio=1, zeroline=True),
    yaxis=dict(title='Y (m)', zeroline=True),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    margin=dict(l=40, r=40, t=60, b=40)
)

fig.write_html('ris_rf_sink_map.html')
print("Saved: ris_rf_sink_map.html")

# Also export a static PNG for the paper (requires plotly-kaleido)
paper_fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'paper', 'sections', 'figures')
try:
    os.makedirs(paper_fig_dir, exist_ok=True)
    png_path = os.path.join(paper_fig_dir, 'ris_rf_sink_map.png')
    pio.write_image(fig, png_path, format='png', width=1200, height=900, scale=2)
    print(f"Saved paper figure: {png_path}")
except Exception as e:
    print(f"PNG export failed (install kaleido to enable static export): {e}")

