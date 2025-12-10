import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Delaunay
import re

# --- Constants ---
EARTH_RADIUS = 6371000  # meters

def parse_wkt(wkt_string):
    coords = re.findall(r'(-?[\d\.]+ -?[\d\.]+ -?[\d\.]+)', wkt_string)
    points = [list(map(float, c.split())) for c in coords]
    return points

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    all_points = []
    for index, row in df.iterrows():
        points = parse_wkt(row['WKT'])
        for p in points:
            point_id = row.get('name', row.get('id', f'point_{index}'))
            # Input is Lon, Lat, Height
            all_points.append({'id': point_id, 'lon': p[0], 'lat': p[1], 'h': p[2]})
    return all_points

def geodetic_to_enu(lon, lat, h, lon_orig, lat_orig, h_orig):
    """Converts Geodetic coords (lon, lat, h) to local East, North, Up (ENU) coords in meters."""
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon_orig_rad = np.deg2rad(lon_orig)
    lat_orig_rad = np.deg2rad(lat_orig)

    # Using equirectangular projection for local XY
    x = (lon_rad - lon_orig_rad) * np.cos(lat_orig_rad) * EARTH_RADIUS
    y = (lat_rad - lat_orig_rad) * EARTH_RADIUS
    z = h - h_orig
    return np.array([x, y, z])

def point_in_polygon(point_xy, polygon_xy):
    """Ray casting algorithm to determine if a 2D point is inside a polygon.
    polygon_xy: Nx2 array, should be closed or open; function will handle closure.
    """
    x, y = point_xy
    poly = np.array(polygon_xy)
    if len(poly) < 3:
        return False
    # Ensure closed polygon by appending first point if needed
    if not (np.isclose(poly[0, 0], poly[-1, 0]) and np.isclose(poly[0, 1], poly[-1, 1])):
        poly = np.vstack([poly, poly[0]])
    inside = False
    for i in range(len(poly) - 1):
        x1, y1 = poly[i]
        x2, y2 = poly[i + 1]
        # Check if point is between y1 and y2
        if ((y1 > y) != (y2 > y)):
            # Compute x coordinate of intersection of polygon edge with the horizontal ray at y
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if x < xinters:
                inside = not inside
    return inside

# --- 1. Load Geodetic Data ---
s3_data = load_data_from_csv('src/data/S3.csv')
sink_data = load_data_from_csv('src/data/sink.csv')
lake_data = load_data_from_csv('src/data/LAKE.csv')

all_data_geodetic = s3_data + sink_data + lake_data

# --- 2. Convert to Local ENU (meters) --- 
node_0 = next((item for item in all_data_geodetic if str(item['id']) == '0'), None)
if not node_0:
    raise ValueError("Could not find origin node '0' in the data.")

lon_0, lat_0, h_0 = node_0['lon'], node_0['lat'], node_0['h']

all_data_enu = []
for d in all_data_geodetic:
    enu_coords = geodetic_to_enu(d['lon'], d['lat'], d['h'], lon_0, lat_0, h_0)
    all_data_enu.append({'id': d['id'], 'coords': enu_coords})

# --- 3. Define Final Coordinate System (in meters) ---
# Z axis remains vertical
z_axis = np.array([0, 0, 1])

# X axis is the horizontal direction from origin (0,0,0) to node -1
node_neg_1_enu = next((item for item in all_data_enu if str(item['id']) == '-1'), None)
if not node_neg_1_enu:
    raise ValueError("Could not find node '-1' in the ENU data.")

x_vec_3d = node_neg_1_enu['coords']  # Already relative to origin
x_vec_horizontal = np.array([x_vec_3d[0], x_vec_3d[1], 0])
if np.linalg.norm(x_vec_horizontal) == 0:
    raise ValueError("Nodes 0 and -1 produce zero horizontal vector. Check input data.")
x_axis = x_vec_horizontal / np.linalg.norm(x_vec_horizontal)

# Y axis is orthogonal to Z and X
y_axis = np.cross(z_axis, x_axis)

rotation_matrix = np.linalg.inv(np.array([x_axis, y_axis, z_axis]).T)

# --- 4. Apply Final Rotation ---
for d in all_data_enu:
    d['final_coords'] = (rotation_matrix @ d['coords'].T).T

# --- 5. For each lake, set Z to the minimum ground point inside the polygon ---
# Build helper collections
sampling_points = [d for d in all_data_enu if 'lake' not in str(d['id']).lower()]
lake_ids = []
for d in all_data_geodetic:
    if 'lake' in str(d['id']).lower():
        if d['id'] not in lake_ids:
            lake_ids.append(d['id'])

lake_elevations = {}
for lake_id in lake_ids:
    # Collect polygon vertices for this lake in final coords
    poly_vertices = [d['final_coords'][:2] for d in all_data_enu if d['id'] == lake_id]
    poly = np.array(poly_vertices)
    if len(poly) == 0:
        continue
    # Remove duplicate closing point if present
    if len(poly) >= 2 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]

    # Gather sampling points inside the polygon
    inside_z = []
    for p in sampling_points:
        pt_xy = p['final_coords'][:2]
        if point_in_polygon(pt_xy, poly):
            inside_z.append(p['final_coords'][2])

    if inside_z:
        lake_z = float(np.min(inside_z))
    else:
        # Fallback: use min Z among points within polygon bounding box
        minx, miny = poly.min(axis=0)
        maxx, maxy = poly.max(axis=0)
        bb_z = [p['final_coords'][2] for p in sampling_points if (minx <= p['final_coords'][0] <= maxx and miny <= p['final_coords'][1] <= maxy)]
        lake_z = float(np.min(bb_z)) if bb_z else 0.0

    lake_elevations[lake_id] = lake_z

    # Apply lake elevation to all vertices of this lake
    for d in all_data_enu:
        if d['id'] == lake_id:
            d['final_coords'][2] = lake_z

# --- 6. Prepare visualization arrays ---
final_points = np.array([d['final_coords'] for d in all_data_enu])

# --- 7. Report farthest point in XY plane (non-lake) ---
xy_distances = np.linalg.norm(np.array([d['final_coords'][:2] for d in sampling_points]), axis=1)
farthest_point_index = int(np.argmax(xy_distances))
farthest_point = sampling_points[farthest_point_index]

print(f"The farthest sampling point in the XY plane is: Node '{farthest_point['id']}'")
print(f"Distance: {xy_distances[farthest_point_index]:.2f} meters")

# Report lake elevations
for lid, lz in lake_elevations.items():
    print(f"Lake '{lid}' elevation (min inside polygon): {lz:.2f} m")

# --- 8. Create and Save Visualizations ---
fig_cloud = go.Figure(data=[go.Scatter3d(
    x=final_points[:, 0], y=final_points[:, 1], z=final_points[:, 2],
    mode='markers', marker=dict(size=2, color=final_points[:, 2], colorscale='Viridis', opacity=0.85)
)])
fig_cloud.update_layout(title='Point Cloud (Lakes at min ground elevation)', scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'))

tri2d = Delaunay(final_points[:, :2])
fig_mesh = go.Figure(data=[go.Mesh3d(
    x=final_points[:, 0], y=final_points[:, 1], z=final_points[:, 2],
    i=tri2d.simplices[:, 0], j=tri2d.simplices[:, 1], k=tri2d.simplices[:, 2],
    colorscale='Viridis', intensity=final_points[:, 2], showscale=True
)])
fig_mesh.update_layout(title='Surface Model (Lakes at min ground elevation)', scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'))

fig_cloud.write_html("lakes_min_point_cloud.html")
fig_mesh.write_html("lakes_min_surface_model.html")

print("Updated lakes to minimum ground elevation. Check 'lakes_min_point_cloud.html' and 'lakes_min_surface_model.html'.")
