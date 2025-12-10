import pandas as pd
import numpy as np
import re
import pyvista as pv
from scipy.spatial import Delaunay
import os

def parse_wkt_point(wkt_string):
    """Parses a WKT POINT Z string and returns a tuple of (lon, lat, alt)."""
    match = re.search(r'POINT Z \(([-0-9.]+) ([-0-9.]+) ([-0-9.]+)\)', wkt_string)
    if match:
        lon = float(match.group(1))
        lat = float(match.group(2))
        alt = float(match.group(3))
        return lon, lat, alt
    return None

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance between two points on the earth."""
    R = 6371000  # Earth radius in meters
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

def convert_geo_to_cartesian(points, origin_point):
    """Converts geographic coordinates (lon, lat) to Cartesian coordinates (x, y) in meters."""
    cartesian_points = []
    origin_lon, origin_lat, _ = origin_point

    for lon, lat, alt in points:
        x = haversine(origin_lon, origin_lat, lon, origin_lat)
        if lon < origin_lon:
            x = -x

        y = haversine(origin_lon, origin_lat, origin_lon, lat)
        if lat < origin_lat:
            y = -y
            
        cartesian_points.append([x, y, alt])
        
    return np.array(cartesian_points)

def create_3d_model(csv_path):
    """Creates a 3D mountain model from CSV data with custom coordinate system and inverted Y-axis."""
    # 1. Read and parse data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
        print("Please ensure 'S3 (3).csv' is located in the 'src/data' directory.")
        return
        
    df['coords'] = df['WKT'].apply(parse_wkt_point)
    df = df.dropna(subset=['coords'])
    df['name'] = df['name'].astype(str)

    # 2. Identify key points
    try:
        origin_geo = df[df['name'] == '0']['coords'].iloc[0]
        px_geo = df[df['name'] == '-1']['coords'].iloc[0]
        py_geo = df[df['name'] == '-2']['coords'].iloc[0]
    except IndexError as e:
        print(f"Error: Could not find one of the key points (0, -1, -2). {e}")
        return

    all_points_geo = np.array(df['coords'].tolist())

    # 3. Coordinate System Transformation
    temp_cartesian_points = convert_geo_to_cartesian(all_points_geo, origin_geo)
    
    origin_cartesian_temp = convert_geo_to_cartesian([origin_geo], origin_geo)[0]
    px_cartesian_temp = convert_geo_to_cartesian([px_geo], origin_geo)[0]
    py_cartesian_temp = convert_geo_to_cartesian([py_geo], origin_geo)[0]

    x_axis = px_cartesian_temp[:2] - origin_cartesian_temp[:2]
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis_initial = py_cartesian_temp[:2] - origin_cartesian_temp[:2]
    y_axis = y_axis_initial - np.dot(y_axis_initial, x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)

    rotation_matrix = np.array([x_axis, y_axis])

    final_points = np.zeros_like(temp_cartesian_points)
    final_points[:, :2] = np.dot(temp_cartesian_points[:, :2], rotation_matrix.T)
    final_points[:, 2] = temp_cartesian_points[:, 2]

    # 4. Invert the Y-axis for upside-down view
    final_points[:, 1] *= -1

    # 5. Create mesh
    points_2d = final_points[:, :2]
    tri = Delaunay(points_2d)
    mesh = pv.PolyData(final_points, faces=np.insert(tri.simplices, 0, 3, axis=1))
    mesh['elevation'] = final_points[:, 2]

    # 6. Visualize and Save
    warp_factor = 0.5
    warped = mesh.warp_by_scalar('elevation', factor=warp_factor)

    output_filename = 'mountain_model.obj'
    warped.save(output_filename)
    print(f"Model saved to {output_filename}")

    plotter = pv.Plotter()
    plotter.add_mesh(warped, show_edges=True, cmap='terrain')
    plotter.show_axes()
    plotter.add_text('Custom Coordinate System - Inverted Y', position='upper_left', font_size=18)
    print("Displaying 3D model. Close the window to exit.")
    plotter.show()

if __name__ == '__main__':
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming 'data' is a subdirectory of the 'src' directory
        csv_file_path = os.path.join(script_dir, 'data', 'S3 (3).csv')
        create_3d_model(csv_file_path)
    except NameError:
        # Fallback for interactive environments where __file__ is not defined
        print("Running in an interactive environment. Using relative path 'src/data/S3 (3).csv'.")
        create_3d_model('src/data/S3 (3).csv')
