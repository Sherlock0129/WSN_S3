import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

S3_PATH = 'src/data/S3.csv'
SINK_PATH = 'src/data/sink.csv'
OUT_DIR = 'outputs'

WKT_RE = re.compile(r'POINT Z \(([-+\d\.eE]+) ([-+\d\.eE]+) ([-+\d\.eE]+)\)')


def parse_wkt_point_z(wkt: str):
    m = WKT_RE.match(wkt.strip())
    if not m:
        raise ValueError(f'Cannot parse WKT: {wkt}')
    lon, lat, z = map(float, m.groups())
    return lon, lat, z


def load_points(*csv_paths):
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p)
        for w in df['WKT']:
            rows.append(parse_wkt_point_z(w))
    arr = np.array(rows)  # (N,3) lon,lat,z
    return arr


def meters_per_degree(lat_deg):
    lat_rad = math.radians(lat_deg)
    m_per_deg_lat = 111_132.92 - 559.82 * math.cos(2*lat_rad) + 1.175 * math.cos(4*lat_rad) - 0.0023 * math.cos(6*lat_rad)
    m_per_deg_lon = 111_412.84 * math.cos(lat_rad) - 93.5 * math.cos(3*lat_rad) + 0.118 * math.cos(5*lat_rad)
    return m_per_deg_lon, m_per_deg_lat


def idw_interpolate(lon, lat, elev, grid_lon, grid_lat, power=2.0, eps=1e-9):
    # Use vectorized IDW (broadcasting). Distances computed with anisotropic scaling to meters.
    lat0 = np.nanmean(lat)
    m_per_deg_lon, m_per_deg_lat = meters_per_degree(lat0)

    # shape: (N,1,1)
    lon_s = lon.reshape(-1, 1, 1)
    lat_s = lat.reshape(-1, 1, 1)

    # shape: (1,H,W)
    glon = grid_lon[None, :, :]
    glat = grid_lat[None, :, :]

    dx = (glon - lon_s) * m_per_deg_lon
    dy = (glat - lat_s) * m_per_deg_lat
    d = np.hypot(dx, dy)

    # Handle exact sample locations to avoid division by zero
    close = d < 1e-6
    if np.any(close):
        # For exact matches, assign the sample elevation directly
        elev_s = elev.reshape(-1, 1, 1)
        z = np.sum(elev_s * close, axis=0)
        mask = np.any(close, axis=0)
    else:
        z = np.zeros_like(grid_lon)
        mask = np.zeros_like(grid_lon, dtype=bool)

    w = 1.0 / np.power(d + eps, power)
    weights = np.sum(w, axis=0)
    weighted = np.sum(w * elev.reshape(-1,1,1), axis=0)
    z_idw = weighted / np.maximum(weights, eps)
    z[~mask] = z_idw[~mask]
    return z


def triangulated_dem(lon, lat, elev, grid_lon, grid_lat):
    # Build TIN in a local meter-based plane and rasterize via linear interpolation
    lat0 = float(np.nanmean(lat))
    lon0 = float(np.nanmean(lon))
    m_per_deg_lon, m_per_deg_lat = meters_per_degree(lat0)
    X = (lon - lon0) * m_per_deg_lon
    Y = (lat - lat0) * m_per_deg_lat
    tri = Triangulation(X, Y)
    interp = LinearTriInterpolator(tri, elev)
    GX = (grid_lon - lon0) * m_per_deg_lon
    GY = (grid_lat - lat0) * m_per_deg_lat
    Z = interp(GX, GY)
    dem = np.array(Z)
    # Fill outside convex hull with nearest neighbor
    mask = np.isnan(dem)
    if np.any(mask):
        xq = GX[mask].reshape(-1, 1)
        yq = GY[mask].reshape(-1, 1)
        # distances to samples (Nq x Np)
        d2 = (xq - X.reshape(1, -1))**2 + (yq - Y.reshape(1, -1))**2
        idx = np.argmin(d2, axis=1)
        dem_flat = dem.reshape(-1)
        mask_flat = mask.reshape(-1)
        dem_flat[np.where(mask_flat)[0]] = elev[idx]
        dem = dem_flat.reshape(dem.shape)
    return dem

def solar_declination_winter_solstice():
    # Mean obliquity and declination at December solstice ~ -23.44 deg
    return math.radians(-23.44)


def solar_alt_az(phi_rad, H_rad, delta_rad):
    # Returns altitude (rad) and azimuth A (rad, from North clockwise)
    sin_h = math.sin(phi_rad) * math.sin(delta_rad) + math.cos(phi_rad) * math.cos(delta_rad) * math.cos(H_rad)
    sin_h = max(-1.0, min(1.0, sin_h))
    h = math.asin(sin_h)
    # Zenith
    cos_theta_z = math.cos(math.pi/2 - h)
    sin_theta_z = math.sin(math.pi/2 - h)
    # Avoid divide-by-zero near zenith
    if sin_theta_z < 1e-9:
        A = 0.0
    else:
        sinA = (math.cos(delta_rad) * math.sin(H_rad)) / sin_theta_z
        cosA = (math.sin(delta_rad) - math.sin(phi_rad) * sin_h) / (math.cos(phi_rad) * sin_theta_z)
        A = math.atan2(sinA, cosA)
        if A < 0:
            A += 2*math.pi
    return h, A


def slope_normal(beta_deg, aspect_deg):
    # beta: tilt from horizontal (deg), aspect: clockwise from North (deg)
    b = math.radians(beta_deg)
    g = math.radians(aspect_deg)
    # ENU normal
    nx = math.sin(b) * math.sin(g)  # East
    ny = math.sin(b) * math.cos(g)  # North
    nz = math.cos(b)                # Up
    return np.array([nx, ny, nz])


def sun_vector(h_rad, A_rad):
    # Convert to ENU unit vector
    # A from North clockwise, h altitude
    x = math.cos(h_rad) * math.sin(A_rad)   # East
    y = math.cos(h_rad) * math.cos(A_rad)   # North
    z = math.sin(h_rad)                     # Up
    return np.array([x, y, z])


def relative_irradiance_over_day(phi_deg, hours_from_noon, beta_deg, aspect_deg, delta_rad):
    phi = math.radians(phi_deg)
    irr = []
    for t in hours_from_noon:
        H = math.radians(15.0 * t)
        h, A = solar_alt_az(phi, H, delta_rad)
        s = sun_vector(h, A)
        n = slope_normal(beta_deg, aspect_deg)
        cos_i = float(np.dot(s, n))
        irr.append(max(0.0, cos_i))
    return np.array(irr)


def compute_hillshade(dem, grid_lon, grid_lat, sun_alt_deg, sun_az_deg):
    lat0 = np.nanmean(grid_lat)
    m_per_deg_lon, m_per_deg_lat = meters_per_degree(lat0)
    # Spacing in meters
    dlon = float(np.abs(grid_lon[0,1] - grid_lon[0,0]))
    dlat = float(np.abs(grid_lat[1,0] - grid_lat[0,0]))
    dx = dlon * m_per_deg_lon
    dy = dlat * m_per_deg_lat
    # Gradients: axis 1 ~ lon (x/East), axis 0 ~ lat (y/North)
    dz_dlat, dz_dlon = np.gradient(dem, dy, dx)
    dzdx = dz_dlon
    dzdy = dz_dlat
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdx, -dzdy)
    zen = math.radians(90.0 - sun_alt_deg)
    az = math.radians(sun_az_deg)
    hs = (np.cos(zen) * np.cos(slope) + np.sin(zen) * np.sin(slope) * np.cos(az - aspect))
    return hs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pts = load_points(S3_PATH, SINK_PATH)
    lon = pts[:,0]
    lat = pts[:,1]
    elev = pts[:,2]

    # Grid for DEM
    pad = 0.002
    lon_min, lon_max = lon.min()-pad, lon.max()+pad
    lat_min, lat_max = lat.min()-pad, lat.max()+pad
    nx, ny = 220, 220
    grid_lon = np.linspace(lon_min, lon_max, nx)
    grid_lat = np.linspace(lat_min, lat_max, ny)
    GLON, GLAT = np.meshgrid(grid_lon, grid_lat)

    DEM = idw_interpolate(lon, lat, elev, GLON, GLAT, power=2.0)

    # Solar geometry
    phi_deg = float(np.median(lat))
    delta = solar_declination_winter_solstice()

    hours = np.linspace(8, 16, 161)
    hours_from_noon = hours - 12.0

    altitudes = []
    for t in hours_from_noon:
        H = math.radians(15.0 * t)
        h, _A = solar_alt_az(math.radians(phi_deg), H, delta)
        altitudes.append(math.degrees(h))
    altitudes = np.array(altitudes)

    # South vs North irradiance for 30° slope
    irr_south = relative_irradiance_over_day(phi_deg, hours_from_noon, beta_deg=30.0, aspect_deg=180.0, delta_rad=delta)
    irr_north = relative_irradiance_over_day(phi_deg, hours_from_noon, beta_deg=30.0, aspect_deg=0.0, delta_rad=delta)

    # Hillshade at solar noon
    h_noon, _Az_noon = solar_alt_az(math.radians(phi_deg), 0.0, delta)
    sun_alt_deg = math.degrees(h_noon)
    sun_az_deg = 180.0  # solar noon
    HS = compute_hillshade(DEM, GLON, GLAT, sun_alt_deg, sun_az_deg)

    # Plot
    fig = plt.figure(figsize=(7, 14), dpi=140)

    # Fig 1: Solar altitude
    ax1 = plt.subplot(4,1,1)
    ax1.plot(hours, altitudes, color='#E69F00', lw=2)
    ax1.set_title('Winter Solstice Solar Altitude near 52°N')
    ax1.set_xlabel('Hour (solar local time)')
    ax1.set_ylabel('Solar altitude (deg)')
    ax1.set_xlim(8, 16)
    ax1.grid(True, alpha=0.3)

    # Fig 2: South vs North irradiance (relative cos incidence)
    ax2 = plt.subplot(4,1,2)
    ax2.plot(hours, irr_south, label='South 30°', color='#0072B2', lw=2)
    ax2.plot(hours, irr_north, label='North 30°', color='#56B4E9', lw=2)
    ax2.set_title('South vs North Slope Irradiance (approx, direct only)')
    ax2.set_xlabel('Hour (solar local time)')
    ax2.set_ylabel('Relative Irradiance (cos incidence)')
    ax2.legend()
    ax2.set_xlim(8, 16)
    ax2.grid(True, alpha=0.3)

    # Fig 3: Interpolated DEM
    ax3 = plt.subplot(4,1,3)
    im3 = ax3.pcolormesh(GLON, GLAT, DEM, shading='auto', cmap='viridis')
    ax3.set_aspect('equal')
    ax3.set_title('Interpolated DEM')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    cbar3 = plt.colorbar(im3, ax=ax3, label='Elevation (m)')

    # Fig 4: Hillshade at winter solstice noon
    ax4 = plt.subplot(4,1,4)
    im4 = ax4.pcolormesh(GLON, GLAT, HS, shading='auto', cmap='gray')
    ax4.set_aspect('equal')
    ax4.set_title(f'Hillshade Winter Solstice Noon (alt≈{sun_alt_deg:.1f}°, az=180°)')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    cbar4 = plt.colorbar(im4, ax=ax4, label='Illumination (relative)')

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, 'winter_solstice_figs.png')
    plt.savefig(out_path, dpi=160)

    # Also dump noon parameters for reference
    with open(os.path.join(OUT_DIR, 'winter_solstice_meta.txt'), 'w') as f:
        f.write(f'Center latitude (deg): {phi_deg}\n')
        f.write(f'Solar declination (deg): {-23.44}\n')
        f.write(f'Solar noon altitude (deg): {sun_alt_deg:.3f}\n')
        f.write('Azimuth at noon (deg from North, CW): 180.0\n')

    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()

