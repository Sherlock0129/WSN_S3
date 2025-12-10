import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, List

# Better CJK font fallback
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# Optional dependencies
try:
    from scipy.interpolate import griddata
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False


def parse_wkt_point_z(wkt: str) -> Tuple[float, float, float]:
    # Expecting: POINT Z (lon lat elev)
    wkt = wkt.strip()
    start = wkt.find('(')
    end = wkt.find(')')
    coords = wkt[start + 1:end].replace(',', ' ').split()
    lon, lat, z = map(float, coords[:3])
    return lon, lat, z


def load_points(csv_paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        lon, lat, z = zip(*df['WKT'].map(parse_wkt_point_z))
        df = df.assign(lon=lon, lat=lat, elev=z)
        dfs.append(df[['lon', 'lat', 'elev']])
    return pd.concat(dfs, ignore_index=True)


def latlon_to_meters(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    if _HAS_PYPROJ:
        lon0 = float(np.mean(lon))
        lat0 = float(np.mean(lat))
        zone = int(math.floor((lon0 + 180) / 6) + 1)
        epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return np.asarray(x), np.asarray(y), f"EPSG:{epsg}"
    else:
        R = 6371000.0
        lon0 = np.deg2rad(np.mean(lon))
        lat0 = np.deg2rad(np.mean(lat))
        x = R * (np.deg2rad(lon) - lon0) * math.cos(lat0)
        y = R * (np.deg2rad(lat) - lat0)
        return x, y, "local-equirect"


def interpolate_dem(xm: np.ndarray, ym: np.ndarray, z: np.ndarray, res: float = 100.0):
    pad = res
    xmin, xmax = xm.min() - pad, xm.max() + pad
    ymin, ymax = ym.min() - pad, ym.max() + pad

    nx = max(2, int(math.ceil((xmax - xmin) / res)))
    ny = max(2, int(math.ceil((ymax - ymin) / res)))

    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)

    pts = np.column_stack([xm, ym])
    if _HAS_SCIPY:
        Z = griddata(pts, z, (GX, GY), method='linear')
        Z_nn = griddata(pts, z, (GX, GY), method='nearest')
        Z = np.where(np.isnan(Z), Z_nn, Z)
    else:
        # Simple inverse distance weighting
        Z = np.zeros_like(GX)
        for i in range(GX.shape[0]):
            for j in range(GX.shape[1]):
                dx = xm - GX[i, j]
                dy = ym - GY[i, j]
                d2 = dx*dx + dy*dy
                w = 1.0 / np.maximum(d2, 1.0)  # avoid div by zero
                Z[i, j] = np.sum(w * z) / np.sum(w)
    return GX, GY, Z


def slope_aspect(Z: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    dz_dy, dz_dx = np.gradient(Z, dy, dx)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))  # radians
    aspect = np.arctan2(dz_dy, -dz_dx)
    aspect = np.where(aspect < 0, aspect + 2*np.pi, aspect)
    return slope, aspect


def hillshade_from_slope_aspect(slope: np.ndarray, aspect: np.ndarray, sun_alt_deg: float, sun_az_deg: float) -> np.ndarray:
    alt = np.deg2rad(sun_alt_deg)
    az = np.deg2rad(sun_az_deg)
    cos_i = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(cos_i, 0.0, 1.0)


def solar_altitude_curve(lat_deg: float, dec_deg: float, hours: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    delta = np.deg2rad(dec_deg)
    H = np.deg2rad(15.0 * (hours - 12.0))
    sin_h = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(H)
    h = np.rad2deg(np.arcsin(np.clip(sin_h, -1.0, 1.0)))
    return h


def sun_azimuth(lat_deg: float, dec_deg: float, hours: np.ndarray) -> np.ndarray:
    """
    Solar azimuth measured from North, clockwise [0, 360).
    Uses a common formula that natively returns azimuth from South; we convert.
    """
    phi = np.deg2rad(lat_deg)
    delta = np.deg2rad(dec_deg)
    H = np.deg2rad(15.0 * (hours - 12.0))
    # Base formula (from South, eastward positive)
    num = np.sin(H)
    den = np.cos(H) * np.sin(phi) - np.tan(delta) * np.cos(phi)
    A_south = np.arctan2(num, den)  # [-pi, pi]
    # Convert from South-based to North-based clockwise: add pi and wrap
    Az = A_south + np.pi
    Az = np.mod(Az, 2*np.pi)
    return Az


def cos_incidence_on_slope(lat_deg: float, dec_deg: float, hours: np.ndarray, slope_deg: float, aspect_deg: float) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    delta = np.deg2rad(dec_deg)
    H = np.deg2rad(15.0 * (hours - 12.0))
    sin_h = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(H)
    h = np.arcsin(np.clip(sin_h, -1.0, 1.0))
    Az = sun_azimuth(lat_deg, dec_deg, hours)
    slope = np.deg2rad(slope_deg)
    aspect = np.deg2rad(aspect_deg)
    cos_i = np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(Az - aspect)
    return np.maximum(0.0, cos_i)


def plot_radiation_figs(center_lat: float, dec_deg: float, slope_deg: float, outputs_dir: str, dpi: int, lang: str):
    hours = np.linspace(8, 16, 121)
    alt_curve = solar_altitude_curve(center_lat, dec_deg, hours)
    cos_s = cos_incidence_on_slope(center_lat, dec_deg, hours, slope_deg=slope_deg, aspect_deg=180.0)
    cos_n = cos_incidence_on_slope(center_lat, dec_deg, hours, slope_deg=slope_deg, aspect_deg=0.0)

    # Figure A: Solar Altitude
    figA = plt.figure(figsize=(6.4, 4.0), dpi=dpi)
    axA = figA.add_subplot(1, 1, 1)
    axA.plot(hours, alt_curve, color='#e69f00', lw=2.2)
    if lang == 'zh':
        axA.set_title('冬至日太阳高度角（纬度 {:.2f}°N）'.format(center_lat))
        axA.set_xlabel('小时')
        axA.set_ylabel('太阳高度角 (°)')
    else:
        axA.set_title('Winter Solstice Solar Altitude at {:.2f}°N'.format(center_lat))
        axA.set_xlabel('Hour')
        axA.set_ylabel('Solar altitude (deg)')
    axA.set_xlim(8, 16)
    axA.grid(alpha=0.3)
    pathA = os.path.join(outputs_dir, 'solar_altitude.png')
    figA.tight_layout()
    figA.savefig(pathA, bbox_inches='tight')

    # Figure B: South vs North irradiance
    figB = plt.figure(figsize=(6.4, 4.0), dpi=dpi)
    axB = figB.add_subplot(1, 1, 1)
    axB.plot(hours, cos_s, label=('南坡 {:.0f}°'.format(slope_deg) if lang == 'zh' else f'South {slope_deg:.0f}°'), color='#f0ad4e', lw=2.0)
    axB.plot(hours, cos_n, label=('北坡 {:.0f}°'.format(slope_deg) if lang == 'zh' else f'North {slope_deg:.0f}°'), color='#5bc0de', lw=2.0)
    if lang == 'zh':
        axB.set_title('南北坡相对直射（入射余弦）')
        axB.set_xlabel('小时')
        axB.set_ylabel('相对辐照（cosθ）')
    else:
        axB.set_title('South vs North Slope Irradiance (approx)')
        axB.set_xlabel('Hour')
        axB.set_ylabel('Relative Irradiance (cos incidence)')
    axB.set_xlim(8, 16)
    axB.legend()
    axB.grid(alpha=0.3)
    pathB = os.path.join(outputs_dir, 'south_north_irradiance.png')
    figB.tight_layout()
    figB.savefig(pathB, bbox_inches='tight')

    return pathA, pathB


def plot_all(points_csvs: List[str], outputs_dir: str, fig_name: str, lang: str = 'zh', slope_deg: float = 30.0, dem_res_m: float = 80.0, dpi: int = 220, separate: bool = False):
    os.makedirs(outputs_dir, exist_ok=True)

    pts = load_points(points_csvs)
    center_lat = pts['lat'].mean()

    # Solar parameters for winter solstice
    dec_deg = -23.44
    hours = np.linspace(8, 16, 121)
    alt_curve = solar_altitude_curve(center_lat, dec_deg, hours)

    # Relative irradiance on south/north slopes
    cos_s = cos_incidence_on_slope(center_lat, dec_deg, hours, slope_deg=slope_deg, aspect_deg=180.0)
    cos_n = cos_incidence_on_slope(center_lat, dec_deg, hours, slope_deg=slope_deg, aspect_deg=0.0)

    # If only radiation figs are required as separate files
    sepA, sepB = None, None
    if separate:
        sepA, sepB = plot_radiation_figs(center_lat, dec_deg, slope_deg, outputs_dir, dpi, lang)

    # DEM interpolation in meters
    xm, ym, crs = latlon_to_meters(pts['lon'].to_numpy(), pts['lat'].to_numpy())
    GX, GY, Z = interpolate_dem(xm, ym, pts['elev'].to_numpy(), res=dem_res_m)
    dx = float(GX[0, 1] - GX[0, 0])
    dy = float(GY[1, 0] - GY[0, 0])
    slope, aspect = slope_aspect(Z, dx=dx, dy=dy)

    # Hillshade at solar noon
    noon_idx = int(np.argmax(alt_curve))
    noon_alt = float(alt_curve[noon_idx])
    noon_az = float(np.rad2deg(sun_azimuth(center_lat, dec_deg, np.array([hours[noon_idx]])))[0])
    HS = hillshade_from_slope_aspect(slope, aspect, noon_alt, noon_az)

    lon_min, lon_max = pts['lon'].min(), pts['lon'].max()
    lat_min, lat_max = pts['lat'].min(), pts['lat'].max()
    extent = [lon_min, lon_max, lat_min, lat_max]

    # Composite figure (4 panels)
    fig = plt.figure(figsize=(6.4, 12), dpi=dpi)

    # 1) Solar altitude
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(hours, alt_curve, color='#e69f00', lw=2.2)
    if lang == 'zh':
        ax1.set_title('冬至日太阳高度角（纬度 {:.2f}°N）'.format(center_lat))
        ax1.set_xlabel('小时')
        ax1.set_ylabel('太阳高度角 (°)')
    else:
        ax1.set_title('Winter Solstice Solar Altitude at {:.2f}°N'.format(center_lat))
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Solar altitude (deg)')
    ax1.set_xlim(8, 16)
    ax1.grid(alpha=0.3)

    # 2) South vs North irradiance (cos incidence)
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(hours, cos_s, label=('南坡 {:.0f}°'.format(slope_deg) if lang == 'zh' else f'South {slope_deg:.0f}°'), color='#f0ad4e', lw=2.0)
    ax2.plot(hours, cos_n, label=('北坡 {:.0f}°'.format(slope_deg) if lang == 'zh' else f'North {slope_deg:.0f}°'), color='#5bc0de', lw=2.0)
    if lang == 'zh':
        ax2.set_title('南北坡相对直射（入射余弦）')
        ax2.set_xlabel('小时')
        ax2.set_ylabel('相对辐照（cosθ）')
    else:
        ax2.set_title('South vs North Slope Irradiance (approx)')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Relative Irradiance (cos incidence)')
    ax2.set_xlim(8, 16)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3) Interpolated DEM
    ax3 = fig.add_subplot(4, 1, 3)
    im3 = ax3.imshow(Z, origin='lower', extent=extent, cmap='viridis')
    if lang == 'zh':
        ax3.set_title('插值 DEM')
        ax3.set_xlabel('经度')
        ax3.set_ylabel('纬度')
    else:
        ax3.set_title('Interpolated DEM')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar3.set_label('Elevation (m)' if lang != 'zh' else '高程 (m)')

    # 4) Hillshade (noon)
    ax4 = fig.add_subplot(4, 1, 4)
    im4 = ax4.imshow(HS, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=1)
    if lang == 'zh':
        ax4.set_title('冬至正午地形阴影（余弦照度）')
        ax4.set_xlabel('经度')
        ax4.set_ylabel('纬度')
    else:
        ax4.set_title('Hillshade Winter Solstice Noon')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
    cbar4 = fig.colorbar(im4, ax=ax4)
    cbar4.set_label('Illumination (cosθ)' if lang != 'zh' else '照度 (cosθ)')

    fig.tight_layout()

    fig_path = os.path.join(outputs_dir, fig_name)
    fig.savefig(fig_path, bbox_inches='tight')

    # Save meta
    meta_path = os.path.join(outputs_dir, 'winter_solstice_meta.txt')
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f"Center latitude (deg): {center_lat}\n")
        f.write(f"Solar declination (deg): {dec_deg}\n")
        f.write(f"Solar noon altitude (deg): {noon_alt:.3f}\n")
        f.write(f"Azimuth at noon (deg from North, CW): {noon_az:.1f}\n")
        f.write(f"DEM grid CRS: {crs}, dx={dx:.1f} m, dy={dy:.1f} m\n")

    return fig_path, meta_path, sepA, sepB


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', nargs='+', default=['src/data/S3.csv', 'src/data/sink.csv'])
    parser.add_argument('--outdir', default='outputs')
    parser.add_argument('--outfile', default='winter_solstice_figs.png')
    parser.add_argument('--lang', default='zh', choices=['zh', 'en'])
    parser.add_argument('--slope', type=float, default=30.0)
    parser.add_argument('--dem_res', type=float, default=80.0)
    parser.add_argument('--dpi', type=int, default=220)
    parser.add_argument('--separate', action='store_true', help='also save two standalone figures for radiation plots')
    args = parser.parse_args()

    fig_path, meta_path, sepA, sepB = plot_all(
        args.points,
        outputs_dir=args.outdir,
        fig_name=args.outfile,
        lang=args.lang,
        slope_deg=args.slope,
        dem_res_m=args.dem_res,
        dpi=args.dpi,
        separate=args.separate,
    )
    print('Saved figure to:', fig_path)
    print('Saved meta to:', meta_path)
    if sepA:
        print('Saved solar altitude figure to:', sepA)
    if sepB:
        print('Saved south vs north irradiance figure to:', sepB)
