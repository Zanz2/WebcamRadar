"""Astronomy tool: FITS images -> voxel grid + celestial-sphere texture.

The kernel `process_image` is serial: numba's prange cannot do atomic float adds, and a racy
parallel `+=` would silently lose brightness. It is slower on many cores, but deterministic.

Needs the optional extra:  uv sync --extra space
Usage:  p2v-space [--fits-dir fits] [...]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _ray_aabb(o, d, bmin, bmax):
    tmin, tmax = -np.inf, np.inf
    for i in range(3):
        if abs(d[i]) > 1e-8:
            t1 = (bmin[i] - o[i]) / d[i]
            t2 = (bmax[i] - o[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False, 0.0, 0.0
        elif o[i] < bmin[i] or o[i] > bmax[i]:
            return False, 0.0, 0.0
    return True, tmin, tmax


@njit(cache=True, nogil=True)
def process_image(image, earth_position, pointing_direction, fov, voxel_grid, voxel_grid_extent,
                  max_distance, num_steps, celestial_sphere_texture, center_ra_rad, center_dec_rad,
                  angular_width_rad, angular_height_rad, update_celestial_sphere,
                  perform_background_subtraction):
    """Same contract as process_image_cpp (image/voxel_grid/texture are float64,
    voxel_grid_extent is a (3, 2) array of (min, max) per axis). Mutates
    voxel_grid and celestial_sphere_texture in place. Pass an empty (0, 0, 0)
    voxel_grid to only update the texture."""
    image_height, image_width = image.shape
    tex_h, tex_w = celestial_sphere_texture.shape
    have_grid = voxel_grid.size > 0
    gnx, gny, gnz = voxel_grid.shape
    x_min, x_max = voxel_grid_extent[0, 0], voxel_grid_extent[0, 1]
    y_min, y_max = voxel_grid_extent[1, 0], voxel_grid_extent[1, 1]
    z_min, z_max = voxel_grid_extent[2, 0], voxel_grid_extent[2, 1]
    bmin = np.array([x_min, y_min, z_min])
    bmax = np.array([x_max, y_max, z_max])

    focal_length = (image_width / 2.0) / math.tan(fov / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0

    zn = math.sqrt(pointing_direction[0] ** 2 + pointing_direction[1] ** 2 + pointing_direction[2] ** 2)
    z_axis = np.array([pointing_direction[0] / zn, pointing_direction[1] / zn, pointing_direction[2] / zn])
    up = np.array([0.0, 0.0, 1.0])
    if ((abs(z_axis[0] - up[0]) < 1e-8 and abs(z_axis[1] - up[1]) < 1e-8 and abs(z_axis[2] - up[2]) < 1e-8)
            or (abs(z_axis[0] + up[0]) < 1e-8 and abs(z_axis[1] + up[1]) < 1e-8 and abs(z_axis[2] + up[2]) < 1e-8)):
        up = np.array([0.0, 1.0, 0.0])
    x_axis = np.array([up[1] * z_axis[2] - up[2] * z_axis[1],
                       up[2] * z_axis[0] - up[0] * z_axis[2],
                       up[0] * z_axis[1] - up[1] * z_axis[0]])
    xn = math.sqrt(x_axis[0] ** 2 + x_axis[1] ** 2 + x_axis[2] ** 2)
    x_axis /= xn
    y_axis = np.array([z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
                       z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
                       z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0]])

    step_size = max_distance / num_steps
    dw = np.empty(3)

    for i in range(image_height):
        for j in range(image_width):
            brightness = image[i, j]
            if not (brightness > 0):
                continue
            x_cam = j - cx
            y_cam = i - cy
            z_cam = focal_length
            norm = math.sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam)
            dcx, dcy, dcz = x_cam / norm, y_cam / norm, z_cam / norm
            for k in range(3):
                dw[k] = x_axis[k] * dcx + y_axis[k] * dcy + z_axis[k] * dcz
            dn = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1] + dw[2] * dw[2])
            dw /= dn

            r = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1] + dw[2] * dw[2])
            dec = math.asin(dw[2] / r)
            ra = math.atan2(dw[1], dw[0])
            if ra < 0:
                ra += 2 * math.pi
            ra_off = ra - center_ra_rad
            dec_off = dec - center_dec_rad
            if ra_off > math.pi:
                ra_off -= 2 * math.pi
            if ra_off < -math.pi:
                ra_off += 2 * math.pi
            within = abs(ra_off) <= angular_width_rad / 2 and abs(dec_off) <= angular_height_rad / 2

            u_idx = int((ra_off + angular_width_rad / 2) / angular_width_rad * tex_w)
            v_idx = int((dec_off + angular_height_rad / 2) / angular_height_rad * tex_h)
            u_idx = min(max(u_idx, 0), tex_w - 1)
            v_idx = min(max(v_idx, 0), tex_h - 1)

            background = celestial_sphere_texture[v_idx, u_idx] if within else 0.0
            if perform_background_subtraction:
                brightness -= background
                if brightness <= 0:
                    continue
            if update_celestial_sphere and within:
                celestial_sphere_texture[v_idx, u_idx] += brightness

            if have_grid:
                hit, t_entry, t_exit = _ray_aabb(earth_position, dw, bmin, bmax)
                if hit:
                    t_entry = max(t_entry, 0.0)
                    t_exit = min(t_exit, max_distance)
                    if t_entry <= t_exit:
                        for s in range(int(t_entry / step_size), int(t_exit / step_size) + 1):
                            d = s * step_size
                            px = earth_position[0] + d * dw[0]
                            py = earth_position[1] + d * dw[1]
                            pz = earth_position[2] + d * dw[2]
                            if x_min <= px <= x_max and y_min <= py <= y_max and z_min <= pz <= z_max:
                                xi = min(max(int((px - x_min) / (x_max - x_min) * gnx), 0), gnx - 1)
                                yi = min(max(int((py - y_min) / (y_max - y_min) * gny), 0), gny - 1)
                                zi = min(max(int((pz - z_min) / (z_max - z_min) * gnz), 0), gnz - 1)
                                voxel_grid[xi, yi, zi] += brightness


# ----------------------------------------------------------------------------- driver


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="p2v-space", description=__doc__.split("\n\n")[0])
    ap.add_argument("--fits-dir", default="fits")
    ap.add_argument("--grid-size", type=int, nargs=3, default=(400, 400, 400))
    ap.add_argument("--grid-extent", type=float, default=3e12, help="half side length of the voxel cube (m)")
    ap.add_argument("--distance-au", type=float, default=41.714231, help="grid centre distance from the Sun (AU)")
    ap.add_argument("--center-ra", type=float, default=280.50)
    ap.add_argument("--center-dec", type=float, default=-20.0)
    ap.add_argument("--percentile", type=float, default=0.1, help="brightness threshold percentile")
    ap.add_argument("--num-steps", type=int, default=20000)
    ap.add_argument("--default-fov-arcmin", type=float, default=2.7)
    ap.add_argument("--patch", type=float, nargs=2, default=(5.0, 5.0), metavar=("W_DEG", "H_DEG"))
    ap.add_argument("--texture", type=int, nargs=2, default=(1024, 1024), metavar=("W", "H"))
    args = ap.parse_args(argv)

    try:
        import astropy.units as u
        import matplotlib
        import matplotlib.pyplot as plt
        from astropy.coordinates import SkyCoord, get_body_barycentric, solar_system_ephemeris
        from astropy.io import fits
        from astropy.time import Time
    except ImportError:
        print("p2v-space needs the 'space' extra:  uv sync --extra space", file=sys.stderr)
        return 1

    AU = 1.496e11
    distance_from_sun = AU * args.distance_au
    max_distance = distance_from_sun * 100
    tex_w, tex_h = args.texture

    def obs_time_of(header):
        date_obs, time_obs = header.get("DATE-OBS"), header.get("TIME-OBS")
        if date_obs is None or time_obs is None:
            raise ValueError("DATE-OBS and TIME-OBS not found in FITS header.")
        return Time(f"{date_obs}T{time_obs}", format="isot", scale="utc")

    def earth_position(t):
        with solar_system_ephemeris.set("builtin"):
            return np.array(get_body_barycentric("earth", t).get_xyz().to(u.meter).value, dtype=np.float64).ravel()

    def pointing(header):
        ra, dec = header.get("RA_TARG"), header.get("DEC_TARG")
        if ra is None or dec is None:
            raise ValueError("RA_TARG and DEC_TARG not found in FITS header.")
        c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs").represent_as("cartesian")
        v = np.array([c.x.value, c.y.value, c.z.value], dtype=np.float64)
        return v / np.linalg.norm(v)

    center_vec = SkyCoord(ra=args.center_ra * u.degree, dec=args.center_dec * u.degree, frame="icrs").cartesian.xyz.value
    grid_center = center_vec * distance_from_sun
    ext = np.array([[c - args.grid_extent, c + args.grid_extent] for c in grid_center], dtype=np.float64)
    voxel_grid = np.zeros(tuple(args.grid_size), dtype=np.float64)
    texture = np.zeros((tex_h, tex_w), dtype=np.float64)

    files = sorted(os.path.join(args.fits_dir, f) for f in os.listdir(args.fits_dir) if f.endswith(".fits")) \
        if os.path.isdir(args.fits_dir) else []
    if not files:
        print(f"No FITS files found in directory '{args.fits_dir}'.")
        return 1
    timed = []
    for f in files:
        try:
            with fits.open(f) as hdul:
                timed.append((f, obs_time_of(hdul[0].header)))
        except ValueError as e:
            print(e)
    timed.sort(key=lambda x: x[1])

    earth_positions, pointing_dirs, obs_times = [], [], []
    for path, _ in timed:
        with fits.open(path) as hdul:
            print(f"Processing FITS file: {path}")
            header = hdul[0].header
            t = obs_time_of(header)
            e_pos, p_dir = earth_position(t), pointing(header)
            data = hdul[0].data
            if data is None:
                if "SCI" in hdul:
                    data = hdul["SCI"].data
                else:
                    data = next((h.data for h in hdul if isinstance(h, (fits.ImageHDU, fits.CompImageHDU))
                                 and h.data is not None), None)
            if data is None or data.ndim != 2:
                raise ValueError(f"{path}: no 2D image data found")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            lo, hi = data.min(), data.max()
            if hi - lo == 0:
                raise ValueError(f"{path}: image data has zero dynamic range")
            image = ((data - lo) / (hi - lo)).astype(np.float64)
            h, w = image.shape

            fov = header.get("FOV")
            if fov is None:
                cd = [header.get(k) for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")]
                if all(v is not None for v in cd):
                    fov = max(math.hypot(cd[0], cd[2]) * w, math.hypot(cd[1], cd[3]) * h)
                else:
                    fov = args.default_fov_arcmin / 60.0
            process_image(image, e_pos, p_dir, math.radians(float(fov)), voxel_grid, ext, max_distance,
                          args.num_steps, texture, math.radians(args.center_ra), math.radians(args.center_dec),
                          math.radians(args.patch[0]), math.radians(args.patch[1]), True, False)
        earth_positions.append(e_pos)
        pointing_dirs.append(p_dir)
        obs_times.append(t)

    n_files = len(timed)
    avg = voxel_grid / n_files
    threshold = np.percentile(avg[avg > 0], args.percentile) if np.any(avg > 0) else 0
    mask = avg > threshold
    xi, yi, zi = np.nonzero(mask)
    nx, ny, nz = avg.shape
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = ext
    xs = xi / nx * (x_max - x_min) + x_min
    ys = yi / ny * (y_max - y_min) + y_min
    zs = zi / nz * (z_max - z_min) + z_min
    intensities = avg[mask]

    if intensities.size:
        b = int(np.argmax(intensities))
        bx, by, bz = xs[b], ys[b], zs[b]
        sph = SkyCoord(x=bx * u.meter, y=by * u.meter, z=bz * u.meter,
                       representation_type="cartesian", frame="icrs").represent_as("spherical")
        print("Brightest Point Coordinates:")
        print(f"RA: {sph.lon.degree:.6f} degrees")
        print(f"Dec: {sph.lat.degree:.6f} degrees")
        print(f"Distance from Origin: {math.sqrt(bx**2 + by**2 + bz**2) / AU:.6f} AU")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(xs, ys, zs, c=intensities, cmap="hot", marker="o", s=5, alpha=0.5)
        ax.scatter([bx], [by], [bz], c="blue", marker="*", s=100, label="Brightest Point")
        ep = np.array(earth_positions)
        ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2], c="green", marker="o", s=200, label="Earth Positions")
        ax.scatter(*([v] for v in grid_center), c="purple", marker="x", s=100, label="Voxel Grid Center")
        times = np.array([t.mjd for t in obs_times])
        tnorm = (times - times.min()) / (times.max() - times.min()) if np.ptp(times) > 0 else np.zeros_like(times)
        cmap = matplotlib.colormaps["viridis"]  # plt.cm.get_cmap was removed in matplotlib 3.9
        L = args.grid_extent * 0.5
        for pos, dvec, tn in zip(earth_positions, pointing_dirs, tnorm):
            ax.quiver(*pos, *(dvec * L), color=cmap(tn), length=1.0, normalize=False, arrow_length_ratio=0.1)
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(times)
        plt.colorbar(mappable, ax=ax, label="Observation Time (MJD)")
        ax.legend()
        plt.colorbar(sc, ax=ax, label="Average Brightness")
        ax.set_title("3D Visualization of Detected Objects, Earth Positions, and Camera Directions")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        plt.show()
    else:
        print("No significant voxels found.")

    zc = nz // 2
    plt.figure(figsize=(8, 6))
    plt.imshow(avg[:, :, zc].T, origin="lower", cmap="hot", extent=(x_min, x_max, y_min, y_max))
    plt.colorbar(label="Average Brightness")
    plt.title(f"Voxel Grid Slice at z = {z_min + zc * (z_max - z_min) / nz:.2f} meters")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
