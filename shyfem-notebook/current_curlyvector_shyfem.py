#!/usr/bin/env python3
"""
SHYFEM Surface Current Visualization with Curvilinear Streamlines
Antsiranana Bay, Madagascar

Creates animated GIF showing surface currents with:
- Fixed color scale (0 to max_speed m/s constant across all frames)
- Curvilinear streamlines integrated using scipy.integrate.solve_ivp
- OSM basemap via cartopy
- Streamlines clipped to ocean mesh boundary
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import solve_ivp
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
from matplotlib.colors import Normalize
import imageio


def load_shyfem_data(nc_file):
    """Load SHYFEM output netCDF file."""
    ds = xr.open_dataset(nc_file)
    return ds


def compute_streamlines_at_time(ds, time_idx, seed_points, trifinder, 
                                 lon, lat, spacing=0.016):
    """
    Compute curvilinear streamlines at given time step.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        SHYFEM data
    time_idx : int
        Time step index
    seed_points : ndarray
        (N, 2) array of seed points [lon, lat]
    trifinder : function
        Function to find triangle containing a point
    lon, lat : ndarray
        Node coordinates
    
    Returns:
    --------
    speed : ndarray
        Current speed at each node
    streamlines : list of tuples
        List of (sx, sy) streamline coordinates
    """
    u = ds['u_velocity'].isel(time=time_idx, level=0).values
    v = ds['v_velocity'].isel(time=time_idx, level=0).values
    speed = np.sqrt(u**2 + v**2)
    
    # Create interpolators
    interp_u = LinearNDInterpolator(list(zip(lon, lat)), u, fill_value=np.nan)
    interp_v = LinearNDInterpolator(list(zip(lon, lat)), v, fill_value=np.nan)
    
    def velocity_field(t, pos):
        x, y = pos
        u_val = interp_u(x, y)
        v_val = interp_v(x, y)
        if np.isnan(u_val) or np.isnan(v_val):
            return [0, 0]
        return [u_val, v_val]
    
    def point_in_mesh(x, y):
        tri_index = trifinder(x, y)
        return tri_index != -1
    
    streamlines = []
    for seed in seed_points:
        x0, y0 = seed
        
        if not point_in_mesh(x0, y0):
            continue
        
        speed_here = np.sqrt(interp_u(x0, y0)**2 + interp_v(x0, y0)**2)
        if speed_here < 0.01 or np.isnan(speed_here):
            continue
        
        # Integration time proportional to inverse speed
        max_time = 0.03 / (speed_here + 0.1)
        sol = solve_ivp(velocity_field, [0, max_time], [x0, y0], 
                        max_step=0.001, method='RK45')
        
        if len(sol.y[0]) > 3:
            sx_full, sy_full = sol.y[0], sol.y[1]
            
            # Clip to mesh boundary
            sx_clipped, sy_clipped = [], []
            for i in range(len(sx_full)):
                if point_in_mesh(sx_full[i], sy_full[i]):
                    sx_clipped.append(sx_full[i])
                    sy_clipped.append(sy_full[i])
                else:
                    break
            
            if len(sx_clipped) > 3:
                # Length proportional to speed
                length_factor = min(speed_here / 0.5, 1.5)
                n_points = int(len(sx_clipped) * length_factor)
                n_points = min(n_points, len(sx_clipped))
                streamlines.append((sx_clipped[:n_points], sy_clipped[:n_points]))
    
    return speed, streamlines


def create_animation(ds, output_file, time_step=12, fps=5):
    """
    Create animated GIF of surface currents.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        SHYFEM data
    output_file : str
        Output GIF filename
    time_step : int
        Sample every N time steps
    fps : int
        Frames per second
    """
    # Calculate global min/max for FIXED color scale
    print("Calculating global speed range...")
    all_u = ds['u_velocity'].isel(level=0).values
    all_v = ds['v_velocity'].isel(level=0).values
    all_speed = np.sqrt(all_u**2 + all_v**2)
    vmin = 0.0
    vmax = float(np.nanmax(all_speed))
    print(f"Fixed range: {vmin:.3f} to {vmax:.3f} m/s")
    
    # Create FIXED normalizer - this ensures constant color scale
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.turbo
    
    # Get mesh coordinates
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    triangles = ds['element_index'].values - 1
    triang = tri.Triangulation(lon, lat, triangles)
    trifinder = triang.get_trifinder()
    
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    
    # Create uniform seed points
    spacing = 0.016  # ~1.6 km
    seed_lon = np.arange(lon_min, lon_max, spacing)
    seed_lat = np.arange(lat_min, lat_max, spacing)
    seed_lon_mesh, seed_lat_mesh = np.meshgrid(seed_lon, seed_lat)
    seed_points = np.column_stack([seed_lon_mesh.ravel(), seed_lat_mesh.ravel()])
    
    # Time indices to animate
    time_indices = list(range(0, len(ds.time), time_step))
    print(f"Creating animation with {len(time_indices)} frames")
    
    # Generate frames
    frames = []
    for frame_num, time_idx in enumerate(time_indices):
        print(f"Frame {frame_num+1}/{len(time_indices)}: t={time_idx}")
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Add OSM basemap
        imagery = OSM()
        ax.add_image(imagery, 10)
        
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Compute streamlines
        speed, streamlines = compute_streamlines_at_time(
            ds, time_idx, seed_points, trifinder, lon, lat, spacing)
        
        # Create tripcolor with FIXED norm
        tcf = ax.tripcolor(triang, speed, cmap=cmap, shading='flat',
                            norm=norm, transform=ccrs.PlateCarree())
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Current Speed (m/s)', shrink=0.7)
        
        # Plot streamlines
        for sx, sy in streamlines:
            ax.plot(sx, sy, 'w-', linewidth=1.5, alpha=0.95, 
                   transform=ccrs.PlateCarree())
            
            if len(sx) > 1:
                dx = sx[-1] - sx[-2]
                dy = sy[-1] - sy[-2]
                ax.arrow(sx[-2], sy[-2], dx, dy, 
                        head_width=0.0025, head_length=0.0035, 
                        fc='white', ec='white', alpha=0.95, linewidth=0,
                        transform=ccrs.PlateCarree())
        
        # Title
        time_str = str(ds.time.isel(time=time_idx).values)[:19]
        ax.set_title(f'SHYFEM Surface Currents - Antsiranana Bay\n{time_str}', 
                    fontsize=12)
        
        # Capture frame
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        frames.append(frame)
        plt.close(fig)
    
    # Save GIF
    print(f"Saving GIF to {output_file}...")
    imageio.mimsave(output_file, frames, fps=fps, loop=0)
    print("Done!")


if __name__ == '__main__':
    # Load data
    ds = load_shyfem_data('surf.ous.nc')
    
    # Create animation
    create_animation(ds, 'current_curlyvector_shyfem.gif', time_step=12, fps=5)
