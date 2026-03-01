#!/usr/bin/env python3
"""
SHYFEM Web Application - Interactive ocean current visualization
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import xarray as xr
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import LinearNDInterpolator
import atexit

app = Flask(__name__)
CORS(app)

# Global data storage
data_store = {}

def cleanup():
    print("Shutting down SHYFEM web app...")

atexit.register(cleanup)

def load_data():
    """Load SHYFEM data and pre-compute particles"""
    print("Loading SHYFEM data...")
    ds = xr.open_dataset('/home/ubuntu/.openclaw/workspace/shyfem-notebook/erddap_data/surf.ous.nc')

    lon = ds['longitude'].values
    lat = ds['latitude'].values
    triangles = ds['element_index'].values - 1
    triang = tri.Triangulation(lon, lat, triangles)
    trifinder = triang.get_trifinder()

    u_all = ds['u_velocity'].isel(level=0).values
    v_all = ds['v_velocity'].isel(level=0).values
    time_vals = ds['time'].values

    # Get global speed range
    all_speed = np.sqrt(u_all**2 + v_all**2)
    vmin = 0.0
    vmax = float(np.nanmax(all_speed))

    print(f"Data loaded: {len(time_vals)} time steps, {len(lon)} nodes, {vmin}-{vmax} m/s")

    # Pre-compute particle positions
    print("Computing particle trajectories...")
    np.random.seed(42)
    n_particles = 150

    # Create particles within mesh
    particle_lons = []
    particle_lats = []
    for _ in range(n_particles * 3):
        test_lon = np.random.uniform(lon.min() + 0.01, lon.max() - 0.01)
        test_lat = np.random.uniform(lat.min() + 0.01, lat.max() - 0.01)
        if trifinder(test_lon, test_lat) != -1:
            particle_lons.append(test_lon)
            particle_lats.append(test_lat)
            if len(particle_lons) >= n_particles:
                break

    particle_lons = np.array(particle_lons)
    particle_lats = np.array(particle_lats)

    # Pre-compute particle positions for all time steps
    particle_history = []
    current_lons = particle_lons.copy()
    current_lats = particle_lats.copy()

    for t_idx in range(len(time_vals)):
        u = u_all[t_idx]
        v = v_all[t_idx]
        interp_u = LinearNDInterpolator(list(zip(lon, lat)), u, fill_value=0)
        interp_v = LinearNDInterpolator(list(zip(lon, lat)), v, fill_value=0)
        
        u_vals = interp_u(current_lons, current_lats)
        v_vals = interp_v(current_lons, current_lats)
        
        dt = 3600
        deg_per_m = 1 / 111000
        current_lons = current_lons + u_vals * dt * deg_per_m
        current_lats = current_lats + v_vals * dt * deg_per_m
        
        # Reset particles that leave mesh
        for i in range(len(current_lons)):
            if trifinder(current_lons[i], current_lats[i]) == -1:
                current_lons[i] = np.random.uniform(lon.min() + 0.01, lon.max() - 0.01)
                current_lats[i] = np.random.uniform(lat.min() + 0.01, lat.max() - 0.01)
        
        particle_history.append(np.column_stack([current_lons, current_lats]))

    print("Particle trajectories computed")

    data_store['lon'] = lon
    data_store['lat'] = lat
    data_store['triang'] = triang
    data_store['u_all'] = u_all
    data_store['v_all'] = v_all
    data_store['time'] = time_vals
    data_store['vmin'] = vmin
    data_store['vmax'] = vmax
    data_store['particle_history'] = particle_history

# Load data at startup
load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/time')
def get_time():
    times = [str(t)[:19] for t in data_store['time']]
    return jsonify({'times': times})

@app.route('/api/speed/<int:frame>')
def get_speed(frame):
    u = data_store['u_all'][frame]
    v = data_store['v_all'][frame]
    speed = np.sqrt(u**2 + v**2)
    return jsonify({
        'speed': speed.tolist(),
        'vmin': data_store['vmin'],
        'vmax': data_store['vmax']
    })

@app.route('/api/image/<int:frame>')
def get_image(frame):
    u = data_store['u_all'][frame]
    v = data_store['v_all'][frame]
    speed = np.sqrt(u**2 + v**2)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.set_aspect('equal')
    
    # Use FIXED colormap and normalizer - same for ALL frames
    cmap = plt.cm.turbo
    cmap.set_over('red')
    cmap.set_under('blue')
    vmin = data_store['vmin']
    vmax = data_store['vmax']
    
    # Create a fixed normalizer
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    
    # Use pcolormesh with triangulation for more consistent colors
    triang = data_store['triang']
    
    # Compute face colors explicitly for consistency
    facecolors = cmap(norm(speed))
    
    ax.triplot(triang, color='none', alpha=0)  # Setup
    ax.tripcolor(triang, speed, cmap=cmap, shading='flat', 
                 norm=norm, clim=[vmin, vmax])
    
    ax.set_xlim([data_store['lon'].min(), data_store['lon'].max()])
    ax.set_ylim([data_store['lat'].min(), data_store['lat'].max()])
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.route('/api/particles/<int:frame>')
def get_particles(frame):
    particles = data_store['particle_history'][frame]
    return jsonify({
        'particles': particles.tolist()
    })

@app.route('/api/wms')
def wms():
    """Simple WMS endpoint"""
    time_idx = int(request.args.get('time', 0))
    return get_image(time_idx)

if __name__ == '__main__':
    print("Starting SHYFEM web server on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)