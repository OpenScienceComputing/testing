"""
Virtual ADCP Data Creation
Creates a synthetic ADCP dataset from SHYFEM model output with realistic noise
"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d

# Load SHYFEM model output
print("Loading SHYFEM model data...")
ds = xr.open_dataset('surf.ous.nc')

# ADCP location (center of domain)
adcp_lon = 49.4005
adcp_lat = -12.1858

print(f"\n=== Virtual ADCP Configuration ===")
print(f"Location: {adcp_lon}°E, {adcp_lat}°S")

# Find nearest SHYFEM node
lon = ds['longitude'].values
lat = ds['latitude'].values
dist = np.sqrt((lon - adcp_lon)**2 + (lat - adcp_lat)**2)
nearest_node = np.argmin(dist)
nearest_dist = dist[nearest_node] * 111  # Convert to km

print(f"Nearest SHYFEM node: {nearest_node}")
print(f"Distance: {nearest_dist:.2f} km")

# Extract model data at ADCP location
u_model = ds['u_velocity'].isel(node=nearest_node).values  # (time, level)
v_model = ds['v_velocity'].isel(node=nearest_node).values
depths_model = ds['level'].values
time_model = ds.time.values

print(f"\nSHYFEM data shape: {u_model.shape}")
print(f"Time steps: {len(time_model)}")
print(f"Vertical levels: {len(depths_model)}")

# ADCP specifications
adcp_depths = np.arange(0, 41, 1)  # 0-40m, 1m bins
time_adcp = pd.date_range(start=time_model[0], end=time_model[-1], freq='1H')

print(f"\nADCP specifications:")
print(f"Depth bins: {len(adcp_depths)} (0-40m, 1m intervals)")
print(f"Temporal sampling: Hourly")
print(f"Total hours: {len(time_adcp)}")

# Step 1: Interpolate vertically (SHYFEM depths -> ADCP depths)
print("\n=== Step 1: Vertical Interpolation ===")
u_on_adcp_depths = np.zeros((len(time_model), len(adcp_depths)))
v_on_adcp_depths = np.zeros((len(time_model), len(adcp_depths)))

for t in range(len(time_model)):
    u_on_adcp_depths[t, :] = np.interp(adcp_depths, depths_model, u_model[t, :])
    v_on_adcp_depths[t, :] = np.interp(adcp_depths, depths_model, v_model[t, :])

print(f"Interpolated to ADCP depths: {u_on_adcp_depths.shape}")

# Step 2: Interpolate temporally (SHYFEM 6-hourly -> ADCP hourly)
print("\n=== Step 2: Temporal Interpolation ===")
time_model_sec = (pd.DatetimeIndex(time_model) - pd.Timestamp(time_model[0])).total_seconds().values
time_adcp_sec = (pd.DatetimeIndex(time_adcp) - pd.Timestamp(time_model[0])).total_seconds().values

u_adcp = np.zeros((len(time_adcp), len(adcp_depths)))
v_adcp = np.zeros((len(time_adcp), len(adcp_depths)))

for d in range(len(adcp_depths)):
    u_adcp[:, d] = np.interp(time_adcp_sec, time_model_sec, u_on_adcp_depths[:, d])
    v_adcp[:, d] = np.interp(time_adcp_sec, time_model_sec, v_on_adcp_depths[:, d])

print(f"Interpolated to hourly: {u_adcp.shape}")

# Step 3: Add realistic ADCP measurement noise
print("\n=== Step 3: Adding Realistic Noise ===")

# Noise parameters (typical for 1200 kHz ADCP)
velocity_noise_std = 0.005  # ±0.5 cm/s random noise
velocity_bias = 0.01        # 1 cm/s systematic bias
depth_error_coeff = 0.002   # 0.2 cm/s per meter depth
spike_probability = 0.05    # 5% of measurements have spikes
spike_magnitude = 0.05      # ±5 cm/s spikes

# Create temporal correlation (12-hour timescale)
n_times = len(time_adcp)
ar1_coeff = np.exp(-1.0 / 12)  # AR(1) process with 12-hour correlation
temporal_drift = np.zeros(n_times)
temporal_drift[0] = np.random.randn() * 0.01
for t in range(1, n_times):
    temporal_drift[t] = ar1_coeff * temporal_drift[t-1] + np.random.randn() * 0.01 * np.sqrt(1 - ar1_coeff**2)

# Add noise
u_noisy = np.copy(u_adcp)
v_noisy = np.copy(v_adcp)

for d in range(len(adcp_depths)):
    # Random noise
    u_noisy[:, d] += np.random.randn(n_times) * velocity_noise_std
    v_noisy[:, d] += np.random.randn(n_times) * velocity_noise_std
    
    # Systematic bias
    u_noisy[:, d] += velocity_bias
    v_noisy[:, d] += velocity_bias
    
    # Depth-dependent error
    depth_error = adcp_depths[d] * depth_error_coeff
    u_noisy[:, d] += np.random.randn(n_times) * depth_error
    v_noisy[:, d] += np.random.randn(n_times) * depth_error
    
    # Temporal drift
    u_noisy[:, d] += temporal_drift
    v_noisy[:, d] += temporal_drift
    
    # Random spikes
    spike_mask = np.random.rand(n_times) < spike_probability
    u_noisy[spike_mask, d] += np.random.randn(np.sum(spike_mask)) * spike_magnitude
    v_noisy[spike_mask, d] += np.random.randn(np.sum(spike_mask)) * spike_magnitude

speed_noisy = np.sqrt(u_noisy**2 + v_noisy**2)
direction_noisy = np.arctan2(v_noisy, u_noisy) * 180 / np.pi

# Statistics
u_diff = u_noisy - u_adcp
v_diff = v_noisy - v_adcp
print(f"Noise statistics:")
print(f"  U difference: {np.mean(u_diff):.4f} ± {np.std(u_diff):.4f} m/s")
print(f"  V difference: {np.mean(v_diff):.4f} ± {np.std(v_diff):.4f} m/s")

# Create xarray Dataset
print("\n=== Creating ADCP Dataset ===")
adcp_ds = xr.Dataset(
    {
        'u': (['time', 'depth'], u_noisy, {
            'long_name': 'Eastward current velocity',
            'standard_name': 'eastward_sea_water_velocity',
            'units': 'm/s',
            'comment': 'Virtual ADCP data with realistic noise'
        }),
        'v': (['time', 'depth'], v_noisy, {
            'long_name': 'Northward current velocity',
            'standard_name': 'northward_sea_water_velocity',
            'units': 'm/s',
            'comment': 'Virtual ADCP data with realistic noise'
        }),
        'speed': (['time', 'depth'], speed_noisy, {
            'long_name': 'Current speed',
            'units': 'm/s',
            'comment': 'Derived from u and v components'
        }),
        'direction': (['time', 'depth'], direction_noisy, {
            'long_name': 'Current direction',
            'units': 'degrees',
            'comment': 'Oceanographic convention (direction toward)'
        })
    },
    coords={
        'time': time_adcp,
        'depth': adcp_depths,
        'longitude': adcp_lon,
        'latitude': adcp_lat
    },
    attrs={
        'title': 'Virtual ADCP Observations',
        'institution': 'Synthetic dataset from SHYFEM model',
        'source': 'SHYFEM ocean model + realistic ADCP noise',
        'location': f'{adcp_lon}°E, {adcp_lat}°S',
        'instrument_type': 'Simulated 1200 kHz ADCP',
        'depth_cell_size': '1 meter',
        'sampling_interval': '1 hour',
        'velocity_noise_std': f'{velocity_noise_std} m/s',
        'velocity_bias': f'{velocity_bias} m/s',
        'depth_error_coefficient': f'{depth_error_coeff} m/s per meter',
        'spike_probability': f'{spike_probability}',
        'spike_magnitude': f'{spike_magnitude} m/s',
        'temporal_correlation': '12 hours (AR1 process)',
        'creation_date': pd.Timestamp.now().isoformat()
    }
)

# Save to NetCDF
output_file = 'virtual_adcp_noisy.nc'
adcp_ds.to_netcdf(output_file)
print(f"\n✓ Saved: {output_file}")

# Also export to CSV (long format)
csv_data = []
for t_idx, time_val in enumerate(time_adcp):
    for d_idx, depth_val in enumerate(adcp_depths):
        csv_data.append({
            'time': time_val,
            'depth': depth_val,
            'u': u_noisy[t_idx, d_idx],
            'v': v_noisy[t_idx, d_idx],
            'speed': speed_noisy[t_idx, d_idx],
            'direction': direction_noisy[t_idx, d_idx]
        })

df = pd.DataFrame(csv_data)
df.to_csv('virtual_adcp.csv', index=False)
print(f"✓ Saved: virtual_adcp.csv")

print("\n=== Summary ===")
print(f"Dataset dimensions: {len(time_adcp)} times × {len(adcp_depths)} depths")
print(f"Speed range: {speed_noisy.min():.3f} - {speed_noisy.max():.3f} m/s")
print(f"Surface mean speed: {speed_noisy[:, 0].mean():.3f} m/s")
print(f"File size: {output_file}")
