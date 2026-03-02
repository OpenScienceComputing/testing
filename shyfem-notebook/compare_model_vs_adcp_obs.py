"""
Model vs ADCP Observations Comparison
Compares SHYFEM model output with virtual ADCP observations
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

print("=== SHYFEM Model vs ADCP Comparison ===\n")

# Load ADCP observations
print("Loading ADCP observations...")
adcp = xr.open_dataset('virtual_adcp_noisy.nc')

adcp_lon = float(adcp.longitude.values)
adcp_lat = float(adcp.latitude.values)
print(f"ADCP location: {adcp_lon}°E, {adcp_lat}°S")

# Load SHYFEM model
print("Loading SHYFEM model...")
shyfem = xr.open_dataset('surf.ous.nc')

# Find nearest SHYFEM node to ADCP
lon = shyfem['longitude'].values
lat = shyfem['latitude'].values
dist = np.sqrt((lon - adcp_lon)**2 + (lat - adcp_lat)**2)
nearest_node = np.argmin(dist)
nearest_dist = dist[nearest_node] * 111

print(f"Nearest SHYFEM node: {nearest_node}")
print(f"Distance: {nearest_dist:.2f} km")

# Extract SHYFEM data at ADCP location
print("\n=== Extracting SHYFEM Data ===")
u_shyfem = shyfem['u_velocity'].isel(node=nearest_node).values
v_shyfem = shyfem['v_velocity'].isel(node=nearest_node).values
depths_shyfem = shyfem['level'].values
time_shyfem = shyfem.time.values

print(f"SHYFEM shape: {u_shyfem.shape}")
print(f"Time steps: {len(time_shyfem)}")
print(f"Vertical levels: {len(depths_shyfem)}")

# ADCP grid
adcp_depths = adcp.depth.values
time_adcp = adcp.time.values

print(f"\nADCP shape: ({len(time_adcp)}, {len(adcp_depths)})")

# Step 1: Interpolate SHYFEM vertically to ADCP depths
print("\n=== Step 1: Vertical Interpolation ===")
u_shyfem_on_adcp_depths = np.zeros((len(time_shyfem), len(adcp_depths)))
v_shyfem_on_adcp_depths = np.zeros((len(time_shyfem), len(adcp_depths)))

for t in range(len(time_shyfem)):
    u_shyfem_on_adcp_depths[t, :] = np.interp(adcp_depths, depths_shyfem, u_shyfem[t, :])
    v_shyfem_on_adcp_depths[t, :] = np.interp(adcp_depths, depths_shyfem, v_shyfem[t, :])

print(f"Interpolated to ADCP depths: {u_shyfem_on_adcp_depths.shape}")

# Step 2: Interpolate SHYFEM temporally to ADCP times
print("\n=== Step 2: Temporal Interpolation ===")
time_shyfem_sec = (pd.DatetimeIndex(time_shyfem) - pd.Timestamp(time_shyfem[0])).total_seconds().values
time_adcp_sec = (pd.DatetimeIndex(time_adcp) - pd.Timestamp(time_shyfem[0])).total_seconds().values

u_model = np.zeros((len(time_adcp), len(adcp_depths)))
v_model = np.zeros((len(time_adcp), len(adcp_depths)))

for d in range(len(adcp_depths)):
    u_model[:, d] = np.interp(time_adcp_sec, time_shyfem_sec, u_shyfem_on_adcp_depths[:, d])
    v_model[:, d] = np.interp(time_adcp_sec, time_shyfem_sec, v_shyfem_on_adcp_depths[:, d])

speed_model = np.sqrt(u_model**2 + v_model**2)

print(f"Interpolated to hourly: {u_model.shape}")

# Get ADCP observations
u_obs = adcp.u.values
v_obs = adcp.v.values
speed_obs = adcp.speed.values

# Calculate errors
print("\n=== Calculating Errors ===")
u_error = u_model - u_obs
v_error = v_model - v_obs
speed_error = speed_model - speed_obs

# Overall statistics
valid = ~np.isnan(speed_obs) & ~np.isnan(speed_model)
speed_rmse = np.sqrt(np.mean(speed_error[valid]**2))
speed_bias = np.mean(speed_error[valid])
speed_corr = np.corrcoef(speed_obs[valid], speed_model[valid])[0, 1]

print(f"\nOverall Statistics:")
print(f"  Speed RMSE: {speed_rmse:.4f} m/s")
print(f"  Speed Bias: {speed_bias:.4f} m/s")
print(f"  Speed R²: {speed_corr**2:.3f}")

# Statistics by depth
print(f"\nStatistics by Depth:")
for depth_idx in [0, 5, 10, 20]:
    if depth_idx < len(adcp_depths):
        valid_d = ~np.isnan(speed_obs[:, depth_idx]) & ~np.isnan(speed_model[:, depth_idx])
        if valid_d.sum() > 0:
            rmse_d = np.sqrt(np.mean(speed_error[:, depth_idx][valid_d]**2))
            corr_d = np.corrcoef(speed_obs[:, depth_idx][valid_d], speed_model[:, depth_idx][valid_d])[0, 1]
            print(f"  {adcp_depths[depth_idx]}m: RMSE={rmse_d:.4f} m/s, R²={corr_d**2:.3f}")

# Create comprehensive comparison figure
print("\n=== Creating Comparison Figure ===")

fig = plt.figure(figsize=(18, 12))

# 1. SHYFEM Speed Hovmöller
ax1 = plt.subplot(3, 3, 1)
cf1 = ax1.contourf(time_adcp, adcp_depths, speed_model.T, levels=20, cmap='viridis')
ax1.set_ylabel('Depth (m)', fontsize=11)
ax1.set_title('SHYFEM Model Speed (m/s)', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.colorbar(cf1, ax=ax1, label='Speed (m/s)')
ax1.grid(alpha=0.3)

# 2. ADCP Speed Hovmöller
ax2 = plt.subplot(3, 3, 2)
cf2 = ax2.contourf(time_adcp, adcp_depths, speed_obs.T, levels=20, cmap='viridis')
ax2.set_ylabel('Depth (m)', fontsize=11)
ax2.set_title('ADCP Observed Speed (m/s)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.colorbar(cf2, ax=ax2, label='Speed (m/s)')
ax2.grid(alpha=0.3)

# 3. Speed Error Hovmöller
ax3 = plt.subplot(3, 3, 3)
vmax_err = max(abs(speed_error.min()), abs(speed_error.max()))
error_norm = TwoSlopeNorm(vmin=-vmax_err, vcenter=0, vmax=vmax_err)
cf3 = ax3.contourf(time_adcp, adcp_depths, speed_error.T, levels=20, cmap='RdBu_r', norm=error_norm)
ax3.contour(time_adcp, adcp_depths, speed_error.T, levels=[0], colors='black', linewidths=2)
ax3.set_ylabel('Depth (m)', fontsize=11)
ax3.set_title('Error (Model - ADCP) [m/s]', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.colorbar(cf3, ax=ax3, label='Error (m/s)')
ax3.grid(alpha=0.3)

# 4. U Error Hovmöller
ax4 = plt.subplot(3, 3, 4)
vmax_u = max(abs(u_error.min()), abs(u_error.max()))
u_norm = TwoSlopeNorm(vmin=-vmax_u, vcenter=0, vmax=vmax_u)
cf4 = ax4.contourf(time_adcp, adcp_depths, u_error.T, levels=20, cmap='RdBu_r', norm=u_norm)
ax4.contour(time_adcp, adcp_depths, u_error.T, levels=[0], colors='black', linewidths=1.5)
ax4.set_ylabel('Depth (m)', fontsize=11)
ax4.set_title('U Error (Model - ADCP) [m/s]', fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.colorbar(cf4, ax=ax4, label='Error (m/s)')
ax4.grid(alpha=0.3)

# 5. V Error Hovmöller
ax5 = plt.subplot(3, 3, 5)
vmax_v = max(abs(v_error.min()), abs(v_error.max()))
v_norm = TwoSlopeNorm(vmin=-vmax_v, vcenter=0, vmax=vmax_v)
cf5 = ax5.contourf(time_adcp, adcp_depths, v_error.T, levels=20, cmap='RdBu_r', norm=v_norm)
ax5.contour(time_adcp, adcp_depths, v_error.T, levels=[0], colors='black', linewidths=1.5)
ax5.set_ylabel('Depth (m)', fontsize=11)
ax5.set_title('V Error (Model - ADCP) [m/s]', fontsize=12, fontweight='bold')
ax5.invert_yaxis()
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.colorbar(cf5, ax=ax5, label='Error (m/s)')
ax5.grid(alpha=0.3)

# 6. Vertical error profile
ax6 = plt.subplot(3, 3, 6)
error_mean = np.mean(speed_error, axis=0)
error_std = np.std(speed_error, axis=0)
ax6.plot(error_mean, adcp_depths, 'b-', linewidth=2.5, marker='o', markersize=5, label='Mean')
ax6.fill_betweenx(adcp_depths, error_mean - error_std, error_mean + error_std, 
                   alpha=0.3, color='blue', label='±1 std')
ax6.axvline(0, color='k', linestyle='--', linewidth=1.5)
ax6.set_xlabel('Speed Error (m/s)', fontsize=11)
ax6.set_ylabel('Depth (m)', fontsize=11)
ax6.set_title('Vertical Error Profile', fontsize=12, fontweight='bold')
ax6.invert_yaxis()
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)

# 7. Surface time series
ax7 = plt.subplot(3, 3, 7)
ax7.plot(time_adcp, speed_obs[:, 0], 'b-', linewidth=2.5, label='ADCP', alpha=0.8)
ax7.plot(time_adcp, speed_model[:, 0], 'r--', linewidth=2, label='SHYFEM', alpha=0.8)
ax7.set_xlabel('Date (2021)', fontsize=11)
ax7.set_ylabel('Speed (m/s)', fontsize=11)
ax7.set_title('Surface (0m) Time Series', fontsize=12, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(alpha=0.3)
ax7.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

valid_0 = ~np.isnan(speed_obs[:, 0]) & ~np.isnan(speed_model[:, 0])
rmse_0 = np.sqrt(np.mean(speed_error[:, 0][valid_0]**2))
corr_0 = np.corrcoef(speed_obs[:, 0][valid_0], speed_model[:, 0][valid_0])[0, 1]
ax7.text(0.02, 0.98, f'RMSE={rmse_0:.4f} m/s\nR²={corr_0**2:.3f}',
         transform=ax7.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 8. 10m time series
ax8 = plt.subplot(3, 3, 8)
ax8.plot(time_adcp, speed_obs[:, 10], 'b-', linewidth=2.5, label='ADCP', alpha=0.8)
ax8.plot(time_adcp, speed_model[:, 10], 'r--', linewidth=2, label='SHYFEM', alpha=0.8)
ax8.set_xlabel('Date (2021)', fontsize=11)
ax8.set_ylabel('Speed (m/s)', fontsize=11)
ax8.set_title('10m Depth Time Series', fontsize=12, fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(alpha=0.3)
ax8.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

valid_10 = ~np.isnan(speed_obs[:, 10]) & ~np.isnan(speed_model[:, 10])
rmse_10 = np.sqrt(np.mean(speed_error[:, 10][valid_10]**2))
corr_10 = np.corrcoef(speed_obs[:, 10][valid_10], speed_model[:, 10][valid_10])[0, 1]
ax8.text(0.02, 0.98, f'RMSE={rmse_10:.4f} m/s\nR²={corr_10**2:.3f}',
         transform=ax8.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 9. Scatter plot
ax9 = plt.subplot(3, 3, 9)
valid = ~np.isnan(speed_obs) & ~np.isnan(speed_model)
scatter = ax9.scatter(speed_obs[valid], speed_model[valid], 
                      alpha=0.3, s=10, c=np.tile(adcp_depths, len(time_adcp))[valid], 
                      cmap='viridis', edgecolors='none')
max_val = max(speed_obs[valid].max(), speed_model[valid].max())
ax9.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1')
ax9.set_xlabel('ADCP Speed (m/s)', fontsize=11)
ax9.set_ylabel('SHYFEM Speed (m/s)', fontsize=11)
ax9.set_title('Model vs Observations', fontsize=12, fontweight='bold')
ax9.legend(fontsize=10)
ax9.grid(alpha=0.3)
ax9.set_aspect('equal')
plt.colorbar(scatter, ax=ax9, label='Depth (m)')

corr_all = np.corrcoef(speed_obs[valid], speed_model[valid])[0, 1]
rmse_all = np.sqrt(np.mean((speed_model[valid] - speed_obs[valid])**2))
ax9.text(0.05, 0.95, f'R²={corr_all**2:.3f}\nRMSE={rmse_all:.4f} m/s',
         transform=ax9.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(f'SHYFEM Model vs ADCP Observations\nLocation: {adcp_lon:.4f}°E, {adcp_lat:.4f}°S', 
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])

output_file = 'model_adcp_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_file}")
print("\n=== Comparison Complete ===")
