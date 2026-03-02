import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob

# Load comparison data
comp_ds = xr.open_dataset('drifter_model_comparison.nc')
drifter_files = sorted(glob.glob('drifter_*.nc'))

print("Creating comparison plot...")

fig = plt.figure(figsize=(18, 10))

# 1. Map
ax1 = plt.subplot(2, 4, 1)
colors = plt.cm.tab10(np.linspace(0, 1, len(drifter_files)))
for idx, drif_file in enumerate(drifter_files):
    drif = xr.open_dataset(drif_file)
    drif_id = drif.attrs.get('drifter_id', idx)
    ax1.plot(drif['longitude'].values, drif['latitude'].values, 
             color=colors[idx], linewidth=2, alpha=0.7, label=f'D{drif_id}')
    ax1.scatter(drif['longitude'].values[0], drif['latitude'].values[0], 
                color=colors[idx], s=100, marker='o', edgecolors='black', linewidths=2)

ax1.set_xlabel('Longitude (°E)', fontsize=11)
ax1.set_ylabel('Latitude (°S)', fontsize=11)
ax1.set_title('Drifter Trajectories', fontsize=12, fontweight='bold')
ax1.legend(fontsize=7, ncol=2)
ax1.grid(alpha=0.3)

# 2. Speed scatter
ax2 = plt.subplot(2, 4, 2)
ax2.scatter(comp_ds.speed_obs.values, comp_ds.speed_model.values, 
            alpha=0.5, s=30, c='steelblue')
max_speed = max(comp_ds.speed_obs.max().values, comp_ds.speed_model.max().values)
ax2.plot([0, max_speed], [0, max_speed], 'r--', lw=2, label='1:1')
ax2.set_xlabel('Drifter Speed (m/s)', fontsize=11)
ax2.set_ylabel('Model Speed (m/s)', fontsize=11)
ax2.set_title('Speed Comparison', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_aspect('equal')

rmse = comp_ds.attrs['speed_rmse']
r2 = np.corrcoef(comp_ds.speed_obs.values, comp_ds.speed_model.values)[0, 1]**2
ax2.text(0.05, 0.95, f'RMSE={rmse:.4f} m/s\nR²={r2:.3f}',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 3. Temperature
ax3 = plt.subplot(2, 4, 3)
ax3.scatter(comp_ds.temp_obs.values, comp_ds.temp_model.values, 
            alpha=0.5, s=30, c='orangered')
min_t = min(comp_ds.temp_obs.min().values, comp_ds.temp_model.min().values)
max_t = max(comp_ds.temp_obs.max().values, comp_ds.temp_model.max().values)
ax3.plot([min_t, max_t], [min_t, max_t], 'r--', lw=2, label='1:1')
ax3.set_xlabel('Drifter Temp (°C)', fontsize=11)
ax3.set_ylabel('Model Temp (°C)', fontsize=11)
ax3.set_title('Temperature', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_aspect('equal')

# 4. Salinity  
ax4 = plt.subplot(2, 4, 4)
ax4.scatter(comp_ds.salt_obs.values, comp_ds.salt_model.values, 
            alpha=0.5, s=30, c='seagreen')
min_s = min(comp_ds.salt_obs.min().values, comp_ds.salt_model.min().values)
max_s = max(comp_ds.salt_obs.max().values, comp_ds.salt_model.max().values)
ax4.plot([min_s, max_s], [min_s, max_s], 'r--', lw=2, label='1:1')
ax4.set_xlabel('Drifter Salinity (PSU)', fontsize=11)
ax4.set_ylabel('Model Salinity (PSU)', fontsize=11)
ax4.set_title('Salinity', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_aspect('equal')

# 5-8. Error distributions
ax5 = plt.subplot(2, 4, 5)
speed_error = comp_ds.speed_model.values - comp_ds.speed_obs.values
ax5.hist(speed_error, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax5.axvline(0, color='r', ls='--', lw=2)
ax5.set_xlabel('Speed Error (m/s)', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title('Speed Error Distribution', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

ax6 = plt.subplot(2, 4, 6)
ax6.scatter(comp_ds.u_obs.values, comp_ds.u_model.values, alpha=0.5, s=30, c='purple')
max_u = max(abs(comp_ds.u_obs).max().values, abs(comp_ds.u_model).max().values)
ax6.plot([-max_u, max_u], [-max_u, max_u], 'r--', lw=2, label='1:1')
ax6.set_xlabel('Drifter U (m/s)', fontsize=11)
ax6.set_ylabel('Model U (m/s)', fontsize=11)
ax6.set_title('U-velocity', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)
ax6.set_aspect('equal')

ax7 = plt.subplot(2, 4, 7)
ax7.scatter(comp_ds.v_obs.values, comp_ds.v_model.values, alpha=0.5, s=30, c='brown')
max_v = max(abs(comp_ds.v_obs).max().values, abs(comp_ds.v_model).max().values)
ax7.plot([-max_v, max_v], [-max_v, max_v], 'r--', lw=2, label='1:1')
ax7.set_xlabel('Drifter V (m/s)', fontsize=11)
ax7.set_ylabel('Model V (m/s)', fontsize=11)
ax7.set_title('V-velocity', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)
ax7.set_aspect('equal')

ax8 = plt.subplot(2, 4, 8)
temp_error = comp_ds.temp_model.values - comp_ds.temp_obs.values
ax8.hist(temp_error, bins=20, color='orangered', alpha=0.7, edgecolor='black')
ax8.axvline(0, color='r', ls='--', lw=2)
ax8.set_xlabel('Temp Error (°C)', fontsize=11)
ax8.set_ylabel('Frequency', fontsize=11)
ax8.set_title('Temperature Error', fontsize=12, fontweight='bold')
ax8.grid(alpha=0.3)

plt.suptitle(f'Drifter-SHYFEM Comparison ({len(drifter_files)} drifters, {len(comp_ds.obs)} observations)', 
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig('drifter_model_comparison.png', dpi=150, bbox_inches='tight')
print('✓ Saved drifter_model_comparison.png')
