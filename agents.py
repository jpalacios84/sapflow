import numpy as np

# Time = 1s
# Volume = liters
S_IN_DAY = 24*60*60

sm = {
    'crack' : {
        'vol' : 1,
        'ts_vol' : [None]*S_IN_DAY,
    },

    'plant' : {
        'vol' : 1,
        'ts_vol' : [None]*S_IN_DAY,
    },

    'rain'          : 1/S_IN_DAY,
    'snowmelt'      : 1/S_IN_DAY,
    'RWU_crack'     : 1/S_IN_DAY,
    'free_drainage' : 0.75/S_IN_DAY,
    'transpiration' : 0.2/S_IN_DAY,
}

for n in range(S_IN_DAY):
    is_rwu_active = n < 12*60*60


    # Crack <- Environment
    ### Rain
    sm['crack']['vol'] += sm['rain']

    ### Snowmelt
    sm['crack']['vol'] += sm['snowmelt']

    # Crack -> Plant
    if is_rwu_active:
        rwu_crack = min(sm['RWU_crack'], sm['crack']['vol'])
        sm['crack']['vol'] = max(0, sm['crack']['vol'] - sm['RWU_crack'])

    # Crack -> free_drainage
    free_drainage = min(sm['free_drainage'], sm['crack']['vol'])
    sm['crack']['vol'] = max(0, sm['crack']['vol'] - sm['free_drainage'])

    # Plant <- RWU_crack
    if is_rwu_active:
        sm['plant']['vol'] += rwu_crack

    sm['plant']['vol'] = max(0, sm['plant']['vol'] - sm['transpiration'])

    for k in ['crack', 'plant']:
        sm[k]['ts_vol'][n] = sm[k]['vol']

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1,figsize=(10,5))

for k in ['crack', 'plant']:
    ax.plot(sm[k]['ts_vol'], label=k.upper())

ax.set_xlim([0, S_IN_DAY])
ax.set_ylim([0, 3])
ax.grid()
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Volume (l)')

plt.tight_layout()
plt.show()
