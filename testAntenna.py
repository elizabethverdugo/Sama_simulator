import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from antennas.dipole_element import DipoleElement
from antennas.beamforming import Beamforming_Antenna

dipole_antenna = DipoleElement(max_gain=2.15, plot=True)

"""  
#Only one pointing direction:
array = Beamforming_Antenna(
    ant_element = dipole_antenna,
    frequency = 2.6e9,
    n_rows = 2,
    n_columns = 2,
    horizontal_spacing = 0.5,
    vertical_spacing = 0.5,
    point_theta = [60],
    point_phi = [90]
)
array.calculate_pattern(plot=True)
"""

#Multiple pointing directions:

array = Beamforming_Antenna(
    ant_element = dipole_antenna,
    frequency = 2.6e9,
    n_rows = 8,
    n_columns = 8,
    horizontal_spacing = 0.5,
    vertical_spacing = 0.5,
    point_theta=[60],
    point_phi=[90]
)

theta_points = [30, 60, 45]
phi_points = [45, 90, 45]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

for theta, phi in zip(theta_points, phi_points):
    array.change_beam_configuration(point_theta=[theta], point_phi=[phi])
    array.calculate_pattern()

    ax.plot(
        np.radians(array.phi),
        array.beam_gain[0, :, 180 - theta],
        label=f'Beam theta={theta}, phi={phi}'
    )


ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_ylim(-30,10)
ax.set_title('Multiple Beam Radiation Patterns (Polar)')
ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))
plt.show()

