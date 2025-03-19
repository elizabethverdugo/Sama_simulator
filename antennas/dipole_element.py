import numpy as np
import matplotlib.pyplot as plt

class DipoleElement:
    def __init__(self, max_gain=2.15, plot=False):
        self.max_gain = max_gain
        self.phi = np.linspace(0,360,360, endpoint=False)
        # Elevation:
        self.theta = np.linspace(-180, 180,360)

        self.gain_pattern = self._calculate_gain_pattern()

        if plot:
            self.plot()

    def _calculate_gain_pattern(self):
        gain_pattern = np.outer(
            np.ones_like(self.phi),
            np.sin(np.radians(self.theta)) **2
        )

        gain_pattern = self.max_gain + 10 * np.log10(gain_pattern + 1e-12)

        return gain_pattern


    def plot(self):
        #plot_theta = np.linspace(0,180,180)
        #plot_gain = self.gain_pattern[0,:180]
        #plot_theta = np.concatenate((self.theta, self.theta[::-1]))
        #plot_gain = np.concatenate((self.gain_pattern[0,:180], self.gain_pattern[0, :180][::-1]))

        plt.figure(figsize=(8,6))
        plt.plot(self.theta,self.gain_pattern[0,:], label='Dipole Pattern (Elevation)')
        plt.xlabel('Elevation angle, theta (Degrees)')
        plt.ylabel('Gain (dBi)')
        plt.title('Dipole Antenna Radiation Pattern (Elevation)')
        plt.legend()
        plt.grid(True)
        plt.ylim(-30, self.max_gain + 1)
        plt.show()


        plt.figure(figsize=(8,6))
        plt.plot(self.phi,self.gain_pattern[:,90], label='Dipole Pattern (Azimuth)')
        plt.xlabel('Azimuth angle, phi (Degrees)')
        plt.ylabel('Gain (dBi)')
        plt.title('Dipole Antenna Radiation Pattern (Azimuth)')
        plt.legend()
        plt.grid(True)
        plt.show()

        #Polar plots:
        plt.figure(figsize=(6,6))
        plt.polar(
            np.radians(self.theta),
            self.gain_pattern[0,:],
            label='Dipole Pattern (Elevation)'
        )
        plt.title('Dipole Antenna Radiation Pattern (Elevation) - Polar')
        plt.legend()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.polar(
            np.radians(self.phi),
            self.gain_pattern[:, 90],
            label='Dipole Pattern (Azimuth)'
        )
        plt.title('Dipole Antenna Radiation Pattern (Azimuth) - Polar')
        plt.legend()
        plt.show()
