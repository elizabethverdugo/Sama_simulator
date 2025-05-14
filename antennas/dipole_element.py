import numpy as np
import matplotlib.pyplot as plt


class DipoleElement:
    def __init__(self, max_gain=2.15, plot=False):
        self.max_gain = max_gain
        self.phi = np.linspace(0,360,360, endpoint=False)
        # Elevation:
        self.theta = np.linspace(-180, 180,360)

        self.gain_pattern = self._calculate_gain_pattern()

        self.theta = np.mod(self.theta, 360)
        sorted_idx = np.argsort(self.theta)
        self.theta = self.theta[sorted_idx]
        self.gain_pattern = self.gain_pattern[:, sorted_idx]

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
        plt.rcParams['font.size'] = 14
        fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))
        ax1.plot(np.radians(self.theta),
            self.gain_pattern[0,:],
            label=r'Elevation cut ($\phi$=0°)'
        )
        theta_idx = np.argmin(np.abs(self.theta - 90))
        ax1.plot(np.radians(self.phi),
                 self.gain_pattern[:, theta_idx],
                 linestyle='--', linewidth=2, color='crimson',
                 label=r'Azimuth cut ($\theta$=90°)'
                 )


        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rlabel_position(135)

        ax1.set_rlim(-30,3)
        ax1.set_rticks([-30, -20, -10, 0, 2])
        #ax1.set_title('Dipole Antenna Radiation Pattern (Elevation and Azimuth) - Polar')
        leg=ax1.legend(loc='upper right')
        leg.set_draggable(True)
        ax1.grid(True)
        #ax1.text(np.radians(90), ax1.get_rmax() + 5, 'Gain(dB)',
         #        ha='center', va='bottom', fontsize=12)
        #ax1.set_ylabel('Gain (dBi)', labelpad=20)

        plt.show()
