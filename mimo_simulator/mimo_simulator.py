import numpy as np

from mimo_simulator.allocation_utilities import water_filling, water_filling1, uniform_power
from mimo_simulator.geometry.distance import orientation_SAMA
from mimo_simulator.Environment import SuburbMacro
from mimo_simulator.channel import calculate_correlation, delay
from mimo_simulator.processing.power_n import calculate_power
from mimo_simulator.processing.angles import calculate_aod, calculate_aoa
from mimo_simulator.channel import (
    acquire_subpath_parameters,
    calculate_offset_aoas,
    associate_subpath,
    calculate_angles,
    calculate_gains_BS,
    calculate_gains_MS
)
from mimo_simulator.capacity_utilities import capacity_bits_per_hz
from mimo_simulator.pathloss import calculate_path_loss
from mimo_simulator.channel_coefficients import calculate_channel_coef
import matplotlib.pyplot as plt
import seaborn as sns
from mimo_simulator.Plot_fun import ecdf


class MIMOSimulator:
    def __init__(self, param_dict, hrx=1.5,
                 beam_pointing=None, downtilts=None, n_sectors=1, beam_gain=None):
        self.S = param_dict['S']
        self.U = param_dict['U']
        self.N = param_dict['N']
        self.M = param_dict['M']
        self.rho_DS_AS = float(param_dict['rho_DS_AS'])
        self.rho_SF_AS = float(param_dict['rho_SF_AS'])
        self.rho_SF_DS = float(param_dict['rho_SF_DS'])
        self.zetha_SF = float(param_dict['zetha_SF'])
        #self.Antenna_Sectors = param_dict['Antenna_Sectors']
        self.frequency = float(param_dict['f_c'])
        self.power = float(param_dict['power'])
        self.noise = float(param_dict['noise'])
        self.delay_method = param_dict.get('delay_method', '3GPP')  # Options: "3GPP", "distance", "both"

        self.channels = []  # to store later
        self.hrx = hrx
        self.beam_pointing = beam_pointing
        self.downtilts = downtilts
        self.n_sectors = n_sectors
        self.beam_gain = beam_gain

    def integrate_channels(self, ran_data):
        """
        import the path data from the main SAMA program
        ran_data: dict or struct object with the channels data
        """
        self.channels.clear()

        # loop over each BS,UE pair :
        for ran_channel in ran_data['channels']:
            channel_data = ChannelData(bs_id=ran_channel['bs_id'], ue_id=ran_channel['ue_id'])  # object!

            for path_info in ran_channel['paths']:
                # path parameters from the main SAMA block
                path_data = PathInfo(
                    delay=path_info['delay'],
                    power=path_info['power'],
                    aod_angle=path_info['aod_angle'],
                    aoa_angle=path_info['aoa_angle']
                )
                channel_data.add_path(path_data)

            self.channels.append(channel_data)






    def run_mimo(
            self,
            d,
            thetaBS,
            OmegaMS,
            hrx,
            delay_method="3GPP",
            **kwargs):
        #**kwargs:ensures that the argument which we pass is stored as a dictionary in the function
        """
        Simulate MIMO using parameters and a delay method.

        Parameters:
        - d: Distance matrix.
        - thetaBS: Angle of departure from BS.
        - OmegaMS: MS orientation matrix.
        - delay_method: Delay calculation method ("3GPP", "distance", "both").
        - kwargs: Additional parameters passed dynamically.

        Returns:
        - Simulation results.

        """

        # Initialize the environment (Suburban Macrocell example)
        env = SuburbMacro(system_type="MIMO", h_ms = self.hrx)



        # Calculate distances and orientation parameters
        OmegaBS, thetav, thetaMS = orientation_SAMA(
            thetaBS=thetaBS,
            OmegaMS=OmegaMS,
            beam_pointing=self.beam_pointing
        )


        #print(f"Running MIMO Simulation for distance={distance}, AoD {thetaBS_in}, OmegaMS {OmegaMS},OmegaBS {OmegaBS}, thetav {thetav}, thetaMS {thetaMS}")

        num_bs, num_ues = d.shape
        self.sector_map = self._build_sector_map(thetaBS, self.beam_pointing)

        # Step 3: Determine DS, AS, and SF
        sigma_DS, sigma_AS, sigma_SF = calculate_correlation(
            N=num_bs,
            rho_DS_AS=self.rho_DS_AS,
            rho_SF_AS=self.rho_SF_AS,
            rho_SF_DS=self.rho_SF_DS,
            zetha_SF=self.zetha_SF,
            eps_AS=env.eps_AS,
            mu_AS=env.mu_AS,
            eps_DS=env.eps_DS,
            mu_DS=env.mu_DS,
            sigma_SH=env.sigma_SH
        )

        # Steps 4-6: Delays, Powers, and Angles
        #tn = delay(self.N, env.r_DS, sigma_DS)
        tn = delay(
            N=self.N,
            r_DS=env.r_DS,
            sigma_DS=sigma_DS,
            dist_map=d,
            c=3e8,
            delay_method=self.delay_method
        )

        Pn = calculate_power(
            r_DS=env.r_DS,
            tn=tn,
            sigma_DS=sigma_DS
        )

        delta_AoD, idx_order, bs_gains = calculate_aod(
            d=d,
            AoD_bs=thetaBS,
            N=self.N,
            r_AS=env.r_AS,
            sigma_AS=sigma_AS,
            mode = "realistic_gain",
            beam_pointing=self.beam_pointing,
            sector_map=self.sector_map,
            beam_gain=self.beam_gain,
            downtilts=self.downtilts
        )

        # Adjust delays and path powers (sorting)
        tn1 = np.take_along_axis(tn, idx_order, axis=-1)
        path_powers = np.take_along_axis(Pn, idx_order, axis=-1)

        subpath_powers, subpath_phases, AoD_subpath_offsets = acquire_subpath_parameters(
            N=self.N,
            M=self.M,
            path_powers=path_powers
        )

        delta_AoA = calculate_aoa(
            num_bs=num_bs,
            num_ues=num_ues,
            N=self.N,
            path_powers=path_powers,
            sigma_AS=sigma_AS,
            AS_type="laplacian")


        AoA_subpath_offsets = calculate_offset_aoas(
            M=self.M,
            path_powers=path_powers
        )


        reassociated_ms_subpath_offsets = associate_subpath(
            bs_subpath_offsets=AoD_subpath_offsets,
            ms_subpath_offsets=AoA_subpath_offsets)


        # Calculate angles for each subpath (4D)
        aod_angles, aoa_angles = calculate_angles(
            theta_BS=thetaBS,
            delta_AoD=delta_AoD,
            AoD_offsets=AoD_subpath_offsets,
            theta_MS=thetaMS,
            delta_AoA=delta_AoA,
            AoA_offsets=reassociated_ms_subpath_offsets
        )

        ms_gains = calculate_gains_MS(
            Angle=aoa_angles
        )

        bs_gains_expandend = bs_gains[:, :, np.newaxis, np.newaxis]
        ms_gains_dB = 10 * np.log10(ms_gains)
        bs_gains_dB = 10 * np.log10(bs_gains_expandend)

        """
        fig1 = plt.figure(figsize=(12, 5))
        plt.hist(aod_angles.flatten(), bins=50, alpha=0.6, label="AoD Angles", edgecolor='black')
        plt.hist(aoa_angles.flatten(), bins=50, alpha=0.6, label="AoA Angles", edgecolor='black')
        plt.legend()
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Count")
        plt.title("Distribution of AoD and AoA Angles")
        plt.grid(True)

        fig2=plt.figure(figsize=(12, 5))
        plt.hist(ms_gains_dB.flatten(), bins=10, alpha=0.6, label="MS Gains", edgecolor='black')
        plt.hist(bs_gains_dB.flatten(), bins=50, alpha=0.6, label="BS Gains", edgecolor='black')
        plt.legend()
        plt.xlabel("Gain (dB)")
        plt.ylabel("Count")
        plt.title("Distribution of MS and BS Gains")
        plt.grid(True)
        plt.yscale("log")

        fig3=plt.figure(figsize=(8, 5))
        plt.scatter(aod_angles.flatten(), 10 * np.log10(bs_gains.flatten()), alpha=0.3, s=2)
        plt.xlabel("AoD Angle (degrees)")
        plt.ylabel("BS Gain (dB)")
        plt.title("BS Gain vs. AoD Angle")
        plt.grid()
        plt.show()
        """

        path_loss_db = calculate_path_loss(
            h_bs=env.h_bs,
            h_ms=env.h_ms,
            d=d,
            f_c=self.frequency,
            C=env.C
        )

        #To plot:
        path_loss_db_flat = path_loss_db.flatten()
        fig4, ax = plt.subplots(figsize=(12, 5))
        sns.histplot(path_loss_db_flat, bins=50, kde=True)
        ax.set_xlabel("Path Loss (dB)", fontsize=16, labelpad=8)
        ax.set_ylabel("Counts", fontsize=18, labelpad=8)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        #plt.title("Path Loss Distribution")
        ax.grid(True)
        plt.show()

        bs, ue = 0,0
        print("PL_dB :", path_loss_db[bs, ue])
        pl_pow = 10**(-path_loss_db[bs, ue]/10)
        path_loss_linear = 10 ** (-path_loss_db / 10)

        # adding randnomness to sigma SF
        sigma_SF_corr_lin = np.tile(sigma_SF[:, None], (1, num_ues))
        sigma_SF_corr_dB = 10 * np.log10(sigma_SF_corr_lin)
        std_SF_corr = sigma_SF_corr_dB.std()

        std_SF_iid = np.sqrt(max(0.0, 8.0 ** 2 - std_SF_corr ** 2))
        sigma_SF_iid_dB = np.random.normal(0.0, std_SF_iid, size=(num_bs, num_ues))
        sigma_SF_iid_lin = 10 ** (sigma_SF_iid_dB / 10)

        sigma_SF_link_lin = sigma_SF_corr_lin * sigma_SF_iid_lin

        pl_linear = path_loss_linear[:, :, None, None]
        sf_linear = sigma_SF_link_lin[:, :, None, None]

        adjusted_subpath_powers = (subpath_powers * pl_linear * sf_linear)

        # Expand powers across subpaths (M=10)
        adjusted_subpath_powers = np.tile(adjusted_subpath_powers, (1, 1, 1, 10))


        adjusted_subpath_powers_flat = adjusted_subpath_powers.flatten()




        """
        fig5=plt.figure(figsize=(12, 5))
        sns.histplot(adjusted_subpath_powers_flat, bins=50, kde=True)
        plt.xlabel("Adjusted Subpath Power")
        plt.ylabel("Counts")
        plt.title("Adjusted Subpath Power Distribution")
        plt.grid(True)
        plt.show()
        """


        h_matrix = calculate_channel_coef(
            num_BS=num_bs,
            num_MS=num_ues,
            N=self.N,
            M=self.M,
            S=self.S,
            U=self.U,
            theta_BS=thetaBS,
            theta_MS=thetaMS,
            delta_AoD=delta_AoD,
            delta_AoA=delta_AoA,
            AoD_offsets=AoD_subpath_offsets,
            AoA_offsets=reassociated_ms_subpath_offsets,
            subpath_powers=adjusted_subpath_powers,
            subpath_phases=subpath_phases,
            G_BS=bs_gains_expandend,
            G_MS=ms_gains,
            sigma_SF=sigma_SF,
            d_bs=0.5,
            d_ms=0.5,
            v=2,
            theta_v=thetav,
            f=self.frequency,
            time=0
        )


        #h_matrix shape is BSs, UEs, N, M, U, S

        #Flattening all rays to check the statistcics per ray
        h_ray_db = 20*np.log10(np.abs(h_matrix).ravel() + 1e-15)

        fig6=plt.figure(figsize=(10, 5))
        plt.hist(h_ray_db, bins=50)  # dB scale
        plt.xlabel("Channel coefficient magnitude (dB)")
        plt.ylabel("Count")
        plt.title("Histogram of |H| (in dB)")
        plt.grid(True)
        plt.show()

        power_split = 10*np.log10(self.N*self.M)        #power split per ray... see the paper

        # Select one random BS-MS pair, e.g. (0,0)
        H_sample = h_matrix[0, 0, :, :, :, :]  # shape (N,M,U,S)

        # Sum over paths/subpaths to see antenna correlation
        H_sum = np.sum(H_sample, axis=(0, 1))
        print("Summed H (antenna correlation matrix):\n", np.abs(H_sum))

        fig7=plt.imshow(np.abs(H_sum), cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title("Example antenna correlation (|H|) BS-MS (0,0)")
        plt.xlabel("BS antennas")
        plt.ylabel("MS antennas")
        plt.show()


        #for capacity we need the BS-UE pair, a narrow-band UxS matrix
        H_link = np.sum(h_matrix, axis=(2,3))       #new shape: (2,500,2,4)
        B, UEs, U, S = H_link.shape

        H_flat = H_link.reshape(B*UEs, U, S)    #newshape: (1000, 2, 4)

        singular_vals = np.zeros((H_flat.shape[0], min (U,S)))

        for k, H in enumerate(H_flat):
            _, s, _ = np.linalg.svd(H, full_matrices=False)
            singular_vals[k, :len(s)] = s

        P_uni = uniform_power(singular_vals, self.power)
        C_uni = capacity_bits_per_hz(singular_vals, P_uni, self.noise)

        P_wf = water_filling(singular_vals, self.power, self.noise)
        C_wf = capacity_bits_per_hz(singular_vals, P_wf, self.noise)

        d_up, F_up = ecdf(C_uni)
        d_wf, F_wf = ecdf(C_wf)


        

        plt.figure(figsize=(6, 4))
        plt.step(d_up, F_up, where='post', label='Uniform Power')
        plt.step(d_wf, F_wf, where='post', label='Water-filling')
        plt.xlabel('Spectral efficiency [bit s$^{-1}$ Hz$^{-1}$ ]', fontsize=16)
        plt.ylabel('Empirical CDF', fontsize=18)
        plt.grid(True)
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()

        """
        return {
            "h_matrix": h_matrix,
            "all_power_allocations": all_power_allocations,
            "aggregate_allocation": aggregate_allocation,
            "all_capacity": all_capacity,
            "aggregate_capacity": aggregate_capacity,
        }
        """

        # Return immediately after printing to isolate this test
        return {"distance_received": d,
                "Angle of departure": thetaBS,
                "MS antenna array orientation": OmegaMS,
                "BS antenna array Orientation": OmegaBS,
                "Angle of the velocity": thetav,
                "Angle between BS-MS LOS and the MS broadside":thetaMS,
                "delays": tn}


    def run_simulations(self, dist_map, az_map, ms_orientation=None, base_station_list=None, parameters=None, hrx=1.5):
        if dist_map is None or az_map is None:
            raise ValueError("Required data are not initialized.")

        # Initialize ms_orientation if not provided
        if ms_orientation is None:
            ms_orientation = self.initialize_ms_orientation(dist_map, parameters)

        num_bs = len(base_station_list)
        ms_orientation_expanded = np.tile(ms_orientation, (num_bs, 1))

        print("Running MIMO simulations...")
        mimo_output = self.run_mimo(
            d=dist_map,
            thetaBS=az_map,
            OmegaMS=ms_orientation_expanded,
            hrx=hrx
        )

        num_ues = dist_map.shape[1]
        num_features = 3  # distance, AoD, orientation
        mimo_results = np.zeros((num_bs, num_ues, num_features), dtype=float)
        mimo_results[:, :, 0] = mimo_output.get("distance_received", 0)
        mimo_results[:, :, 1] = mimo_output.get("Angle of departure", 0)
        mimo_results[:, :, 2] = mimo_output.get("MS antenna array orientation", 0)

        return mimo_results

    def initialize_ms_orientation(self, dist_map, parameters):
        """
        Initialize the MS orientation based on the provided parameters.
        """
        if dist_map is None:
            raise ValueError("Distance map (dist_map) is required to initialize MS orientation.")

        ms_orientation_config = parameters.get("ms_orientation", {})
        ms_orientation_enabled = ms_orientation_config.get("enabled", False)
        ms_orientation_random = ms_orientation_config.get("random", True)
        ms_orientation_range = ms_orientation_config.get("range", [0, 360])

        num_ues = dist_map.shape[1]

        if ms_orientation_enabled:
            if ms_orientation_random:
                ms_orientation = np.random.uniform(
                    low=ms_orientation_range[0],
                    high=ms_orientation_range[1],
                    size=num_ues
                )
            else:
                raise ValueError("Currently only random MS orientation is supported.")
        else:
            ms_orientation = np.zeros(num_ues)

        return ms_orientation

    def _build_sector_map(self, thetaBS, beam_pointing):
        n_bs, n_ue = thetaBS.shape
        sector_map = np.zeros_like(thetaBS, dtype=int)

        for i in range(n_bs):
            for j in range(n_ue):
                angle = thetaBS[i,j]
                sector_map[i,j] = np.argmin(np.abs(angle-beam_pointing))

        return sector_map




