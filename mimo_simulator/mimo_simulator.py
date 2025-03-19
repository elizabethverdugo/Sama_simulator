import numpy as np
import yaml
from mimo_simulator.distance_utility import orientation, orientation_SAMA
from mimo_simulator.Environment import SuburbMacro, UrbMacro, UrbMicro
from mimo_simulator.correlation_parameters import calculate_correlation
from mimo_simulator.delay import delay
from mimo_simulator.power_n import calculate_power
from mimo_simulator.angles_utilities import calculate_aod, calculate_aoa
from mimo_simulator.subpath_utilities import (
    acquire_subpath_parameters,
    calculate_offset_aoas,
    associate_subpath,
    calculate_angles,
    calculate_gains_BS,
    calculate_gains_MS,
    compute_N0,
    power_azimuth_spectrum
)

from mimo_simulator.pathloss import calculate_path_loss
from mimo_simulator.channel_coefficients import calculate_channel_coef
from mimo_simulator.allocation_utilities import water_filling, uniform_allocation
from mimo_simulator.capacity_utilities import calculate_capacity
import matplotlib.pyplot as plt
import seaborn as sns


class PathInfo:
    def __init__(self, delay, power, aod_angle, aoa_angle):
        """
        Initialize the path data BS-UnicodeError
        Parameters:
            -delay: path delays
            -power: path powers
            -aod_angle: angles of departure
            -aoa_angle: angles of arrival
        """

        self.delay = delay
        self.power = power
        self.aod_angle = aod_angle
        self.aoa_angle = aoa_angle

    def summarize(self):
        """
        To print the path info..
        """
        return {
            'delay': self.delay,
            'power': self.power,
            'aod_angle': self.aod_angle,
            'aoa_angle': self.aoa_angle
        }


class ChannelData:
    def __init__(self, bs_id, ue_id):
        """
        Channel data with multiple paths for BS-UE pars
        Parameters:
            -bs_id: ID of the BS
            -ue_id: ID of the UE
        """

        self.bs_id = bs_id
        self.ue_id = ue_id
        self.paths = []

    def add_path(self, path_info):
        """
        To add a path...
        """
        self.paths.append(path_info)

    def summarize(self):
        """
        To print a summary of paths
        """
        path_summaries = [path.summarize() for path in self.paths]
        return {
            'bs_id': self.bs_id,
            'ue_id': self.ue_id,
            'paths': path_summaries
        }

class MIMOSimulator:
    def __init__(self, param_dict):
        self.S = param_dict['S']
        self.U = param_dict['U']
        #self.BS = param_dict['BS']
        #self.MS = param_dict['MS']
        self.N = param_dict['N']
        self.M = param_dict['M']
        self.rho_DS_AS = float(param_dict['rho_DS_AS'])
        self.rho_SF_AS = float(param_dict['rho_SF_AS'])
        self.rho_SF_DS = float(param_dict['rho_SF_DS'])
        self.zetha_SF = float(param_dict['zetha_SF'])
        self.Antenna_Sectors = param_dict['Antenna_Sectors']
        self.frequency = float(param_dict['f_c'])
        self.power = float(param_dict['power'])
        self.noise = float(param_dict['noise'])
        self.delay_method = param_dict.get('delay_method', '3GPP')  # Options: "3GPP", "distance", "both"

        self.channels = []  # to store later

    def add_channel_data(self, channel_data):
        # add channel data for BS-UE paths

        self.channels.append(channel_data)  # to be able to make loops later

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
        env = SuburbMacro(system_type="MIMO")

        # Calculate distances and orientation parameters
        OmegaBS, thetav, thetaMS = orientation_SAMA(
            thetaBS=thetaBS,
            OmegaMS=OmegaMS
        )


        #print(f"Running MIMO Simulation for distance={distance}, AoD {thetaBS_in}, OmegaMS {OmegaMS},OmegaBS {OmegaBS}, thetav {thetav}, thetaMS {thetaMS}")

        num_bs, num_ues = d.shape

        """
        not necessary anymore:
        # Calculate distances and orientation parameters
        d, OmegaBS, OmegaMS, thetaBS, thetav, thetaMS = orientation(env.R)
        """

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

        delta_AoD, idx_order = calculate_aod(
            d=d,
            AoD_bs=thetaBS,
            N=self.N,
            r_AS=env.r_AS,
            sigma_AS=sigma_AS,
            Antenna_Sectors=self.Antenna_Sectors
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

        # Calculate antenna gains
        bs_gains = calculate_gains_BS(
            Angle=aod_angles,
            Antenna_Sectors=self.Antenna_Sectors
        )

        print("Min BS Gain (dB):", 10 * np.log10(np.min(bs_gains)))

        ms_gains = calculate_gains_MS(
            Angle=aoa_angles
        )

        """
        theta_values = aod_angles
        sigma = np.std(theta_values)
        theta_mean = np.mean(theta_values)

        N0 = compute_N0(
            theta_mean=theta_mean,
            sigma=sigma,
            theta_range=np.linspace(-180, 180, num=1000)
        )

        pas_values = power_azimuth_spectrum(
            theta=theta_values,
            theta_mean=theta_mean,
            sigma=sigma,
            N0=N0
        )


        theta_flat = theta_values.flatten()
        pas_flat = pas_values.flatten()

        fig0 = plt.figure(figsize=(12, 5))
        plt.figure(figsize=(10, 5))
        plt.plot(theta_flat, 10 * np.log10(pas_flat), '.', alpha=0.5)  # Scatter plot to avoid overlapping
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Power Azimuth Spectrum (dB)")
        plt.title("Flattened Power Azimuth Spectrum for All BS-UE-Paths")
        plt.grid(True)
        plt.show()
        """
        ms_gains_dB = 10 * np.log10(ms_gains)
        bs_gains_dB = 10 * np.log10(bs_gains)

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

        path_loss_linear = 10 ** (path_loss_db / 10)
        sigma_SF_linear = 10 ** (sigma_SF / 10)

        adjusted_subpath_powers = (
                subpath_powers / (path_loss_linear[:, :, np.newaxis, np.newaxis] *
                sigma_SF_linear[:, np.newaxis, np.newaxis, np.newaxis])
        )


        # Expand powers across subpaths (M=10)
        adjusted_subpath_powers = np.tile(adjusted_subpath_powers, (1, 1, 1, 10))

        #To plot:
        path_loss_db_flat = path_loss_db.flatten()
        adjusted_subpath_powers_flat = adjusted_subpath_powers.flatten()

        """
        fig4= plt.figure(figsize=(12, 5))
        sns.histplot(path_loss_db_flat, bins=50, kde=True)
        plt.xlabel("Path Loss (dB)")
        plt.ylabel("Counts")
        plt.title("Path Loss Distribution")
        plt.grid(True)

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
            G_BS=bs_gains,
            G_MS=ms_gains,
            sigma_SF=sigma_SF,
            d_bs=0.5,
            d_ms=0.5,
            v=2,
            theta_v=thetav,
            f=self.frequency,
            time=0
        )


        H_mag = np.abs(h_matrix.flatten())

        fig6=plt.figure(figsize=(10, 5))
        plt.hist(20 * np.log10(H_mag + 1e-12), bins=50)  # dB scale
        plt.xlabel("Channel coefficient magnitude (dB)")
        plt.ylabel("Count")
        plt.title("Histogram of |H| (in dB)")
        plt.grid(True)
        plt.show()

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

        """
        
        # Calculate power allocation and capacity
        all_power_allocations = []
        all_capacity = []

        H_3dimension = h_matrix.shape[2]  # number of paths
        for i in range(H_3dimension):
            U, S_Values, Vh = np.linalg.svd(h_matrix[:, :, i])
            S_diag = np.zeros((U.shape[0], Vh.shape[0]), dtype=float)
            np.fill_diagonal(S_diag, S_Values)
            SV_square = np.square(S_diag)

            # Power allocation using water filling
            power_allocation = water_filling(SV_square, self.power, self.noise)
            all_power_allocations.append(power_allocation)

            # Capacity calculation
            Capacity = calculate_capacity(S_diag, power_allocation, self.noise)
            all_capacity.append(Capacity)

        # Convert to numpy arrays for easier handling
        all_power_allocations = np.array(all_power_allocations)
        all_capacity = np.array(all_capacity)

        aggregate_allocation = np.sum(all_power_allocations, axis=0)
        aggregate_capacity = np.sum(all_capacity, axis=0)

        return {
            "h_matrix": h_matrix,
            "all_power_allocations": all_power_allocations,
            "aggregate_allocation": aggregate_allocation,
            "all_capacity": all_capacity,
            "aggregate_capacity": aggregate_capacity,
        }"""
        # Return immediately after printing to isolate this test
        return {"distance_received": d,
                "Angle of departure": thetaBS,
                "MS antenna array orientation": OmegaMS,
                "BS antenna array Orientation": OmegaBS,
                "Angle of the velocity": thetav,
                "Angle between BS-MS LOS and the MS broadside":thetaMS,
                "delays": tn}


    def run_simulations(self, dist_map, az_map, ms_orientation=None, base_station_list=None, parameters=None):
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
            OmegaMS=ms_orientation_expanded
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
