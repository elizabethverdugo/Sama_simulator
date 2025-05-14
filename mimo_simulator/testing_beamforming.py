





for beam, [phi_tilt, theta_tilt] in enumerate(zip(beam_ant.point_phi, beam_ant.point_theta)):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    print(beam_ant.beam_gain[beam, :, theta_tilt].min(), beam_ant.beam_gain[beam, :, theta_tilt].max())
    ax.plot(
        np.deg2rad(beam_ant.phi),
        beam_ant.beam_gain[beam, :, theta_tilt],
        label=fr'Azimuth cut ($\theta={theta_tilt}^\circ$)',
        linestyle='-'
    )
    ax.plot(
        np.deg2rad(beam_ant.theta),
        beam_ant.beam_gain[beam, phi_tilt, :],
        label=fr'Elevation cut ($\phi={phi_tilt}^\circ$)',
        linestyle='--'
    )
    ax.set_title("Array beam pattern cuts", fontsize=14)
    ax.set_rlim(-30, 3)
    ax.set_rticks([-30, -20, -10, 0])
    leg = ax.legend(loc='upper right')
    plt.show()
    leg.set_draggable(True)