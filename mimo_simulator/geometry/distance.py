import random
import numpy as np

def orientation(R):
    # R is not used anymore!! 16.05.2025 (It was before having the map of distances)
    # Given a BS (fix), this is the orientation for each MS, taking one at each time
    #R is the radius of the hexagon
    d = np.round(random.uniform(0,R),2)        #generate a random distance
    OmegaBS = 0     #BS antenna array Orientation, defined as the difference between
                    # the broadside of the BS array and the absolute north reference direction
    OmegaMS = np.round(random.uniform(0,360),2)    #The MS antenna array orientation

    thetaBS = np.round(random.uniform(0,360),2)   #LOS AoD direction between the BS and MS,
                                                        # with respect to the broadside of the BS array
    #After adding a given antenna, this is the custom beamforming

    thetav = np.round(random.uniform(0,360),2)   #Angle of the velocity vector with respect to the MS broadside (maybe won't use it)
    tmp = np.abs(OmegaBS - OmegaMS + thetaBS + 180)
    thetaMS = np.round(tmp,2)       #Angle between BS-MS LOS and the MS broadside

    return [d, OmegaBS, OmegaMS, thetaBS, thetav, thetaMS]


def orientation_SAMA(thetaBS, OmegaMS, beam_pointing):
    # Given a BS (fix), this is the orientation for each MS, taking one at each time
    #thetaBS: LOS AoD direction between the BS and MS, with respect to the broadside of the BS array
    #OmegaMS: The MS antenna array orientation
    #OmegaBS changing after using beamforming of dipole array

    #OmegaBS = 0     #BS antenna array Orientation, defined as the difference between
                    # the broadside of the BS array and the absolute north reference direction

    OmegaBS = np.zeros_like(thetaBS)

    n_BS = thetaBS.shape[0]
    n_UE = thetaBS.shape[1]
    for i in range(n_BS):
        for j in range(n_UE):
            angle = thetaBS[i,j]
            #Find the closest beam direction from the base station's predefined beams by
            #computing the absolute angular difference
            closest_beam = beam_pointing[np.argmin(np.abs(angle - beam_pointing))]
            #Assign the closest beam direction as the orientation of the BS toward this UE
            OmegaBS[i, j] = closest_beam


    thetav = np.round(random.uniform(0,360),2)   #Angle of the velocity vector with respect to the MS broadside (maybe won't use it)

    tmp = np.abs(OmegaBS - OmegaMS + thetaBS + 180)
    thetaMS = np.round(tmp,2)       #Angle between BS-MS LOS and the MS broadside

    return [OmegaBS, thetav, thetaMS]