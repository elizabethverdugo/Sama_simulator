import random
import numpy as np

def orientation(R):
    # Given a BS (fix), this is the orientation for each MS, taking one at each time
    #R is the radius of the hexagon
    d = np.round(random.uniform(0,R),2)        #generate a random distance
    OmegaBS = 0     #BS antenna array Orientation, defined as the difference between
                    # the broadside of the BS array and the absolute north reference direction
    OmegaMS = np.round(random.uniform(0,360),2)    #The MS antenna array orientation

    thetaBS = np.round(random.uniform(0,360),2)   #LOS AoD direction between the BS and MS,
                                                        # with respect to the broadside of the BS array
    thetav = np.round(random.uniform(0,360),2)   #Angle of the velocity vector with respect to the MS broadside (maybe won't use it)
    tmp = np.abs(OmegaBS - OmegaMS + thetaBS + 180)
    thetaMS = np.round(tmp,2)       #Angle between BS-MS LOS and the MS broadside

    return [d, OmegaBS, OmegaMS, thetaBS, thetav, thetaMS]


def orientation_SAMA(thetaBS, OmegaMS):
    # Given a BS (fix), this is the orientation for each MS, taking one at each time
    #thetaBS: LOS AoD direction between the BS and MS, with respect to the broadside of the BS array
    #OmegaMS: The MS antenna array orientation

    OmegaBS = 0     #BS antenna array Orientation, defined as the difference between
                    # the broadside of the BS array and the absolute north reference direction
    thetav = np.round(random.uniform(0,360),2)   #Angle of the velocity vector with respect to the MS broadside (maybe won't use it)

    tmp = np.abs(OmegaBS - OmegaMS + thetaBS + 180)
    thetaMS = np.round(tmp,2)       #Angle between BS-MS LOS and the MS broadside

    return [OmegaBS, thetav, thetaMS]