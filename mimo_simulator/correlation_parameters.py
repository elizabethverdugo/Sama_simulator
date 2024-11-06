import numpy as np
from scipy.linalg import sqrtm


def calculate_correlation(N, rho_DS_AS, rho_SF_AS, rho_SF_DS, zetha_SF, eps_AS, mu_AS, eps_DS, mu_DS, sigma_SH):
    #N: Number of paths
    #rho_DS_AS: Correlation between DS and AS
    #rho_SF_AS: Correlation between SF and AS
    #rho_SF_DS: Correlation between SF and DS
    #zetha_SF: SF correlation
    #eps_AS, mu_AS: Parameter for AS
    #eps_DS, mu_DS: Parameter for DS
    #sigma_SH: Shadow fading std dev

    #DS: Delay spread
    #AS: Azimuth spread
    #SF: shadow fading (lognormal usually)

    #A: Intra-site Correlation matrix: eq (5.6-1)
     A =  np.array([[1, rho_DS_AS, rho_SF_DS],
                    [rho_DS_AS, 1, rho_SF_AS],
                    [rho_SF_DS, rho_SF_AS, 1]], dtype=float)


    #B: Inter-site Correlation eq (5.6-2)
     B =  np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, zetha_SF]], dtype=float)

    # #Debuging the matriz A
    #  print("A:")
    #  print(A)
    #  print("B:")
    #  print(B)


     A_minus_B = A-B        #Temporal term

     C = sqrtm(A_minus_B)

     # print("A-C:")
     # print(A_minus_B)
     #
     # print("A-C non-negatives:")
     # print(A_minus_B_clipped)
     #
     print("C:")
     print(C)

    #Here the step to calculate alpha_n (for the nth BS) from the matrix C and expression (5.6-3)
    # Generate zetha globally
     zetha = np.random.randn(3, 1)     #This one is global,  # applicable to all bases (do not generate for each n)

    #Generate W for each path N:
     W = np.random.randn(3, N)         #W is 3xN

     tmp1 =  np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, np.sqrt(zetha_SF)]], dtype=float)      #temporal variable


     tmp2 = np.dot(C,W) + np.dot(tmp1,zetha)            #temporal variable
     #tmp2 is the 3x1 vector in expression (5.6-3). Now it is a 3xN matrix

     alpha_n = tmp2[0, :]   #first row
     beta_n = tmp2[1, :]    #second row
     gamma_n = tmp2[2, :]   #third row

     sigma_DS = 10 ** (np.multiply(eps_DS,alpha_n) + mu_DS)
     sigma_AS = 10 ** (np.multiply(eps_AS,beta_n) + mu_AS)
     sigma_SF = 10 ** (np.multiply(sigma_SH, gamma_n) /10)


     #return [A, C, eps_AS, mu_AS, sigma_DS, sigma_AS, sigma_SF, D]
     return [sigma_DS, sigma_AS, sigma_SF]
