from mimo_simulator.comsys import *

class Environment:
    def __init__(self, scenary, h_bs=None, h_ms=None, C=None, d=None, R=None):
        self.scenary = scenary
        if scenary == "Suburban Macro":
            self.h_bs = 32  # default value for #height base station in Suburban Macro
            self.h_ms = 1.5 #height mobile station
            self.C = 0  #Constant factor
            self.d = 3  #distance BS to BS
            self.R = 1700 #Hexagonal radius


        elif scenary == "Urban Macro":
            self.h_bs = 32  # default value for h_bs in Suburban Macro
            self.h_ms = 1.5
            self.C = 3  #Constant factor
            self.d = 3  #distance BS to BS
            self.R = 1200  #Hexagonal radius


        elif scenary == "Urban Micro":
            self.h_bs = 12.5
            self.h_ms = 1.5
            self.C = None
            self.d = 1  #distance BS to BS
            self.R = 500 #Hexagonal radius


        else:
            self.h_bs = h_bs if h_bs is not None else 0  # or assign another default value
            self.h_ms = h_ms if h_ms is not None else 0
            self.C = C if C is not None else 0
            self.d = d if d is not None else 0
            self.R = R if R is not None else 0
            print("Warning: No valid scenario option was selected. Using provided parameters as defaults.")
        # End of the if block

class SuburbMacro(Environment, CommunicationSystem):
    def __init__(self, system_type="default_system_type", mean_AS=0, dBS = 0, eps_AS=0, mu_AS=0, eps_DS=0, mu_DS=0, sigma_SH=0, r_DS = 0, r_AS=0):
        Environment.__init__(self, "Suburban Macro")  # No need to pass h_bs, h_ms, C, and d

        if system_type == "MIMO":
            mean_AS = 5  # Default value for mean_AS when system_type is "MIMO"
            dBS = 6  # BS array elements spacing (in terms of lambda) Should this be an input at Config.py?
            # These values depend on the scenary
            eps_AS = 0.13   #parameters of lognormal function
            mu_AS = 0.69
            eps_DS = 0.18
            mu_DS = -6.18
            sigma_SH = 8
            r_DS = 1.4
            r_AS = 1.2


        CommunicationSystem.__init__(self, system_type, mean_AS,dBS, eps_AS, mu_AS, eps_DS, mu_DS, sigma_SH, r_DS, r_AS)


class UrbMacro(Environment, CommunicationSystem):
    def __init__(self,system_type="default_system_type", mean_AS=0, dBS = 0, eps_AS=0, mu_AS=0, eps_DS=0, mu_DS=0, sigma_SH=0, r_DS = 0, r_AS=0):
        Environment.__init__(self, "Urban Macro")

        if system_type == "MIMO":
            mean_AS = 2  # Default value for mean_AS when system_type is "MIMO"
            dBS = 4  # BS array elements spacing (in terms of lambda)
            eps_AS = 0.34
            mu_AS = 0.810
            eps_DS = 0.288
            mu_DS = -6.80
            sigma_SH = 8
            r_DS = 1.7
            r_AS = 1.3

        CommunicationSystem.__init__(self, system_type, mean_AS, dBS, eps_AS, mu_AS, eps_DS, mu_DS, sigma_SH, r_DS, r_AS)

class UrbMicro(Environment):
    def __init__(self,system_type="default_system_type", mean_AS=0, dBS = 0, eps_AS=0, mu_AS=0, eps_DS=0, mu_DS=0, sigma_SH=0, r_DS = 0, r_AS=0):
        Environment.__init__(self, "Urban Micro")
        if system_type == "MIMO":
            mean_AS = 16  # Default value for mean_AS when system_type is "MIMO"
            dBS = 2  # BS array elements spacing (in terms of lambda)
            eps_AS = 0
            mu_AS = 0
            eps_DS = 0
            mu_DS = 0
            sigma_SH = 10
            r_DS = 0        #it should be N/A
            r_AS = 0         #it should be N/A

        CommunicationSystem.__init__(self, system_type, mean_AS, dBS, eps_AS, mu_AS, eps_DS, mu_DS, sigma_SH, r_DS, r_AS)


