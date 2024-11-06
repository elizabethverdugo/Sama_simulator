
class CommunicationSystem:
    def __init__(self, system_type,mean_AS,dBS, eps_AS, mu_AS, eps_DS, mu_DS, sigma_SH, r_DS, r_AS):
        self.system_type = system_type
        self.mean_AS = mean_AS
        self.dBS = dBS
        self.eps_AS = eps_AS
        self.mu_AS = mu_AS
        self.eps_DS = eps_DS
        self.mu_DS = mu_DS
        self.sigma_SH = sigma_SH
        self.r_DS= r_DS
        self.r_AS = r_AS