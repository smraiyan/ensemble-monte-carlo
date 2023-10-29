import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from sklearn.preprocessing import Normalizer 

e = sp.constants.elementary_charge 
eps0 = sp.constant.epsilon_0 
gam, beta = 0.05, 6e-4 
P_spin, S0 = 1,1  
i_pump = 4e-3 
l, w = 50e-9, 150e-9 
Vb = np.pi*w**2*l/4 
Ni = 1e16

# For GaN
tau_cap = 38e-12
tau_sb = 384.525e-12
tau_s = 77.778e-12 
eps = 8.9*eps0 
B_sp = 1.1e-8
C = 3e-28
tau_ph = 1.8e-12
vg = 0.87e10
n_tr = 4e17     
dg_dn = 3.4e-11 
gain = lambda n,S: dg_dn*(n-n_tr)/(1+eps*S) 

# For AlN
tau_cap = 27e-12
tau_sb = 384.525e-12
tau_s = 889.827e-12 
eps = 9.4*eps0 
B_sp = 4e-11
C = 2.1e-29
tau_ph = 3.2e-12
vg = 0.79e10
n_tr = 7e18     
dg_dn = 1.1e-12 
gain = lambda n,S: dg_dn*(n-n_tr)/(1+eps*S) 

# AlGaN with x = 0.3
x_Al = 0.3
tau_cap = 27e-12*x_Al+38e-12*(1-x_Al) 
tau_sb = 384.525e-12
tau_s = 202.117e-12 
eps = (9.4*x_Al+8.9*(1-x_Al))*eps0 
B_sp = 4e-11*x_Al+1.1e-8*(1-x_Al)
C = 2.1e-29*x_Al+3e-28*(1-x_Al)
tau_ph = 3.2e-12*x_Al+1.8e-122*(1-x_Al)
vg = 0.79e10*x_Al+0.87e10*(1-x_Al)
n_tr = 7e18*x_Al+4e17*(1-x_Al)     
dg_dn = 1.1e-12*x_Al+3.4e-11*(1-x_Al)
gain = lambda n,S: dg_dn*(n-n_tr)/(1+eps*S) 

# For single pump current
i_pump = 10e-3
def F(x,t):
    Dx0 = - x[0]/tau_cap + ((1 + P_spin)/2) * i_pump/(e*Vb) - (x[0] - x[1])/tau_sb
    Dx1 = - x[1]/tau_cap + ((1 - P_spin)/2) * i_pump/(e*Vb) + (x[0] - x[1])/tau_sb
    Dx2 = (1/Vb)*x[0]/tau_cap - vg*gain(x[2],x[5])*x[5] - (x[2] - x[3])/tau_s - B_sp*x[2]*(x[2] + x[3])/2 - C*x[2]*((x[2] + x[3])**2)/2
    Dx3 = (1/Vb)*x[1]/tau_cap - vg*gain(x[3],x[4])*x[4] + (x[2] - x[3])/tau_s - B_sp*x[3]*(x[2] + x[3])/2 - C*x[3]*((x[2] + x[3])**2)/2
    Dx4 = gam*vg*gain(x[3],x[4])*x[4] + gam*beta*B_sp*x[3]*(x[2] + x[3])/2 - x[4]/tau_ph 
    Dx5 = gam*vg*gain(x[2],x[5])*x[5] + gam*beta*B_sp*x[2]*(x[2] + x[3])/2 - x[5]/tau_ph 
    return [Dx0, Dx1, Dx2, Dx3, Dx4, Dx5]

# x = [nb_plus, nb_minus, n_plus, n_minus, S_plus, S_minus] 
x_init = [Ni*P_spin,Ni*(1-P_spin),Ni*P_spin,Ni*(1-P_spin),S0/2,S0/2] 
time = np.linspace(0,5e-8,500)   
x_sol = odeint(F,x_init,time) 
Norm = Normalizer('l2')   
x_sol_norm = Norm.transform(x_sol)

intensity = x_sol_norm[:,4] + x_sol_norm[:,5]
PolB = (x_sol_norm[:,0] - x_sol_norm[:,1])/(x_sol_norm[:,0] + x_sol_norm[:,1])
Pol  = (x_sol_norm[:,2] - x_sol_norm[:,3])/(x_sol_norm[:,2] + x_sol_norm[:,3])
PolL = (x_sol_norm[:,4] - x_sol_norm[:,5])/(x_sol_norm[:,4] + x_sol_norm[:,5]) 
print(PolB[-1], Pol[-1], PolL[-1], intensity[-1]) 

# For multiple pump currents
i_pumps = np.linspace(0.5, 25, 25)*1e-3  
x_init = [Nab*P_spin, Nab*(1-P_spin), Na*P_spin, Na*(1-P_spin), S0/2, S0/2] 
intensity = np.zeros(len(i_pumps)) 
intp, intm = np.zeros(len(i_pumps)), np.zeros(len(i_pumps)) 
pols = np.zeros(len(i_pumps))

for i_pump in i_pumps: 
    def F(x,t): 
        Dx0 =  - x[0]/tau_cap + ((1 + P_spin)/2) * i_pump/(e*Vb) - (x[0] - x[1])/tau_sb
        Dx1 =  - x[1]/tau_cap + ((1 - P_spin)/2) * i_pump/(e*Vb) + (x[0] - x[1])/tau_sb
        Dx2 = (Vb/Vmqw)*x[0]/tau_cap - vg*gain(x[2],x[5])*x[5] - (x[2] - x[3])/tau_s - B_sp*x[2]*(x[2] + x[3])/2 - C*x[2]*((x[2] + x[3])**2)/2
        Dx3 = (Vb/Vmqw)*x[1]/tau_cap - vg*gain(x[3],x[4])*x[4] + (x[2] - x[3])/tau_s - B_sp*x[3]*(x[2] + x[3])/2 - C*x[3]*((x[2] + x[3])**2)/2
        Dx4 = gam*vg*gain(x[3],x[4])*x[4] + gam*beta*B_sp*x[3]*(x[2] + x[3])/2 - x[4]/tau_ph 
        Dx5 = gam*vg*gain(x[2],x[5])*x[5] + gam*beta*B_sp*x[2]*(x[2] + x[3])/2 - x[5]/tau_ph 
        return [Dx0, Dx1, Dx2, Dx3, Dx4, Dx5]
    x_sol = odeint(F, x_init, time) 
    intp[i_pump==i_pumps], intm[i_pump==i_pumps] = x_sol[:,4][-1], x_sol[:,5][-1] 
    intensity[i_pump==i_pumps] = x_sol[:,4][-1] + x_sol[:,5][-1] 
    pols[i_pump == i_pumps] = (x_sol[:,2][-1] - x_sol[:,3][-1])/(x_sol[:,2][-1] + x_sol[:,3][-1]) 
print(intensity, intp, intm, pols)