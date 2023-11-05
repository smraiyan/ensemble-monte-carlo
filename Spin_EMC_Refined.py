from numpy import * 
import scipy as sp 
from matplotlib import pyplot as plt
from numpy import random as RN
from scipy.linalg import expm, norm
from scipy.special import erfinv
from scipy.stats import linregress 
from scipy.integrate import odeint, simps
from scipy.optimize import curve_fit
import time 
from alive_progress import alive_bar as abar

# %% List of Global System Parameters 
global vs, vxyzt, isp, N, lenT, Time, dt, ks, rs, kmeant, rmeant, tpskip 
global e, j, m0, hbar, ep0, g, a, c, kB, sx, sy, sz, Eg, Eso, EgL, Gamma
global ep, m, T, kT, muB, A, b, E, E0, eta, ye, ae, epInf, Bx, Kx, aKf, bKf, cKf
global Ni, Da, aNP, dens, vSnd, f, fk, vsat, ksat, w0, EtaR
global Ndis, Ed, Nc, Gammaflag, fdis, factor, selNdis, selT, selNi

# Temparature, doping and defect density
Ts = array([25, 50, 100, 150, 200, 250, 300])
Nis = array([1e16, 1e17, 4.2e17, 5e17, 1e18, 7.8e18, 1e19, 1e20])*1e6 
Ndiss = array([1e3, 1e4, 5e5, 5e6, 1e7, 1e8, 4e8, 1e9, 1e10, 1e11, 1e12])*1e4 
selT, selNi, selNdis = 4, 0, 1
T, Ni, Ndis =  Ts[selT], Nis[selNi], Ndiss[selNdis]

# %% Defining Global System Parameters
N = int(50000)     # Number of injected Carriers 
lenT = 1000        # Number of Time Points
T = Ts[selT]            # Temperature in K
Ni = Nis[selNi]         # Impurity density (n-type) 
Ndis = Ndiss[selNdis]   # Dislocation density (negatively charged)
fdis = 1.0      # Probablity of filling up a dislocation site with an electron
isp = 1.0       # Initial x-Spin polarization 
tpskip = 10     # How much time points to be skipped   
e = sp.constants.elementary_charge  # Electronic Charge
m0 = sp.constants.electron_mass     # Electron Mass in Free Space
kB = sp.constants.Boltzmann   # Boltzmann Const
hbar = sp.constants.hbar      # Modified Plank's Const
ep0 = sp.constants.epsilon_0  # Permittivity of the Free Space
g = 2                         # Gyromagnetic Ratio 
sx = matrix([ [0, 1], [1, 0 ] ]) 
sy = matrix([ [0, -1j], [1j, 0] ])  
sz = matrix([[ 1, 0],[ 0, -1 ]])   # Pauli Matrices
E = array([100, 0, 0])     # E-field (1 V/cm)
f, fk, kT = e*E/m, e*E/hbar, kB*T

# Material Parameters  
# For GaN (comment while simulating AlN/AlGaN) 
a = 3.186e-10                  # Fundamental Lattice Const
c = 5.186e-10                  # Basal Lattice Const
Eg = (3.427-(((5.31156e-4)*T*T)/(T+992)))*e   # Bandgap 
EgL = 4.9*e                    # B.G at L minima 
Eso = 0.008*e                  # Val band split
ep = 8.9*ep0                   # DC Dielectric Const
epInf = 5.35*ep0               # HF Dieletric Const
m = 0.2*m0                     # Effective mass
muB = e*hbar/(2*m)             # Bohr Magneton 
aNP = (e/Eg)*(1-m/m0)**2       # Non-parabolicity Const
eta = Eso/(Eso+Eg) 
Gamma = 1e15    # Auto-Calibrable Scattering rate               
ye = 0.32*1e-30     # Dresselhaus SOC term
ae = 0.009*e*1e-10  # Rashba SOC term
Bk = 3.9580123
Da = 9.2*e                  # Deformation potential
dens = 6.15e3               # Material denasity (kg/m3) of GaN
vSnd = 7960                 # Sound velocity through GaN
w0 = 91.2*1e-3/hbar         # Optical Phonon Energy hbar*w0
# Boltzmann Dist Const
A = sqrt(m/(2*pi*kT))
b = m/(2*kT) 
Ed = 12e-3*e        # Donor activation energy
Nc = 4.3e14*T**1.5    # Effective DOS in Con band 
Nv = 8.9e15*T**1.5    # Effective DOS in Val band 
EtaR = -E*(e*hbar/m)*pi*Eso*(Eso+2*Eg)/(Eg*(Eg+Eso)*(3*Eg+2*Eso)) 

# For AlN (comment while simulating GaN/AlGaN)
a = 0.31117e-10                # Fundamental Lattice Const
c = 0.49788e-10                # Basal Lattice Const
Eg = (6.292-(((1.799e-3)*T*T)/(T+1462)))*e   # Bandgap 
EgL = 6.9*e                    # B.G at L minima 
Eso = 0.019*e                  # Val band split
ep = 9.14*ep0                  # DC Dielectric Const
epInf = 4.84*ep0               # HF Dieletric Const
m = 0.4*m0                     # Effective mass
muB = e*hbar/(2*m)             # Bohr Magneton 
aNP = (e/Eg)*(1-m/m0)**2       # Non-parabolicity Const
eta = Eso/(Eso+Eg)  
Gamma = 1e14    # Auto-Calibrable Scattering rate              
ye = 2.2591*1e-30       # Dresselhaus SOC term
ae = 0.77011*e*1e-13    # Rashba SOC term
Bk = 12.5725643
Da = 12.7*e                 # Deformation potential
dens = 3.255e3              # Material denasity (kg/m3) of GaN
vSnd = 11270                # Sound velocity through GaN
w0 = 99.2*1e-3/hbar         # Optical Phonon Energy hbar*w0 
# Boltzmann Dist Const
A = sqrt(m/(2*pi*kT))
b = m/(2*kT) 
Ed = 25e-3*e          # Donor activation energy
Nc = 1.2e15*T**1.5    # Effective DOS in Con band 
Nv = 9.4e16*T**1.5    # Effective DOS in Val band 
EtaR = -E*(e*hbar/m)*pi*Eso*(Eso+2*Eg)/(Eg*(Eg+Eso)*(3*Eg+2*Eso)) 

# For AlGaN, Param_AlGaN = x*Param_AlN + (1-x)*Param_GaN 
# (comment while simulating AlN/GaN)
x_al = 0.3 
a = 0.31117e-10*x_al + 3.186e-10*(1-x_al)
c = 0.49788e-10*x_al + 5.186e-10*(1-x_al)             
Eg = x_al*(6.292-(((1.799e-3)*T*T)/(T+1462)))*e + (1-x_al)*(3.427-(((5.31156e-4)*T*T)/(T+992)))*e 
EgL = 6.9*e*x_al + 4.9*e*(1-x_al) 
Eso = 0.019*e*x_al + 0.008*e*(1-x_al) 
ep = 9.14*ep0*x_al + 8.9*ep0*(1-x_al) 
epInf = 4.84*ep0*x_al + 5.35*ep0*(1-x_al) 
m = 1/(x_al/(0.4*m0) + (1-x_al)/(0.2*m0)) 
muB = e*hbar/(2*m)             # Bohr Magneton 
aNP = (e/Eg)*(1-m/m0)**2       # Non-parabolicity Const
eta = Eso/(Eso+Eg)  
Gamma = 1e14*x_al + 1e15*(1-x_al)    # Auto-Calibrable Scattering rate              
ye = 2.2591e-30*x_al + 0.32e-30*(1-x_al)       # Dresselhaus SOC term
ae = 0.77011*e*1e-13*x_al + 0.009*e*1e-13*(1-x_al)    # Rashba SOC term
Bk = 12.5725643*x_al + 3.9580123*(1-x_al)
Da = 12.7*e*x_al + 9.2*e*(1-x_al)                 # Deformation potential
dens = 3.255e3*x_al + 6.15e3*(1-x_al)              # Material denasity (kg/m3) of GaN
vSnd = 11270*x_al + 7960*(1-x_al)                # Sound velocity through GaN
w0 = (99.2*x_al + 91.2*(1-x_al))*1e-3/hbar         # Optical Phonon Energy hbar*w0 
# Boltzmann Dist Const
A = sqrt(m/(2*pi*kT))
b = m/(2*kT) 
Ed = 25e-3*e*x_al + 12e-3*e*(1-x_al)          # Donor activation energy
Nc = (1.2e15*x_al + 4.3e14*(1-x_al))*T**1.5    # Effective DOS in Con band 
Nv = (9.4e16*x_al + 8.9e15*(1-x_al))*T**1.5    # Effective DOS in Val band 
EtaR = -E*(e*hbar/m)*pi*Eso*(Eso+2*Eg)/(Eg*(Eg+Eso)*(3*Eg+2*Eso)) 


# %% Necessary Functions
#---------------------------- Random Number Generator -------------------------------#
def R():
    return RN.rand(1)[0]

#-------------------------------- Material Functions ---------------------------------#
def B(k):
    global aKf, bKf, cKf
    kt = norm(array([k[0],k[1]]))*1e-10 
    return aKf + bKf*kt + cKf*kt*kt

# Screening parameter (1/Screening length)  
def lmbda():
    global e, Ni, ep, kT, Ndis, Nc, kT, c, fdis, Ed
    phid = Nc*exp(-Ed/(2*kT))
    nn = -0.5*(phid + fdis*Ndis/c)+sqrt((0.5*(phid+fdis*Ndis/c))**2+phid*(Ni-fdis*Ndis/c))
    np = nn*(2-nn/Ni)
    return sqrt((e*e*np)/(ep*kT))

# Position of dislocation energy level above valence band 
def trapLevel():
    global fdis, ep, c, Eg, kT, e, Nc, Ed, Ni, Ndis 
    phid = Nc*exp(-Ed/(2*kT))
    nn = -0.5*(phid+fdis*Ndis/c)+sqrt((0.5*(phid+fdis*Ndis/c))**2+phid*(Ni-fdis*Ndis/c))
    Ef = Eg-(kT/e)*log(Nc/nn)
    Eo = e*e/(4*pi*ep*c)
    Ex = 3*fdis*log(fdis/(c*c*c*sqrt(pi*Ni)))
    ET = 0.232*fdis-Ef-Eo*Ex
    return ET*e 

def gamma(k):
    global aNP, e, m, hbar
    knorm = norm(k) 
    Gam = ((hbar*knorm)**2)/(2*m) 
    return e*(-1+sqrt(1+4*Gam*aNP/e))/(2*aNP)

# ---------------- PDF Functions -------------------- #
def POP_Pdf(th,k):
    global aNP, e, m, hbar, w0
    knorm = norm(k)
    Ek0 = hbar*w0/e
    Gam = ((hbar*knorm)**2)/(2*m); 
    Ek = (-1+sqrt(1+4*Gam*aNP/e))/(2*aNP); 
    knum = ((sqrt(1+2*((aNP/e)*(hbar*knorm)**2)/m) + sign(Ek0-Ek)*2*(aNP/e)*hbar*w0)**2) -1
    kden = 2*(aNP/e)*hbar*hbar/m
    kmod = sqrt(knum/kden)
    Gamp = ((hbar*kmod)**2)/(2*m)
    Ekp = (-1+sqrt(1+4*Gamp*aNP/e))/(2*aNP) 
    b = (knorm*knorm+kmod*kmod)/(2*knorm*kmod)
    ak = sqrt((1 + aNP*Ek)/(1+2*aNP*Ek) )
    akp = sqrt((1 + aNP*Ekp)/(1+2*aNP*Ekp) )
    ck = sqrt((aNP*Ek)/(1+2*aNP*Ek) )
    ckp = sqrt((aNP*Ekp)/(1+2*aNP*Ekp) )
    a = (ak*akp)/(ck*ckp)
    ths = linspace(0,2*pi,500)
    P = lambda th: (((a+b)**2)*log((b-cos(th))/(b-1))-4*(sin(th/2)**2)*(a+0.5*b+0.5*(cos(th/2)**2)))                                             
    Pdf = lambda th: P(th)/simps(P(ths),ths)
    return Pdf(th) 

def Dis_Pdf(phi,k):
    b = 2*(norm(k)/lmbda())**2
    a = ((2*b+1)**1.5)/(2*pi*(b+1))
    phis = linspace(0,2*pi,500)
    P = lambda phi: a/(((2*b*(sin(phi))**2)+1)**2)
    Pdf = lambda phi: P(phi)/simps(P(phis), phis)
    return Pdf(phi) 

def WBoltz(k):
    global aNP, m, hbar, kT, N 
    E = gamma(k) 
    return exp(-E/kT)

#---------------------------- Scattering Functions W(k) ------------------------------#
def AcousticW(k):
    global m, kT, Da, dens, vSnd, hbar, aNP, e
    Gam = ((hbar*norm(k))**2)/(2*m)  
    E = gamma(k) 
    Fa = ((1+aNP*E/e)**2+(1/3)*(aNP*E/e)**2)/((1+ 2*aNP*E/e)**2)
    W = ((((2*m)**1.5)*kT*Da*Da)/(2*pi*dens*vSnd*vSnd*((hbar)**4)))*sqrt(Gam)*(1+2*aNP*E/e)*Fa
    return W

def ImpurityW(k):
    global Ni, e, ep, kT, hbar, m, aNP, N 
    Gam = ((hbar*norm(k))**2)/(2*m)
    E = gamma(k) 
    Nk = (m**1.5)*sqrt(Gam)*(1+ 2*aNP*E/e)/(sqrt(2)*pi*pi*(hbar**3))
    lmb = lmbda()
    F = ((1*e*e/ep)**2)*Nk*Ni/(32*hbar*(norm(k)**4))
    c = (lmb/(2*norm(k)))**2
    W = 4*pi*F*(1/c)/(1+c)
    return W

def PolOptPhnW(k):
    global e, m, w0, hbar, epInf, ep, aNP, kT 
    Gam = ((hbar*norm(k))**2)/(2*m)
    E = gamma(k) 
    Nw0 = 1/(exp(hbar*w0/kT)-1)
    # absorb if E<=hbar*w0 else Emission
    if E<=hbar*w0:
        Ep=E+hbar*w0
    else:
        Ep=E-hbar*w0
        Nw0=Nw0 + 1
    Gamp = Ep*(1+aNP*Ep/e)
    A = (2*(1+aNP*E/e)*(1+aNP*Ep/e)+aNP*(Gam+Gamp)/e)**2
    B = (-2*aNP*sqrt(Gam*Gamp)/e)*(4*(1+aNP*E/e)*(1+aNP*Ep/e)+aNP*(Gam+Gamp)/e )
    C = 4*(1+aNP*E/e)*(1+aNP*Ep/e)*(1+ 2*aNP*E/e)*(1+2*aNP*Ep/e)
    F0 = (A*log(abs((sqrt(Gam)+sqrt(Gamp))/(sqrt(Gam)-sqrt(Gamp))))+B)/C
    W = (e*e*sqrt(m)*w0/(4*pi*sqrt(2)*hbar))*(1/epInf-1/ep)*(1+2*aNP*Ep/e)*F0/sqrt(Gam)
    return W

def DislocW(k):
    global hbar, ep, c, Ndis, m, e, fdis
    Ld = 1/lmbda()
    tau = (hbar**3)*((ep*c)**2)*((1+(2*Ld*norm(array([k[0],k[1]])))**2)**1.5)/(Ndis*m*(fdis**2)*(e*Ld)**4) 
    return 1/tau

#---------------------------------- Scattering Update Functions k -----------------------------#
def AcousticUpdate(k):
    global hbar, vSnd, aNP, m, e
    knorm = norm(k)
    th = arccos(1-2*R())
    phi = 2*pi*R()
    qnum = 4*hbar*vSnd*((1+2*(aNP/e)*((hbar*knorm)**2)/m)**0.5)-4*hbar*hbar*knorm*cos(th)/m
    qden = 2*hbar*hbar/m-4*(aNP/e)*((hbar*vSnd)**2)
    q = abs(qnum/qden)
    wq = vSnd*q
    Ekq = hbar*wq/e
    Gam = ((hbar*norm(k))**2)/(2*m)
    Ek = (-1+sqrt(1+ 4*Gam*aNP/e))/(2*aNP)
    kmod = sqrt(knorm**2 + q**2 + sign(Ekq-Ek)*2*knorm*q)
    kp = kmod*array([sin(th)*cos(phi), sin(th)*sin(phi),cos(th)])
    kp = Scat2RealAxis(kp)
    return kp
     
def DislocUpdate(k):
    b = 2*(norm(k)/lmbda())**2
    a = ((2*b+1)**1.5)/(2*pi*(b+1))
    phis = linspace(0, 2*pi, 200)
    P = lambda phi: a/(((2*b*(sin(phi))**2)+1)**2)
    Pdf = lambda phi: P(phi)/simps(P(phis),phis)
    # Von Neumann Method
    while 1:
        R1 = R()*2*pi
        R2 = R()*max(Pdf(phis))
        if R2 <= Pdf(R1):
            phi = R1
            break
    kr = sqrt(k[0]**2+k[1]**2)
    kp = array([kr*cos(phi),kr*sin(phi),k[2]])
    return kp       
    
def ImpurityUpdate(k):
    kmod = norm(k)
    phi = 2*pi*R()
    a = 1+0.5*(lmbda()/kmod)**2 
    Rn = R()
    th = arccos((1+a*(1-2*Rn))/(a+(1-2*Rn)))
    kp = kmod*array([sin(th)*cos(phi),sin(th)*sin(phi),cos(th)])
    kp = Scat2RealAxis(kp)
    return kp    
    
def PolOptPhnUpdate(k):
    global aNP, e, m, hbar, w0
    phi = 2*pi*R()
    knorm = norm(k)
    Ek0 = hbar*w0/e
    Gam = ((hbar*knorm)**2)/(2*m); 
    Ek = (-1+sqrt(1+4*Gam*aNP/e))/(2*aNP); 
    knum = ((sqrt(1+2*((aNP/e)*(hbar*knorm)**2)/m)+sign(Ek0-Ek)*2*(aNP/e)*hbar*w0)**2)-1
    kden = 2*(aNP/e)*hbar*hbar/m
    kmod = sqrt(knum/kden)
    Gamp = ((hbar*kmod)**2)/(2*m)
    Ekp = (-1+sqrt(1+4*Gamp*aNP/e))/(2*aNP) 
    b = (knorm*knorm + kmod*kmod)/(2*knorm*kmod)
    ak = sqrt((1+aNP*Ek)/(1+2*aNP*Ek))
    akp = sqrt((1 + aNP*Ekp)/(1+2*aNP*Ekp))
    ck = sqrt((aNP*Ek)/(1+2*aNP*Ek))
    ckp = sqrt((aNP*Ekp)/(1+2*aNP*Ekp))
    a = (ak*akp)/(ck*ckp)
    ths = linspace(0,pi,200)
    P = lambda th:(((a+b)**2)*log((b-cos(th))/(b-1))-4*(sin(th/2)**2)*(a+0.5*b+0.5*(cos(th/2)**2)))                                             
    Pdf = lambda th: P(th)/simps(P(ths),ths)
    # Von Neumann Method
    while 1:
        R1 = R()*pi
        R2 = R()*max(Pdf(ths))
        if R2 <= Pdf(R1):
            th = R1
            break
    kp = kmod*array([cos(phi)*sin(th),sin(phi)*sin(th),cos(th)])
    kp = Scat2RealAxis(kp)
    return kp    

#------------------- Mapping into real axis from scat-axis ------------------------#
def Scat2RealAxis(k):
    Cx = sqrt(k[1]**2+k[2]**2)
    Cy = sqrt((k[1]**2+k[2]**2)**2+(k[0]*k[1])**2+(k[0]*k[2])**2) 
    Cz = norm(k) 
    T11 = 0
    T12 = (k[1]**2+k[2]**2)/Cy  
    T13 = k[0]/Cz
    T21 = -k[2]/Cx
    T22 = -k[0]*k[1]/Cy        
    T23 = k[1]/Cz
    T31 = k[1]/Cx
    T32 = -k[0]*k[2]/Cy        
    T33 = k[2]/Cz
    T = array([[T11, T12, T13],[T21, T22, T23],[T31, T32, T33] ])
    return matmul(T,k)

# --------------------------- Selection of Scattering Mechanisms ------------------------------- #
# Enumeration of Scattering Mechanisms goes like the following.        
# (0) Acoustic Phonon, (1) Polar Optical phonon  , (2) Impurity
# (3) Dislocation, (4) Self Scattering. 
# Random Selection of Scat-Mechanism using Monte Carlo Algo    
def ScatSelect(k):
    global Gamma, Gammaflag
    W = zeros(5)
    W[0] = AcousticW(k)
    W[1] = PolOptPhnW(k)
    W[2] = ImpurityW(k)
    W[3] = DislocW(k)
    W[4] = Gamma - sum(W[0:4])
    if W[4]<=0:
        print('Recalibrating Gamma at least greater than ', sum(W[0:4]))
        Gamma = 1.1*sum(W[0:4])       
        Gammaflag = 0
    Wc = cumsum(W)
    GamR = R()*Gamma
    iDcheck = where((Wc-GamR)>=0)[0]
    if not sum(iDcheck):
        iD = len(Wc)-1
    else:
        iD = iDcheck[0]
    return iD

# Updating k using the selected mechanisms
def ScatUpdate(k, iD): 
    if (iD==0):
        return AcousticUpdate(k)
    elif (iD==1):
        return PolOptPhnUpdate(k)
    elif (iD==2):
        return ImpurityUpdate(k)
    elif (iD==3):
        return DislocUpdate(k)
    elif (iD==4):
        return k 

# Ensemble Scattering update of k        
def ScatCarr():
    global ks, N
    for i in range(N):
        iD = ScatSelect(ks[i,:])
        ks[i,:] = ScatUpdate(ks[i,:], iD)

# ------------------------- Mean Functions -------------------------------# 
def MeanK(k, NumComp):
    global N
    if NumComp == 0:
        sumk, wgt = 0, 0
        for i in range(N):
            sumk += k[i,0]*WBoltz(k[i,0])
            wgt += WBoltz(k[i,0])
        return sumk/wgt
    elif NumComp == 1:
        sumk, wgt = 0, 0
        for i in range(N):
            sumk += k[i,1]*WBoltz(k[i,1])
            wgt += WBoltz(k[i,1])
        return sumk/wgt
    elif NumComp == 2:
        sumk, wgt = 0, 0
        for i in range(N):
            sumk += k[i,2]*WBoltz(k[i,2])
            wgt += WBoltz(k[i,2])
        return sumk/wgt
    elif NumComp == 210:
        sumk, wgt = 0, 0
        for i in range(N):
            sumk += norm(k[i,:])*WBoltz(k[i,:])
            wgt += WBoltz(k[i,:])
        return sumk/wgt
    else:
        return mean(k)

def MeanR(r):
    global N
    sumR = sum(r, axis=0)
    return sumR/N 

# -------------------------- Fit Functions ---------------------- #
def ExpFitT(x, y, tf):
    global selT, selNdis, selNi
    ival = y[0]
    def FitFunc(t, yinf, tau):
        return yinf + (ival-yinf)*exp(-t/tau)
    yinf = y[len(y)-1]
    tau = tf        # Initial Guess
    YinfTau, CovMat = curve_fit(FitFunc,x,y,p0=(yinf, tau))
    yinf, tau = YinfTau
    xx, yy = x, FitFunc(xx, yinf, tau)
    RMSE = sqrt(mean(abs(y-yy)))
    return tau, RMSE

def ExpFitR(x, y, xf):
    global selT, selNdis, selNi
    ival = y[0]
    def FitFunc(t, yinf, tau):
        return yinf + (ival-yinf)*exp(-t/tau)
    yinf = y[len(y)-1]
    tau = xf              # Initial Guess
    YinfTau, CovMat = curve_fit(FitFunc,x,y,p0=(yinf, tau))
    yinf, tau = YinfTau
    xx, yy = x, FitFunc(xx, yinf, tau)
    RMSE = sqrt(mean(abs(y-yy)))
    return tau, RMSE

def SinFitT(x, y, tf):
    global selT, selNdis, selNi 
    def FitFunc(t, A, tau, w, delta):
        return A*exp(-t/tau)*sin(w*t + delta) 
    A = max(abs(y))
    tau, w, delta = tf, 1/tau, pi/2
    ATauWDelta, CovMat = curve_fit(FitFunc, x, y, p0 = (A, tau, w, delta))
    A, tau, w, delta = ATauWDelta
    xx, yy = x, FitFunc(xx, A, tau, w, delta) 
    RMSE = sqrt(mean(abs(y-yy)))
    return tau, RMSE

#------------------------------ Time Vector Set-up ----------------------------------#
def TimeSet():
    global lenT, dt, Time, Gamma
    Time = zeros(lenT)
    dt = zeros(lenT-1)
    for i in range(len(dt)):
        dt[i] = -log(1-R())/Gamma;
    Time[1:len(Time)] = cumsum(dt)

# ---------------------------- Carrier Initialization --------------------------------#  
def CarInit():
    global rs, ks, vs, vxyzt, N, lenT, A, b, m, kT, hbar, isp, kmeant, rmeant, Gammaflag, tpskip 
    Gammaflag = 1
    rs, ks, kmeant, rmeant = zeros([N,3]), zeros([N,3]), zeros(lenT), zeros([lenT,3]) 
    vs = zeros([N, 3])
    vs[:,0] = isp*ones(N)
    vxyzt = zeros([lenT, 3])
    vxyzt[0:tpskip+1,:] = isp*array([1,0,0])
    for i in range(N):
        ux =  sqrt( -2*kT*log( 1-R() ) / m )
        uy = (1/sqrt(b))*erfinv( (2*R()/A)*sqrt(b/pi) - 1 )
        uz = (1/sqrt(b))*erfinv( (2*R()/A)*sqrt(b/pi) - 1 )
        ks[i,:] = array([ux, uy, uz])*m/hbar
    kmeant[0] = MeanK(ks, 210)  

# ------------------------------- Newtonian Update -----------------------------------# 
def UpdateK(t):
    global N, ks, rs, f, fk, hbar, m
    for i in range(N):
        u = hbar*ks[i,:]/m 
        delk = fk*t
        ks[i,:] = ks[i,:] + delk
        delr = u*t + 0.5*f*t**2
        rs[i,:] = rs[i,:] + delr

# ----------------------------- Spin Polarization Updater -----------------------------#
# -------------------- Solution of Coupled eq --> dS/dt = OMG X S -------------------- #
def soc_function(ks):
    global BetaL, aNP, hbar, m, eta, Eg, vs, vx, vy, vz, vL, N, lenT, sx, sy, sz, kT, dt, e, factor, g, ae, ye, EtaR
    OMGD = (g*ye/hbar)*(Bk*ks[2]*ks[2]-ks[0]*ks[0]-ks[1]*ks[1])*array([ks[1],-ks[0],0]) 
    OMGR = (g*ae/hbar)*array([ks[1],-ks[0],0])
    OMGE = 2*cross(EtaR,ks)
    return OMGR+OMGD+OMGE

def SpinFinder(tp):
    global ks, BetaL, aNP, hbar, m, eta, Eg, vs, vx, vy, vz, vL, N, lenT, sx, sy, sz, kT, dt, e, factor, g, ae, ye, EtaR
    W, vavg = zeros(N), zeros(3) 
    t = dt[tp]
    for i in range(N):
        OMG = soc_function(ks[i,:])  
        def f(v,t):
            f1 = OMG[2]*v[1]-OMG[1]*v[0]
            f2 = -OMG[2]*v[0]+OMG[0]*v[2]
            f3 = OMG[1]*v[0]-OMG[0]*v[1]
            return [f1,f2,f3]
        t_slv = linspace(0,t,500) 
        vs[i,:] = odeint(f,vs[i,:],t_slv)[len(t_slv)-1,:]
        Gam = ((hbar*norm(ks[i,:]))**2)/(2*m)
        En = e*(-1+sqrt(1+4*Gam*aNP/e))/(2*aNP)
        W[i] = exp(-En/kT)
        vavg += vs[i,:]*W[i]
    vavg = vavg/sum(W)
    vxyzt[tp+1,:] = vavg 


# %% Flight Scatter Loop
t1 = time.time()  
while 1: 
    tpcount = 0                # tpskip counter 
    TimeSet()                  # Defining Time Grid
    CarInit()                 # Initialize k and spin of the injected Carriers 
    with abar(lenT-1, title='Progress') as bar: 
        for tp in range(len(dt)):        
            UpdateK(dt[tp])        # Free flight 
            ScatCarr()             # Random Scattering of Carriers
            if not Gammaflag:      # Chaching if any W(k) exceeds Gamma
                break              # if exceeds, break the loop and re-define Gamma 
            kmeant[tp+1] = MeanK(ks, 0)  # Storing avg k at the runtime
            rmeant[tp+1,:] = MeanR(rs) 
            if tpcount>=tpskip:
                SpinFinder(tp)           # Updates vx, vy, vz, vL at the runtime
            tpcount += 1                 # increment of tpskip counter 
            bar()                        # showing progress
        if Gammaflag:        # If Gamma is above sum(W), break the inifinity loop
            break            # else restart the simulation with the redefined Gamma 
t2 = time.time() 
print('Time Elapsed',(t2-t1),'seconds')

# %% Retrieve tau 
t = Time[tpskip:len(Time)]
t = t-t[0] 

# Spin Relaxation Time
taux, rmsex = ExpFitT(t,vxyzt[tpskip:len(Time),0],1e-12)                   
print('Spin Ralaxation Time = ',taux*1e12, ' ps') 

# Momentum
tauk, rmsek = ExpFitT(t,kmeant,1e-14)   # Momentum Relaxation 
meuk = tauk*e/m
print('Simulation Mobiity = ', meuk*1e4, ' sq-cm/V')

# Spin Diffusion Length
tau_rx, rmse_rx = ExpFitR(array(rmeant)[:,2][tpskip:],vxyzt[tpskip:len(Time),0],1e-6)
print('Spin Diffusion Length, Ds =', tau_rx*1e6, ' um') 
