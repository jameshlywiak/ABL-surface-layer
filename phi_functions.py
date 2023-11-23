import numpy as np

gravity = 9.81

def Vickers_Mahrt99_phim(zeta):
    ## Requirements:
    ##  Advection is minimal
    ##  Observation height sufficiently below IBL height
    ## von karman = 0.39

    phim = np.ones(zeta.shape)
    
    phim[zeta>0] = (1 + 16*zeta)**(1/3)
    phim[zeta<0] = (1 - 35*zeta)**(-1/4)

    return phim

def Dyer74_phim(zeta):
    phim = np.ones(zeta.shape)
    
    phim[zeta>0] = 1 + 5*zeta
    phim[zeta<0] = (1 - 16*zeta)**(-1/4)

    return phim

def Beljaars_Holtslag1991_phim(zeta, a=1, b=0.667, c=5, d=0.35):
    phim = np.ones(zeta.shape)
    
    phim[zeta>0] = 1 + zeta*(a + b*np.exp(-d*zeta)*(1 + c - d*zeta))
    phim[zeta<0] = (1 - 16*zeta)**(-1/4)

    return phim

def SHEBA_phim(zeta, a=5, b=5/6.5):
    ## SHEBA applies to stable ASL over ice
    
    phim = np.ones(zeta.shape)
    
    phim[zeta>0] = 1 + (a*zeta*(1 + zeta)**(1/3)) / (1 + b*zeta)

    return phim

def SHEBA_phih(zeta, a=5, b=5, c=3):
    ## SHEBA applies to stable ASL over ice
    
    phih = np.ones(zeta.shape)
    
    phih[zeta>0] = 1 + (a*zeta + b*zeta**2) / (1 + c*zeta + zeta**2)

    return phih