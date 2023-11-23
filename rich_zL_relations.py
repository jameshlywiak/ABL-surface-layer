import numpy as np

gravity = 9.81

def zeta_from_rb_GF96(rb, rbc=-4.5, z=0, C=10, zi=600, beta=1.25, Ch=3e-3):
    ## from Grachev and Fairall 1996
    ## COARE data

    if rbc == 0:
        rbc = - z/(zi*Ch*beta**3)

    zeta = C*rb / (1 + rb/rbc)
    
    return zeta

def zeta_from_rb(rb, alpha=1, betam=5, betah=5):
    ## from England and McNider 1995
    ## need to add unstable but it's a mess
    zeta = np.ones(rb.shape)

    if betam==betah and alpha=1:
        zeta[rb>0] = rb[rb>0] / (1 - betam*rb[rb>0])
    else:
        zeta[rb>0] = (alpha - 2*betam*rb[rb>0]*np.sqrt(alpha**2 + 
                                4*(betah - alpha*betam)*rb[rb>0])) / (
                        2*(betam**2*rb[rb>0] - betah) )
    
    return zeta

def rb_shear(phim, phih, zeta):
    
    rb = zeta*(phih/(phim**2))

    return rb