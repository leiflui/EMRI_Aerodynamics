import numpy as np
from kerrgeopy.spacetime import *
from scipy.special import ellipk, ellipe, elliprj, elliprf, elliprd, ellipj
from .frequencies import _ellippiinc, precession_frequency, mino_frequencies
from .constants import *
from .stable import *

def Kerr_riemann(M, a, r, theta):
    """
    Compute the Riemann tensor for Kerr spacetime at a given point.
    
    Parameters:
        M (float): Black hole mass
        a (float): Black hole spin parameter
        r (float): Radial coordinate
        theta (float): Polar angle coordinate
        
    Returns:
        R (np.ndarray): Riemann tensor components R**ρ_[σμν] (4x4x4x4 array)
    """
    
    # Riemann tensor components
    R = np.zeros((4, 4, 4, 4))

    cot_theta = 1/np.tan(theta)
    
    # Independent non-zero component
    R[0, 0, 0, 3] = -((16*a*M**2*r**2*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[0, 0, 1, 2] = (4*a**2*M**2*r*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3)
    R[0, 1, 0, 1] = (M*r*(-27*a**4 - 4*a**2*r**2 + 16*r**4 - 4*a**2*(6*a**2 + 7*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/((a**2 + r*(-2*M + r))*(a**2 
                    + 2*r**2  + a**2*np.cos(2*theta))**3)
    R[0, 1, 0, 2] = (2*a**2*M*(3*a**2 + r*(-2*M + 3*r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[0, 1, 1, 3] = -((12*a*M*r*(a**2 + r**2)*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3))
    R[0, 1, 2, 3] = -((a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(-5*a**4 + a**2*(2*M - 9*r)*r - 4*r**4 + a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(2*theta)))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3)
    R[0, 2, 0, 2] = -((2*M*r*(-9*a**4 + 2*a**2*r**2 + 4*r**4 - 2*a**2*(3*a**2 + 5*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[0, 2, 1, 3] = -((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(-2*a**4 + a**2*(2*M - 3*r)*r - r**4 + a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[0, 2, 2, 3] = (12*a*M*r*(a**2 + r**2)*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3
    R[0, 3, 0, 3] = -((4*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*(-a**4 - 2*r**4 - a**2*r*(2*M + 3*r) - a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[1, 0, 0, 1] = -((4*M*r*(-3*a**2 + 2*r**2 - 3*a**2*np.cos(2*theta))*(-5*a**2 + 8*M*r - 4*r**2 + a**2*np.cos(2*theta)))/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**4)
    R[1, 0, 0, 2] = (12*a**2*M*(a**2 + r*(-2*M + r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[1, 0, 1, 3] = -((8*a*M*r*(3*a**2 + r*(-4*M + 3*r))*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[1, 0, 2, 3] = (a*M*(a**2 + r*(-2*M + r))*(9*a**4 - 52*a**2*r**2 - 48*r**4 + 4*a**2*(2*a**2 + 5*r**2)*np.cos(2*theta) 
                    - a**4*np.cos(4*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[1, 2, 0, 3] = (2*a*M*(a**2 + r*(-2*M + r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3
    R[1, 2, 1, 2] = (M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/(2*(r**2 + a**2*np.cos(theta)**2)**2)
    R[1, 3, 1, 3] = (8*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*(2*a**4 + r**4 + a**2*r*(-2*M + 3*r) - a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[1, 3, 2, 3] = -((24*a**2*M*(a**2 + r**2)*(a**2 + r*(-2*M + r))*np.cos(theta)*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(theta)**3)/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**4)
    R[2, 0, 0, 1] = (12*a**2*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[2, 0, 0, 2] = (8*M*r*(-3*a**2 + 2*r**2 - 3*a**2*np.cos(2*theta))*(-2*a**2 + (2*M - r)*r + a**2*np.cos(2*theta)))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[2, 0, 1, 3] = -((2*a*M*(-3*a**4 + 22*a**2*r**2 + 12*r**4 - 2*a**2*(a**2 + 7*r**2)*np.cos(2*theta) + a**4*np.cos(4*theta))*np.sin(2*theta))/(a**2 
                    + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[2, 0, 2, 3] = (8*a*M*r*(3*a**2 + r*(-2*M + 3*r))*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[2, 1, 0, 3] =-((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3)
    R[2, 1, 1, 2] = -((M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/(2*(a**2 + r*(-2*M + r))*(r**2 + a**2*np.cos(theta)**2)**2))
    R[2, 3, 0, 2] = -((8*a*M*r*(3*a**2 + r*(-2*M + 3*r))*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[2, 3, 1, 3] = -((24*a**2*M*(a**2 + r**2)*np.cos(theta)*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(theta)**3)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[2, 3, 2, 3] = -((4*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*(5*a**4 + 4*r**4 + a**2*r*(-2*M + 9*r) - a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[3, 0, 0, 3] = (4*M*r*(-3*a**2 + 2*r**2 - 3*a**2*np.cos(2*theta))*(-a**2 + 4*M*r - 2*r**2 - a**2*np.cos(2*theta)))/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**4
    R[3, 0, 1, 2] = -((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(a**2 + 2*r*(-2*M + r) + a**2*np.cos(2*theta))*cot_theta )/((a**2 + r*(-2*M + r))*(a**2 
                    + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[3, 1, 0, 1] = -((12*a*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[3, 1, 0, 2] = -((4*a*M*(-2*a**2 + (2*M - r)*r + a**2*np.cos(2*theta))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*cot_theta )/((a**2 + r*(-2*M + r))*(a**2 
                    + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[3, 1, 1, 3] = (2*M*r*(-9*a**4 + 2*a**2*r**2 + 4*r**4 - 2*a**2*(3*a**2 + 5*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/((a**2 + r*(-2*M + r))*(a**2
                    + 2*r**2 + a**2*np.cos(2*theta))**3)
    R[3, 1, 2, 3] = (2*a**2*M*(3*a**2 + r*(-2*M + 3*r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[3, 2, 0, 1] = -((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(-5*a**2 + 8*M*r - 4*r**2 + a**2*np.cos(2*theta))*cot_theta )/((a**2 + r*(-2*M + r))*(a**2 
                    + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[3, 2, 0, 2] = (12*a*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3
    R[3, 2, 1, 3] = (2*a**2*M*(3*a**2 + r*(-4*M + 3*r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[3, 2, 2, 3] = -((M*r*(-27*a**4 - 4*a**2*r**2 + 16*r**4 - 4*a**2*(6*a**2 + 7*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[3, 3, 0, 3] = (16*a*M**2*r**2*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    

    #Non-zero components by symmetry of Riemann tensor
    R[0, 0, 2, 1] = - R[0, 0, 1, 2]
    R[0, 0, 3, 0] = - R[0, 0, 0, 3]
    R[0, 1, 1, 0] = - R[0, 1, 0, 1]
    R[0, 1, 2, 0] = - R[0, 1, 0, 2]  
    R[0, 1, 3, 1] = - R[0, 1, 1, 3] 
    R[0, 1, 3, 2] = - R[0, 1, 2, 3]
    R[0, 2, 0, 1] = - R[0, 0, 1, 2] - R[0, 1, 2, 0]
    R[0, 2, 1, 0] = - R[0, 2, 0, 1]
    R[0, 2, 2, 0] = - R[0, 2, 0, 2]
    R[0, 2, 3, 1] = - R[0, 2, 1, 3]
    R[0, 2, 3, 2] = - R[0, 2, 2, 3]
    R[0, 3, 1, 2] = - R[0, 1, 2, 3] - R[0, 2, 3, 1]
    R[0, 3, 2, 1] = - R[0, 3, 1, 2]
    R[0, 3, 3, 0] = - R[0, 3, 0, 3]
    R[1, 0, 1, 0] = - R[1, 0, 0, 1]
    R[1, 0, 2, 0] = - R[1, 0, 0, 2]
    R[1, 0, 3, 1] = - R[1, 0, 1, 3] 
    R[1, 0, 3, 2] = - R[1, 0, 2, 3] 
    R[1, 2, 2, 1] = - R[1, 2, 1, 2]  
    R[1, 2, 3, 0] = - R[1, 2, 0, 3]
    R[1, 3, 0, 1] = - R[1, 0, 1, 3] - R[1, 1, 3, 0]
    R[1, 3, 0, 2] = - R[1, 0, 2, 3] - R[1, 2, 3, 0]
    R[1, 3, 1, 0] = - R[1, 3, 0, 1]
    R[1, 3, 2, 0] = - R[1, 3, 0, 2] 
    R[1, 3, 3, 1] = - R[1, 3, 1, 3]  
    R[1, 3, 3, 2] = - R[1, 3, 2, 3]
    R[2, 0, 1, 0] = - R[2, 0, 0, 1]
    R[2, 0, 2, 0] = - R[2, 0, 0, 2]
    R[2, 0, 3, 1] = - R[2, 0, 1, 3]
    R[2, 0, 3, 2] = - R[2, 0, 2, 3]
    R[2, 1, 2, 1] = - R[2, 1, 1, 2]
    R[2, 1, 3, 0] = - R[2, 1, 0, 3]
    R[2, 3, 0, 1] = - R[2, 0, 1, 3] - R[2, 1, 3, 0]
    R[2, 3, 1, 0] = - R[2, 3, 0, 1]
    R[2, 3, 2, 0] = - R[2, 3, 0, 2]
    R[2, 3, 2, 0] = - R[2, 2, 0, 3] - R[2, 0, 3, 2]
    R[2, 3, 3, 1] = - R[2, 3, 1, 3]
    R[2, 3, 3, 2] = - R[2, 3, 2, 3]
    R[3, 0, 2, 1] = - R[3, 0, 1, 2] 
    R[3, 0, 3, 0] = - R[3, 0, 0, 3]
    R[3, 1, 1, 0] = - R[3, 1, 0, 1] 
    R[3, 1, 2, 0] = - R[3, 1, 0, 2]
    R[3, 1, 3, 1] = - R[3, 1, 1, 3]
    R[3, 1, 3, 2] = - R[3, 1, 2, 3]
    R[3, 2, 1, 0] = - R[3, 2, 0, 1] 
    R[3, 2, 2, 0] = - R[3, 2, 0, 2]
    R[3, 2, 3, 1] = - R[3, 2, 1, 3]
    R[3, 2, 3, 2] = - R[3, 2, 2, 3]
    R[3, 3, 1, 2] = - R[3, 1, 2, 3] - R[3, 2, 3, 1] 
    R[3, 3, 2, 1] = - R[3, 3, 1, 2]
    R[3, 3, 3, 0] = - R[3, 3, 0, 3] - R[3, 0, 3, 3]    
    return R



def psi_phase(lam, a, p, e, x, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
    """
    Compute the phase ψ(λ) as per Eq. (56)-(61) of the paper.
    
    Parameters:
        lam (float): Mino time λ
        a (float): Black hole spin
        p (float): semi-latus rectum
        e (float): orbital eccentricity
        x (float): inclination parameter
        qr0, qz0, qpsi0 (float): Initial phases (default 0)
        upsilon_r, upsilon_z, upsilon_psi (float): Frequencies (optional)
        
    Returns:
        float: Phase ψ(λ)
    """
    # Compute E, L, Q from orbital parameters
    E, L, Q = constants_of_motion(a, p, e, x)
    
    # Compute K = K (Eq. 49)
    K = Q + (a * E - L)**2
    sqrtK = np.sqrt(K)

    #Calculate the radial and polar roots
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x)
    zm, zp = stable_polar_roots(a, p, e, x)
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)

    # Compute moduli k_r and k_z (Eq. 18, 19)
    k_r = np.sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))
    k_z = np.sqrt(a**2 * (1 - E**2) * z1**2 / z2**2)
    m_r = k_r**2  # Scipy uses m = k**2
    m_z = k_z**2

    # Compute h_r and h_ψ (Eq. 34, 61)
    h_r = (r1 - r2) / (r1 - r3)
    h_psi = h_r * (r3 - 1j * sqrtK) / (r2 - 1j * sqrtK)

    #Calculate the precession frequency
    Upsilon_psi_r, Upsilon_psi_z = precession_frequency(a, p, e, x)
    Upsilon_psi = Upsilon_psi_r + Upsilon_psi_z
    q_psi = Upsilon_psi*lam + qpsi0
    psi = q_psi

    #Using these complementary function, we can now define the other parts of the phases
    Upsilon_r, Upsilon_theta, Upsilon_phi, Gamma = mino_frequencies(a, p, e, x)
    qr = Upsilon_r*lam + qr0
    qtheta = Upsilon_theta*lam 
    qz = qtheta + qz0
    #Here, we define some complementary function
    sn_r, cn_r, dn_r, ph_r = ellipj(qr/np.pi*ellipk(k_r), k_r)
    sn_z, cn_z, dn_z, ph_z = ellipj(2*qz/np.pi*ellipk(k_z), k_z)
    xi_r = ph_r
    xi_z = ph_z
    def psi_r_tilde(a, p, e, x, xi):
        psi_r_tilde = 2*(r2 - r3)*((K - a**2)*E+a*L)/((K + r2**2)*(K + r3**2)*np.sqrt((1 - E**2)*(r1 - r3)*(r2 - r4)))*np.imag((r2 + 1j*sqrtK)*(r3 
                        + 1j*sqrtK)*_ellippiinc(xi, h_psi, k_r))    
        return psi_r_tilde
    
    def psi_z_tilde(a, p, e, x, xi):
        psi_z_tilde = -((a**2 - K)*E - a*L)/(sqrtK*z2*_ellippiinc(xi, a**2*z1**2/K, k_z)) 
        return psi_z_tilde
    psi_r = psi_r_tilde(a, p, e, x, xi_r) - psi_r_tilde(a, p, e, x, np.pi)/(2*np.pi)*qr
    psi_z = psi_z_tilde(a, p, e, x, xi_z) - psi_z_tilde(a, p, e, x, np.pi)/np.pi*qr
    psi += psi_r + psi_z
    return psi

def Marck_tetrad(lam, a, p, e, x, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
    """
    Compute the tetrad as per Eq. (56)-(61) of van de Meent 2021.
    
    Parameters:
        lam (float): Mino time λ
        a (float): Black hole spin
        p (float): semi-latus rectum
        e (float): orbital eccentricity
        psi (float): phase
        x (float): inclination parameter
        qr0, qz0, qpsi0 (float): Initial phases (default 0)
        upsilon_r, upsilon_z, upsilon_psi (float): Frequencies (optional)
        
    Returns:
        np.array: e0, e1, e2, e3 (Marck tetrads)
    """
    # Compute E, L, Q from orbital parameters
    constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants

    # Compute K = K (Eq. 49)
    K = Q + (a * E - L)**2
    sqrtK = np.sqrt(K)

    #Calculate the radial and polar roots
    radial_roots = stable_radial_roots(a, p, e, x)
    r1, r2, r3, r4 = radial_roots
    polar_roots = stable_polar_roots(a, p, e, x)
    zm, zp = polar_roots
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)

    #Compute the Mino frequencies
    Upsilon_r, Upsilon_theta, Upsilon_phi, Gamma = mino_frequencies(a, p, e, x)

    #Calculate the radial and polar solutions
    r, t_r, phi_r = radial_solutions(a, constants, radial_roots)
    qr = Upsilon_r*lam + qr0
    r = r(qr0)
    theta, t_theta, phi_theta = polar_solutions(a, constants, polar_roots)
    qz = Upsilon_theta*lam + qz0 
    theta = theta(qz0)
    z = np.cos(theta)
    y = np.sqrt(1 - z**2)

    # Define some parameters for Kerr geodesics
    Delta = r**2 - 2*r + a**2
    Xi = np.sqrt((K - a**2*z**2)/(K + r**2))
    Sigma = r**2 + a**2*z**2

    # Define the radial and polar potentials 
    R = (E*(r**2 + a**2) - a*L)**2 - Delta*(r**2 + (a*E - L)**2 + Q)
    Z = Q - (Q + a**2*(1 - E**2) + L**2)*z**2 + a**2*(1 - E**2)*z**4
    
    # Determine signs based on phase (critical for orthogonality)
    r_dot = np.sqrt(R)
    z_dot = np.sqrt(Z)
    theta_dot = z_dot/y

    #Define the Marck tetrads
    e0 = np.array([-E, r_dot/Delta, theta_dot, L])
    e1_tilde = 1/sqrtK*np.array([-((r*r_dot*Xi + a**2/Xi*theta_dot*z*y)/Sigma), (r*((a**2 + r**2)*E - a*L)*Xi)/Delta, a/Xi*z*((-L)/y + a*E*y), (a*y*((a**2 
        + r**2)/Xi*theta_dot*z + r*r_dot*Xi*y))/Sigma])
    e2_tilde = np.array([(Xi*E*(r**2+a**2)+a*L*(1/Xi-Xi)-a**2*E*1/Xi*y**2)/Sigma, -r_dot*Xi/Delta, 
        - z_dot/Xi/y, (-(r**2+a**2)*L/Xi+a*(a*L*Xi+(r**2+a**2)*E*(1/Xi-Xi))*y**2)/Sigma])
    e3 = 1/sqrtK*np.array([(a*((-r_dot)*z + r*theta_dot*y))/Sigma, (a*((a**2 + r**2)*E - a*L)*z)/Delta, r*L/y - a*r*E*y, (1/(2*Sigma))*(y*(-2*r*(a**2 
        + r**2)*theta_dot + 2*a**2*r_dot*y*z))])

    psi = psi_phase(lam, a, p, e, x, qr0, qz0, qpsi0)

    e1 = np.cos(psi)*e1_tilde + np.sin(psi)*e2_tilde
    e2 = -np.sin(psi)*e1_tilde + np.cos(psi)*e2_tilde
    
    return e0, e1, e2, e3


def spin_vector(lam, q, a, p, e, x, s, phi_s, theta_s, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
    """
    Compute the phase spin vector S_a as per Eq. (3.30) of Drummond 2024.
    
    Parameters:
        lam (float): Mino time λ
        q (float): mass ratio
        a (float): Black hole spin
        p (float): semi-latus rectum
        e (float): orbital eccentricity
        x (float): inclination parameter
        s (float): spin magnitude
        phi_s (float): spin orientation
        qr0, qz0, qpsi0 (float): Initial phases (default 0)
        upsilon_r, upsilon_z, upsilon_psi (float): Frequencies (optional)
        
    Returns:
        float: Phase S_i
    """
    #Call the Marck tetrads
    e0, e1, e2, e3 = Marck_tetrad(lam, a, p, e, x, qr0, qz0, qpsi0)
    s_perp = s*np.sin(theta_s)
    s_para = s*np.cos(theta_s)

    #Construct the spin vector
    S1 = s_perp*np.cos(phi_s)*e1
    S2 = s_perp*np.sin(phi_s)*e2
    S3 = s_para*e3

    S_i = S1 + S2 + S3
    return S_i
    
def spin_tensor(lam, q, a, p, e, x, s, phi_s, theta_s, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
    """
    Compute the phase spin vector S_a as per Eq. (3.17)-(3.18) of Drummond 2024.
    
    Parameters:
        lam (float): Mino time λ
        q (float): mass ratio
        a (float): Black hole spin
        p (float): semi-latus rectum
        e (float): orbital eccentricity
        x (float): inclination parameter
        s (float): spin magnitude
        phi_s (float): spin orientation
        qr0, qz0, qpsi0 (float): Initial phases (default 0)
        upsilon_r, upsilon_z, upsilon_psi (float): Frequencies (optional)
        
    Returns:
        float: Phase S_ij
    """
    # Compute E, L, Q from orbital parameters
    constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants

    # Compute K = K (Eq. 49)
    K = Q + (a * E - L)**2
    sqrtK = np.sqrt(K)

    #Calculate the radial and polar roots
    radial_roots = stable_radial_roots(a, p, e, x)
    r1, r2, r3, r4 = radial_roots
    polar_roots = stable_polar_roots(a, p, e, x)
    zm, zp = polar_roots
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)

    #Calculate the radial and polar solutions
    sinI = np.sqrt(1 - x**2)
    r, t_r, phi_r = radial_solutions(a, constants, radial_roots)
    r = r(qr0)
    theta, t_theta, phi_theta = polar_solutions(a, constants, polar_roots)
    theta = theta(qz0)
    z = np.cos(theta)
    y = np.sqrt(1 - z**2)

    #Define the covariant and contravariant Kerr metrics
    spacetime = KerrSpacetime(a=a, M=1)
    Kerr_cov = spacetime.metric(0, r, theta, 0)
    Kerr_contra = np.linalg.inv(Kerr_cov)
    det_g = np.linalg.det(Kerr_cov)
    sqrt_neg_g = np.sqrt(-det_g)
    
    # Levi-Civita symbol (4D)
    levi_civita_sym = np.zeros((4, 4, 4, 4))
    for i, j, k, l in np.ndindex(4, 4, 4, 4):
        if len(set([i, j, k, l])) == 4:
            perm = np.array([i, j, k, l])
            n_inv = 0
            for mu in range(4):
                for nu in range(mu+1, 4):
                    if perm[mu] > perm[nu]:
                        n_inv += 1
            sign = 1 if n_inv % 2 == 0 else -1
            levi_civita_sym[i, j, k, l] = sign
    
    # Covariant Levi-Civita tensor
    epsilon_cov = sqrt_neg_g * levi_civita_sym
    S_vec = spin_vector(lam, q, a, p, e, x, s, phi_s, theta_s, qr0, qz0, qpsi0)

    #Here we obtain the 4-velocity
    orbit = StableOrbit(a, p, e, x)
    u_t, u_r, u_theta, u_phi = orbit.four_velocity()
    u = np.array([u_t(lam), u_r(lam), u_theta(lam), u_phi(lam)])
    
    # Compute S_{αβ} = ε_{αβγδ} u^γ S^δ
    S_cov = np.zeros((4, 4))
    for alpha in range(4):
        for beta in range(4):
            for gamma in range(4):
                for delta in range(4):
                    S_cov[alpha, beta] += epsilon_cov[alpha, beta, gamma, delta] * u[gamma] * S_vec[delta]
    
    # Raise indices to get contravariant spin tensor
    S_contra = np.einsum('ai,bj,ab->ij', Kerr_contra, Kerr_contra, S_cov)
    return S_contra

def spin_curvature_acc(lam, q, a, p, e, x, s, phi_s, theta_s, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
    """
    Compute acceleration from spin-curvature force.
    
    Parameters:
        lam (float): Mino time λ
        q (float): mass ratio
        a (float): Black hole spin
        p (float): semi-latus rectum
        e (float): orbital eccentricity
        x (float): inclination parameter
        s (float): spin magnitude
        phi_s (float): spin orientation
        qr0, qz0, qpsi0 (float): Initial phases (default 0)
        upsilon_r, upsilon_z, upsilon_psi (float): Frequencies (optional)
        
    Returns:
        a (np.ndarray): Acceleration a^μ (4-vector)
    """
    # Compute E, L, Q from orbital parameters
    constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants

    # Compute K = K (Eq. 49)
    K = Q + (a * E - L)**2
    sqrtK = np.sqrt(K)

    #Calculate the radial and polar roots
    radial_roots = stable_radial_roots(a, p, e, x)
    r1, r2, r3, r4 = radial_roots
    polar_roots = stable_polar_roots(a, p, e, x)
    zm, zp = polar_roots
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)

    #Calculate the radial and polar solutions
    sinI = np.sqrt(1 - x**2)
    r, t_r, phi_r = radial_solutions(a, constants, radial_roots)
    r = r(qr0)
    theta, t_theta, phi_theta = polar_solutions(a, constants, polar_roots)
    theta = theta(qz0)
    z = np.cos(theta)
    y = np.sqrt(1 - z**2)

    #Calculate the spin and Riemann tensor
    M = 1
    R = Kerr_riemann(M, a, r, theta)
    S = spin_tensor(lam, q, a, p, e, x, s, phi_s, theta_s, qr0, qz0, qpsi0)
    acc = np.zeros(4)

    #Here we obtain the 4-velocity
    orbit = StableOrbit(a, p, e, x)
    u_t, u_r, u_theta, u_phi = orbit.four_velocity()
    u = np.array([u_t(lam), u_r(lam), u_theta(lam), u_phi(lam)])
    
    for mu_idx in range(4):
        term = 0.0
        for nu in range(4):
            for rho in range(4):
                for sig in range(4):
                    # R^μ_{νλσ} u^ν S^{λσ}
                    term += R[mu_idx, nu, rho, sig] * u[nu] * S[rho, sig]
        acc[mu_idx] = -1/(2) * term
    return acc

def ELK_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
    """
    Compute evolution of Kerr geodesic constants of motion from Lynch 2022 Eq. (B1)-(B5)
    
    Parameters:
        lam (float): Mino time λ
        q (float): mass ratio
        a (float): Black hole spin
        p (float): semi-latus rectum
        e (float): orbital eccentricity
        x (float): inclination parameter
        s (float): spin magnitude
        phi_s (float): spin orientation
        qr0, qz0, qpsi0 (float): Initial phases (default 0)
        upsilon_r, upsilon_z, upsilon_psi (float): Frequencies (optional)
        
    Returns:
        E_dot, L_dot, Q_dot (np.ndarray): derivative of constants with respect to affine parameter
    """
    # Compute E, L, Q from orbital parameters
    constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants

    # Compute K = K (Eq. 49)
    K = Q + (a * E - L)**2
    sqrtK = np.sqrt(K)

    #Calculate the radial and polar roots
    radial_roots = stable_radial_roots(a, p, e, x)
    r1, r2, r3, r4 = radial_roots
    polar_roots = stable_polar_roots(a, p, e, x)
    zm, zp = polar_roots
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)

    #Calculate the radial and polar solutions
    sinI = np.sqrt(1 - x**2)
    r, t_r, phi_r = radial_solutions(a, constants, radial_roots)
    r = r(qr0)
    theta, t_theta, phi_theta = polar_solutions(a, constants, polar_roots)
    theta = theta(qz0)
    z = np.cos(theta)
    y = np.sqrt(1 - z**2)

    #Define some Kerr parameter
    Sigma = r**2 + a**2*z**2
    Delta = r**2 - 2*r + a**2
    omega = np.sqrt(r**2 + a**2)
    
    #Compute the metric components
    spacetime = KerrSpacetime(a=a, M=1)
    Kerr_cov = spacetime.metric(0, r, theta, 0)
    gtt = Kerr_cov[0,0]
    gphiphi = Kerr_cov[3,3]
    grr =  Kerr_cov[1,1]
    
    #Here we obtain the 4-velocity
    orbit = StableOrbit(a, p, e, x)
    u_t, u_r, u_theta, u_phi = orbit.four_velocity()
    u = np.array([u_t(lam), u_r(lam), u_theta(lam), u_phi(lam)])

    #Compute the energy and angular momentum evolution from the acceleration
    acc = spin_curvature_acc(lam, q, a, p, e, x, s, phi_s, theta_s, qr0, qz0, qpsi0)
    E_dot = -Sigma*acc[0]*gtt
    L_dot = Sigma*acc[3]*gphiphi
    E, L, Q = constants_of_motion(a, p, e, x)
    K_dot = 2/Delta*(E_dot*(omega**4*E - a*omega**2*L) + L_dot*(a**2*L - a*omega**2*E)) - 2*Sigma*Delta*u[1]*acc[1]*grr**2
    return E_dot, L_dot, K_dot

def pex_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
    """
    Compute evolution of Kerr geodesic constants of motion from Lynch 2022 Eq. (B1)-(B5)
    
    Parameters:
        lam (float): Mino time λ
        q (float): mass ratio
        a (float): Black hole spin
        p (float): semi-latus rectum
        e (float): orbital eccentricity
        x (float): inclination parameter
        s (float): spin magnitude
        phi_s (float): spin orientation
        qr0, qz0, qpsi0 (float): Initial phases (default 0)
        upsilon_r, upsilon_z, upsilon_psi (float): Frequencies (optional)
        
    Returns:
        p_dot, e_dot, x_dot (np.ndarray): derivative of quasi-Keplerian parameters with respect to affine parameter
    """
    # Compute E, L, Q from orbital parameters
    constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants

    # Compute K = K (Eq. 49)
    K = Q + (a * E - L)**2
    sqrtK = np.sqrt(K)

    #Calculate the radial and polar roots
    radial_roots = stable_radial_roots(a, p, e, x)
    r1, r2, r3, r4 = radial_roots
    polar_roots = stable_polar_roots(a, p, e, x)
    zm, zp = polar_roots
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)

    #Calculate the radial and polar solutions
    sinI = np.sqrt(1 - x**2)
    r, t_r, phi_r = radial_solutions(a, constants, radial_roots)
    r = r(qr0)
    theta, t_theta, phi_theta = polar_solutions(a, constants, polar_roots)
    theta = theta(qz0)
    z = np.cos(theta)
    y = np.sqrt(1 - z**2)
    beta = a**2*(1 - E**2)
    
    #Define the other necessary coefficients
    Sigma_1 = r1**2 + a**2*z**2
    Delta_1 = r1**2 - 2*r1 + a**2
    omega_1 = np.sqrt(r1**2 + a**2)
    F_1 = omega_1**2*E - a*L
    kappa_1 = 4*E*F_1*r1 - 2*r1*Delta_1 - 2*(r1 - 1)*(r1**2 + K)
    Sigma_2 = r2**2 + a**2*z**2
    Delta_2 = r2**2 - 2*r2 + a**2
    omega_2 = np.sqrt(r2**2 + a**2)
    F_2 = omega_2**2*E - a*L
    kappa_2 = 4*E*F_2*r2 - 2*r2*Delta_2 - 2*(r2 - 1)*(r2**2 + K)

    #Calculate the evolution of the constants of motion
    E_dot, L_dot, K_dot = ELK_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, qr0, qz0, qpsi0)

    #Calculate the evolution of r1 and r2
    r1_dot =  -2*F_1/kappa_1*(omega_1**2*E_dot - a*L_dot) + Delta_1/kappa_1*K_dot
    r2_dot =  -2*F_2/kappa_2*(omega_2**2*E_dot - a*L_dot) + Delta_2/kappa_2*K_dot

    #Calculate the evolution of zm
    zm_dot = 1/(2*zm*(zp**2 - beta*zm**2))*(x**2*K_dot - 2*(L - a*x**2*E)*(L_dot - a*x**2*E_dot))

    #Compute the evolution of quasi-Keplerian orbital parameters 
    p_dot = 2/(r1+r2)**2*(r2**2*r1_dot + r1**2*r2_dot)
    e_dot = 2/(r1+r2)**2*(r2**2*r1_dot - r1**2*r2_dot)
    x_dot = -zm/x*zm_dot


    return p_dot, e_dot, x_dot

def phase_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
    """
    Compute evolution of Kerr geodesic constants of motion from Lynch 2022 Eq. (B1)-(B5)
    
    Parameters:
        lam (float): Mino time λ
        q (float): mass ratio
        a (float): Black hole spin
        p (float): semi-latus rectum
        e (float): orbital eccentricity
        x (float): inclination parameter
        s (float): spin magnitude
        phi_s (float): spin orientation
        qr0, qz0, qpsi0 (float): Initial phases (default 0)
        upsilon_r, upsilon_z, upsilon_psi (float): Frequencies (optional)
        
    Returns:
        qr_dot, qz_dot (np.ndarray): derivative of orbital phase with respect to affine parameter
    """
    # Compute E, L, Q from orbital parameters
    constants = constants_of_motion(a, p, e, x)
    E, L, Q = constants

    # Compute K = K (Eq. 49)
    K = Q + (a * E - L)**2
    sqrtK = np.sqrt(K)

    #Calculate the radial and polar roots
    radial_roots = stable_radial_roots(a, p, e, x)
    r1, r2, r3, r4 = radial_roots
    polar_roots = stable_polar_roots(a, p, e, x)
    zm, zp = polar_roots
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)

    #Calculate the radial and polar solutions
    sinI = np.sqrt(1 - x**2)
    r, t_r, phi_r = radial_solutions(a, constants, radial_roots)
    r = r(qr0)
    theta, t_theta, phi_theta = polar_solutions(a, constants, polar_roots)
    theta = theta(qz0)
    z = np.cos(theta)
    y = np.sqrt(1 - z**2)

    #Define some Kerr parameter
    Sigma = r**2 + a**2*z**2
    Delta = r**2 - 2*r + a**2
    omega = np.sqrt(r**2 + a**2)
    beta = a**2*(1 - E**2)

    #Define the partial derivative of the potentials
    dVzdz = -2*z*(L**2 + Q + a**2*(E**2 - 1)*(2*z**2 - 1))
    dVrdr = 2*(K - K*r + r*(E + (3 - 2*r)*r) - a**2) 
    dz_dot_dE = -(a**2*E*z**2*(z**2 - 1))/np.sqrt(Q - z**2*(L**2 + Q + a**2*(E**2 - 1)*(z**2 - 1)))
    dz_dot_dL = L*z**2/np.sqrt(Q - z**2*(L**2 + Q + a**2*(E**2-1)*(z**2-1)))
    dz_dot_dK = 0
    dr_dot_dE = (r**2 + a**2)/(2*np.sqrt(-a*L + E*(r**2 + a**2) - (a**2 + (r - 2)*r)*(K + r**2)))
    dr_dot_dL = a/(2*np.sqrt(-a*L + E*(r**2 + a**2) - (a**2 + (r - 2)*r)*(K + r**2)))
    dr_dot_dK = (r*(r - 2) + a**2)/(2*np.sqrt(-a*L + E*(r**2 + a**2) - (a**2 + (r - 2)*r)*(K + r**2)))
    
    #dz_dot_dE = -(a**2*E*z**2*(z**2-1))/np.sqrt(Q - z**2*(L**2 + Q + a**2*(E**2-1)*(z**2-1)))
    #dz_dot_dL = L*z**2/np.sqrt(Q - z**2*(L**2 + Q + a**2*(E**2-1)*(z**2-1)))
    #dz_dot_dQ = (1 - z**2)/(2*np.sqrt(Q - z**2*(L**2 + Q + a**2*(E**2-1)*(z**2-1))))
    #dr_dot_dE = (a**2 + r**2 + 2*a*((-a)*E + L)*(a**2 + (-2 + r)*r))/(2*np.sqrt((-a)*L + E*(a**2 + r**2) - (a**2 + (-2 + r)*r)*(((-a)*E + L)**2 + Q + r**2)))
    #dr_dot_dL = (2*(a*E - L)*(a**2 + (-2 + r)*r) - a)/(2*np.sqrt((-a)*L + E*(a**2 + r**2) - (a**2 + (-2 + r)*r)*(((-a)*E + L)**2 + Q + r**2)))
    #dr_dot_dQ = -(a**2 + (-2 + r)*r)/(2*np.sqrt((-a)*L + E*(a**2 + r**2) - (a**2 + (-2 + r)*r)*(((-a)*E + L)**2 + Q + r**2)))

    #Define the Mino frequencies
    Upsilon_r, Upsilon_theta, Upsilon_phi, Gamma = mino_frequencies(a, p, e, x)
    Upsilon_z = Upsilon_theta
    #Compute the spin-curvature acceleration
    acc = spin_curvature_acc(lam, q, a, p, e, x, s, phi_s, theta_s, qr0, qz0, qpsi0)

    #Compute the evolution of the constants of motion
    E_dot, L_dot, K_dot = ELK_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, qr0, qz0, qpsi0)
    Q_dot = K_dot - 2*(L - a*E)*(L_dot - a*E_dot)


    #Compute the phase evolution
    qr_dot = 2*Upsilon_r/dVrdr*(Sigma**2*acc[1] - (dr_dot_dE*E_dot + dr_dot_dL*L_dot + dr_dot_dK*K_dot)) + Upsilon_r
    qz_dot = 2*Upsilon_z/dVzdz*(Sigma**2*acc[2]*y - (dz_dot_dE*E_dot + dz_dot_dL*L_dot + dz_dot_dK*K_dot)) + Upsilon_z
    

    return qr_dot, qz_dot
import numpy as np
from kerrgeopy.spacetime import *
from scipy.special import ellipk, ellipe, elliprj, elliprf, elliprd, ellipj
from .frequencies import _ellippiinc, precession_frequency, mino_frequencies
from .constants import *
from .stable import *

def Kerr_riemann(M, a, r, theta):
    """
    Compute the Riemann tensor for Kerr spacetime at a given point.
    
    Parameters:
        M (float): Black hole mass
        a (float): Black hole spin parameter
        r (float): Radial coordinate
        theta (float): Polar angle coordinate
        
    Returns:
        R (np.ndarray): Riemann tensor components R**ρ_[σμν] (4x4x4x4 array)
    """
    
    # Riemann tensor components
    R = np.zeros((4, 4, 4, 4))

    cot_theta = 1/np.tan(theta)
    
    # Independent non-zero component
    R[0, 0, 0, 3] = -((16*a*M**2*r**2*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[0, 0, 1, 2] = (4*a**2*M**2*r*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3)
    R[0, 1, 0, 1] = (M*r*(-27*a**4 - 4*a**2*r**2 + 16*r**4 - 4*a**2*(6*a**2 + 7*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/((a**2 + r*(-2*M + r))*(a**2 
                    + 2*r**2  + a**2*np.cos(2*theta))**3)
    R[0, 1, 0, 2] = (2*a**2*M*(3*a**2 + r*(-2*M + 3*r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[0, 1, 1, 3] = -((12*a*M*r*(a**2 + r**2)*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3))
    R[0, 1, 2, 3] = -((a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(-5*a**4 + a**2*(2*M - 9*r)*r - 4*r**4 + a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(2*theta)))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3)
    R[0, 2, 0, 2] = -((2*M*r*(-9*a**4 + 2*a**2*r**2 + 4*r**4 - 2*a**2*(3*a**2 + 5*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[0, 2, 1, 3] = -((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(-2*a**4 + a**2*(2*M - 3*r)*r - r**4 + a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[0, 2, 2, 3] = (12*a*M*r*(a**2 + r**2)*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3
    R[0, 3, 0, 3] = -((4*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*(-a**4 - 2*r**4 - a**2*r*(2*M + 3*r) - a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[1, 0, 0, 1] = -((4*M*r*(-3*a**2 + 2*r**2 - 3*a**2*np.cos(2*theta))*(-5*a**2 + 8*M*r - 4*r**2 + a**2*np.cos(2*theta)))/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**4)
    R[1, 0, 0, 2] = (12*a**2*M*(a**2 + r*(-2*M + r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[1, 0, 1, 3] = -((8*a*M*r*(3*a**2 + r*(-4*M + 3*r))*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[1, 0, 2, 3] = (a*M*(a**2 + r*(-2*M + r))*(9*a**4 - 52*a**2*r**2 - 48*r**4 + 4*a**2*(2*a**2 + 5*r**2)*np.cos(2*theta) 
                    - a**4*np.cos(4*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[1, 2, 0, 3] = (2*a*M*(a**2 + r*(-2*M + r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3
    R[1, 2, 1, 2] = (M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/(2*(r**2 + a**2*np.cos(theta)**2)**2)
    R[1, 3, 1, 3] = (8*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*(2*a**4 + r**4 + a**2*r*(-2*M + 3*r) - a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[1, 3, 2, 3] = -((24*a**2*M*(a**2 + r**2)*(a**2 + r*(-2*M + r))*np.cos(theta)*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(theta)**3)/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**4)
    R[2, 0, 0, 1] = (12*a**2*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[2, 0, 0, 2] = (8*M*r*(-3*a**2 + 2*r**2 - 3*a**2*np.cos(2*theta))*(-2*a**2 + (2*M - r)*r + a**2*np.cos(2*theta)))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[2, 0, 1, 3] = -((2*a*M*(-3*a**4 + 22*a**2*r**2 + 12*r**4 - 2*a**2*(a**2 + 7*r**2)*np.cos(2*theta) + a**4*np.cos(4*theta))*np.sin(2*theta))/(a**2 
                    + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[2, 0, 2, 3] = (8*a*M*r*(3*a**2 + r*(-2*M + 3*r))*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    R[2, 1, 0, 3] =-((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3)
    R[2, 1, 1, 2] = -((M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/(2*(a**2 + r*(-2*M + r))*(r**2 + a**2*np.cos(theta)**2)**2))
    R[2, 3, 0, 2] = -((8*a*M*r*(3*a**2 + r*(-2*M + 3*r))*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[2, 3, 1, 3] = -((24*a**2*M*(a**2 + r**2)*np.cos(theta)*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(theta)**3)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[2, 3, 2, 3] = -((4*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*(5*a**4 + 4*r**4 + a**2*r*(-2*M + 9*r) - a**2*(a**2 + r*(-2*M 
                    + r))*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
    R[3, 0, 0, 3] = (4*M*r*(-3*a**2 + 2*r**2 - 3*a**2*np.cos(2*theta))*(-a**2 + 4*M*r - 2*r**2 - a**2*np.cos(2*theta)))/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**4
    R[3, 0, 1, 2] = -((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(a**2 + 2*r*(-2*M + r) + a**2*np.cos(2*theta))*cot_theta )/((a**2 + r*(-2*M + r))*(a**2 
                    + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[3, 1, 0, 1] = -((12*a*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[3, 1, 0, 2] = -((4*a*M*(-2*a**2 + (2*M - r)*r + a**2*np.cos(2*theta))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*cot_theta )/((a**2 + r*(-2*M + r))*(a**2 
                    + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[3, 1, 1, 3] = (2*M*r*(-9*a**4 + 2*a**2*r**2 + 4*r**4 - 2*a**2*(3*a**2 + 5*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/((a**2 + r*(-2*M + r))*(a**2
                    + 2*r**2 + a**2*np.cos(2*theta))**3)
    R[3, 1, 2, 3] = (2*a**2*M*(3*a**2 + r*(-2*M + 3*r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[3, 2, 0, 1] = -((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(-5*a**2 + 8*M*r - 4*r**2 + a**2*np.cos(2*theta))*cot_theta )/((a**2 + r*(-2*M + r))*(a**2 
                    + 2*r**2 + a**2*np.cos(2*theta))**3))
    R[3, 2, 0, 2] = (12*a*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3
    R[3, 2, 1, 3] = (2*a**2*M*(3*a**2 + r*(-4*M + 3*r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[3, 2, 2, 3] = -((M*r*(-27*a**4 - 4*a**2*r**2 + 16*r**4 - 4*a**2*(6*a**2 + 7*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/(a**2 + 2*r**2 
                    + a**2*np.cos(2*theta))**3)
    R[3, 3, 0, 3] = (16*a*M**2*r**2*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
    

    #Non-zero components by symmetry of Riemann tensor
    R[0, 0, 2, 1] = - R[0, 0, 1, 2]
    R[0, 0, 3, 0] = - R[0, 0, 0, 3]
    R[0, 1, 1, 0] = - R[0, 1, 0, 1]
    R[0, 1, 2, 0] = - R[0, 1, 0, 2]  
    R[0, 1, 3, 1] = - R[0, 1, 1, 3] 
    R[0, 1, 3, 2] = - R[0, 1, 2, 3]
    R[0, 2, 0, 1] = - R[0, 0, 1, 2] - R[0, 1, 2, 0]
    R[0, 2, 1, 0] = - R[0, 2, 0, 1]
    R[0, 2, 2, 0] = - R[0, 2, 0, 2]
    R[0, 2, 3, 1] = - R[0, 2, 1, 3]
    R[0, 2, 3, 2] = - R[0, 2, 2, 3]
    R[0, 3, 1, 2] = - R[0, 1, 2, 3] - R[0, 2, 3, 1]
    R[0, 3, 2, 1] = - R[0, 3, 1, 2]
    R[0, 3, 3, 0] = - R[0, 3, 0, 3]
    R[1, 0, 1, 0] = - R[1, 0, 0, 1]
    R[1, 0, 2, 0] = - R[1, 0, 0, 2]
    R[1, 0, 3, 1] = - R[1, 0, 1, 3] 
    R[1, 0, 3, 2] = - R[1, 0, 2, 3] 
    R[1, 2, 2, 1] = - R[1, 2, 1, 2]  
    R[1, 2, 3, 0] = - R[1, 2, 0, 3]
    R[1, 3, 0, 1] = - R[1, 0, 1, 3] - R[1, 1, 3, 0]
    R[1, 3, 0, 2] = - R[1, 0, 2, 3] - R[1, 2, 3, 0]
    R[1, 3, 1, 0] = - R[1, 3, 0, 1]
    R[1, 3, 2, 0] = - R[1, 3, 0, 2] 
    R[1, 3, 3, 1] = - R[1, 3, 1, 3]  
    R[1, 3, 3, 2] = - R[1, 3, 2, 3]
    R[2, 0, 1, 0] = - R[2, 0, 0, 1]
    R[2, 0, 2, 0] = - R[2, 0, 0, 2]
    R[2, 0, 3, 1] = - R[2, 0, 1, 3]
    R[2, 0, 3, 2] = - R[2, 0, 2, 3]
    R[2, 1, 2, 1] = - R[2, 1, 1, 2]
    R[2, 1, 3, 0] = - R[2, 1, 0, 3]
    R[2, 3, 0, 1] = - R[2, 0, 1, 3] - R[2, 1, 3, 0]
    R[2, 3, 1, 0] = - R[2, 3, 0, 1]
    R[2, 3, 2, 0] = - R[2, 3, 0, 2]
    R[2, 3, 2, 0] = - R[2, 2, 0, 3] - R[2, 0, 3, 2]
    R[2, 3, 3, 1] = - R[2, 3, 1, 3]
    R[2, 3, 3, 2] = - R[2, 3, 2, 3]
    R[3, 0, 2, 1] = - R[3, 0, 1, 2] 
    R[3, 0, 3, 0] = - R[3, 0, 0, 3]
    R[3, 1, 1, 0] = - R[3, 1, 0, 1] 
    R[3, 1, 2, 0] = - R[3, 1, 0, 2]
    R[3, 1, 3, 1] = - R[3, 1, 1, 3]
    R[3, 1, 3, 2] = - R[3, 1, 2, 3]
    R[3, 2, 1, 0] = - R[3, 2, 0, 1] 
    R[3, 2, 2, 0] = - R[3, 2, 0, 2]
    R[3, 2, 3, 1] = - R[3, 2, 1, 3]
    R[3, 2, 3, 2] = - R[3, 2, 2, 3]
    R[3, 3, 1, 2] = - R[3, 1, 2, 3] - R[3, 2, 3, 1] 
    R[3, 3, 2, 1] = - R[3, 3, 1, 2]
    R[3, 3, 3, 0] = - R[3, 3, 0, 3] - R[3, 0, 3, 3]    
    return R

def F_aero_TF(lam, rho, M, aS, p, e, xI, beta):
    """
    Compute the aerodynamic forces in the tetrad frame using the result from Dyson, et al. (2024).
    
    Parameters:
        lam (float): Mino time λ
        rho (float): fluid density 
        M (float): Secondary Black hole mass
        aS (float): Secondary Black hole spin
        p (float): semi-latus rectum
        e (float): orbital eccentricity
        x (float): inclination parameter
        
    Returns:
        tuple: (Forces in tetrad frame) F1, F2, F3 
    """
    #Calculate the velocities
    qt0, qr0, qtheta0, qphi0 = 0,0,0,0
    orbit = kg.StableOrbit(aS, p, e, xI, initial_phases=(qt0, qr0, qtheta0, qphi0))
    t_func, r_func, theta_func, phi_func = orbit.trajectory()
    ut_func, ur_func, utheta_func, uphi_func = orbit.four_velocity()
    mino_time = lam
    r = r_func(mino_time)
    theta = theta_func(mino_time)
    vr = ur_func(mino_time)/ut_func(mino_time)
    vtheta = utheta_func(mino_time)/ut_func(mino_time)
    vphi = uphi_func(mino_time)/ut_func(mino_time)
    v = np.sqrt(vr**2 + r**2*(vtheta**2 + np.sin(theta)*vphi**2))
    
    #Drag force from Traykova, et al. (2024)
    spacetime = kg.KerrSpacetime(a=a, M=1)
    gamma = 1/np.sqrt(1-v**2)
    bmax = p/(1-e)
    bmin = kg.constants.separatrix(a, e, x)
    Lambda = bmax/bmin
    bcr = M/v**2*np.sqrt((-1 + 8*v**2 + np.sqrt(1 + 8*v**2) + 4*v**2*(5 + 2*np.sqrt(1 + 8*v**2)))/2)
    chi = (3.7 + 9.24*v**2 - 12.16*v**3 + 4.79*v**4 - np.log(10*v*bcr/M))/M**2
    Fd = - gamma**2*v**2*np.pi*bcr**2/M**2
    Fd = 4*np.pi/v**2*gamma*(1 + v**2)**2*(np.log(Lambda) + chi)
    
    #Aerodynamical forces from Dyson, et al. (2025)
    Fd += aS**2*(1.11-3.2*v + 0.74*aS - 1.07*np.sqrt(v)*aS - 0.59*np.cos(beta) + 1.38*np.sqrt(v)*np.cos(beta))
    Fm = aS*np.sin(beta)*(2.55 - 31.81*np.sqrt(v) + 69.81*v - 59.15*v**(3/2) - 0.21*aS*np.sqrt(v)*np.cos(beta))
    Fl = aS**2*np.sin(2*beta)*(0.82 - 2.6*np.sqrt(v) + 5.06*v - 3.79*v**(3/2) + 0.28*aS + 1.28*v*aS 
                              - 0.26*np.sqrt(v)*aS*np.cos(beta) + 0.11*aS*np.cos(beta)-1.22*np.sqrt(v)*aS 
                              - 0.02*np.cos(beta) + 0.04*np.sqrt(v)*np.cos(beta) + 0.23*np.sqrt(v)*aS**2)
    F1 = rho*M**2*Fm
    F2 = rho*M**2*(Fl*np.cos(beta) - Fd*np.sin(beta))
    F3 = rho*M**2*(Fd*np.cos(beta) + Fl*np.sin(beta))
    return F1, F2, F3

def Marck_tetrads_Skoupy():
    return
    

