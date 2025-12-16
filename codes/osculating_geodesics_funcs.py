import numpy as np
from .spacetime import *
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
    R[3, 3, 2, 1] = -R[3, 3, 1, 2]
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
    
    # Compute K_val = K (Eq. 49)
    K_val = Q + (a * E - L)**2
    sqrtK = np.sqrt(K_val)

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
    qr = Upsilon_r*lam
    qtheta = Upsilon_theta*lam
    qz = qtheta 
    #Here, we define some complementary function
    sn_r, cn_r, dn_r, ph_r = ellipj(qr/np.pi*ellipk(k_r), k_r)
    sn_z, cn_z, dn_z, ph_z = ellipj(2*qz/np.pi*ellipk(k_z), k_z)
    xi_r = ph_r
    xi_z = ph_z
    def psi_r_tilde(a, p, e, x, xi):
        psi_r_tilde = 2*(r2 - r3)*((K_val - a**2)*E+a*L)/((K_val + r2**2)*(K_val + r3**2)*np.sqrt((1 - E**2)*(r1 - r3)*(r2 - r4)))*np.imag((r2 + 1j*sqrtK)*(r3 
                        + 1j*sqrtK)*_ellippiinc(xi, h_psi, k_r))    
        return psi_r_tilde
    
    def psi_z_tilde(a, p, e, x, xi):
        psi_z_tilde = -((a**2 - K_val)*E - a*L)/(sqrtK*z2*_ellippiinc(xi, a**2*z1**2/K_val, k_z)) 
        return psi_z_tilde
    psi_r = psi_r_tilde(a, p, e, x, xi_r) - psi_r_tilde(a, p, e, x, np.pi)/(2*np.pi)*qr
    psi_z = psi_z_tilde(a, p, e, x, xi_z) - psi_z_tilde(a, p, e, x, np.pi)/np.pi*qr
    psi += psi_r + psi_z
    return psi

def Marck_tetrad(lam, a, p, e, x, chi_r, chi_phi):
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
    sinI = np.sqrt(1 - x**2)
    r = p/(1+e*np.cos(chi_r))
    z = sinI*np.cos(chi_phi)
    y = np.sqrt(1 - z**2)


    # Compute E, L, Q from orbital parameters
    E, L, Q = constants_of_motion(a, p, e, x)

    # Compute K_val = K (Eq. 49)
    K_val = Q + (a * E - L)**2
    sqrtK = np.sqrt(K_val)

    # Define some parameters for Kerr geodesics
    Delta = r**2 - 2*r + a**2
    Xi = np.sqrt((K_val - a**2*z**2)/(K_val + r**2))
    Sigma = r**2 + a**2*z**2

    # Define the radial and polar potentials 
    R = (E*(r**2 + a**2) - a*L)**2 - Delta*(r**2 + (a*E - L)**2 + Q)
    #Z = Q - z**2*(a**2*(1 - E**2)*(1 - z**2) + L**2 + Q)
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

    psi = psi_phase(lam, a, p, e, x)

    e1 = np.cos(psi)*e1_tilde + np.sin(psi)*e2_tilde
    e2 = -np.sin(psi)*e1_tilde + np.cos(psi)*e2_tilde
    
    return e0, e1, e2, e3

def spin_vector(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
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
    e0, e1, e2, e3 = Marck_tetrad(lam, a, p, e, x, chi_r, chi_phi)
    s_perp = s*np.sin(theta_s)
    s_para = s*np.cos(theta_s)

    #Construct the spin vector
    S1 = s_perp*np.cos(phi_s)*e1
    S2 = s_perp*np.sin(phi_s)*e2
    S3 = s_para*e3
    S_i = S1 + S2 + S3
    return q**2*S_i
    
def spin_tensor(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
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
    #Define the the r and theta parameter
    sinI = np.sqrt(1 - x**2)
    r = p/(1+e*np.cos(chi_r))
    theta = np.arccos(sinI*np.cos(chi_phi))

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
    S_vec = spin_vector(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi)

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

def spin_curvature_acc(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
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
    #Define the the r and theta parameter
    sinI = np.sqrt(1 - x**2)
    r = p/(1+e*np.cos(chi_r))
    theta = np.arccos(sinI*np.cos(chi_phi))

    #Calculate the spin and Riemann tensor
    M = 1
    R = Kerr_riemann(M, a, r, theta)
    S = spin_tensor(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi)
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
        acc[mu_idx] = -1/(2*q) * term
    return acc

def ELK_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
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
    #Define the the r and theta parameter
    sinI = np.sqrt(1 - x**2)
    r = p/(1+e*np.cos(chi_r))
    z = sinI*np.cos(chi_phi)
    theta = np.arccos(sinI*np.cos(chi_phi))
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
    acc = spin_curvature_acc(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi)
    E_dot = -Sigma*acc[0]*gtt
    L_dot = Sigma*acc[3]*gphiphi
    E, L, Q = constants_of_motion(a, p, e, x)
    K_dot = 2/Delta*(E_dot*(omega**4*E - a*omega**2*L) + L_dot*(a**2*L - a*omega**2*E)) - 2*Sigma*Delta*u[1]*acc[1]*grr**2
    return E_dot, L_dot, K_dot

def pex_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
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
    #Define the the r and theta parameter
    sinI = np.sqrt(1 - x**2)
    r = p/(1+e*np.cos(chi_r))
    z = sinI*np.cos(chi_phi)
    
    #Calculate the constants of motion
    E, L, Q = constants_of_motion(a, p, e, x)
    K_val = Q + (a * E - L)**2
    beta = a**2*(1 - E**2)
    
    #Calculate the radial and polar roots
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x)
    zm, zp = stable_polar_roots(a, p, e, x)
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)
    
    #Define the other necessary coefficients
    Sigma_1 = r1**2 + a**2*z**2
    Delta_1 = r1**2 - 2*r1 + a**2
    omega_1 = np.sqrt(r1**2 + a**2)
    F_1 = omega_1**2*E - a*L
    kappa_1 = 4*E*F_1*r1 - 2*r1*Delta_1 - 2*(r1 - 1)*(r1**2 + K_val)
    Sigma_2 = r2**2 + a**2*z**2
    Delta_2 = r2**2 - 2*r2 + a**2
    omega_2 = np.sqrt(r2**2 + a**2)
    F_2 = omega_2**2*E - a*L
    kappa_2 = 4*E*F_2*r2 - 2*r2*Delta_2 - 2*(r2 - 1)*(r2**2 + K_val)

    #Calculate the evolution of the constants of motion
    E_dot, L_dot, K_dot = ELK_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi)

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

def phase_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
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
    #Define the the r and theta parameter
    sinI = np.sqrt(1 - x**2)
    r = p/(1+e*np.cos(chi_r))
    z = sinI*np.cos(chi_phi)
    y = np.sqrt(1-z**2)
    Sigma = r**2 + a**2*z**2
    Delta = r**2 - 2*r + a**2
    omega = np.sqrt(r**2 + a**2)

    #Calculate the constants of motion
    E, L, Q = constants_of_motion(a, p, e, x)
    K_val = Q + (a * E - L)**2
    beta = a**2*(1 - E**2)

    #Calculate the radial and polar roots
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x)
    zm, zp = stable_polar_roots(a, p, e, x)
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)

    #Define the partial derivative of the potentials
    dVzdz = -2*z*(L**2 + Q + a**2*(E**2 - 1)*(2*z**2 - 1))
    dVrdr = 2*(K_val - K_val*r + r*(E + (3 - 2*r)*r) - a**2) 
    dz_dot_dE = -(a**2*E*z**2*(z**2 - 1))/np.sqrt(Q - z**2*(L**2 + Q + a**2*(E**2 - 1)*(z**2 - 1)))
    dz_dot_dL = L*z**2/np.sqrt(Q - z**2*(L**2 + Q + a**2*(E**2-1)*(z**2-1)))
    dz_dot_dK = 0
    dr_dot_dE = (r**2 + a**2)/(2*np.sqrt(-a*L + E*(r**2 + a**2) - (a**2 + (r - 2)*r)*(K_val + r**2)))
    dr_dot_dL = a/(2*np.sqrt(-a*L + E*(r**2 + a**2) - (a**2 + (r - 2)*r)*(K_val + r**2)))
    dr_dot_dK = (r*(r - 2) + a**2)/(2*np.sqrt(-a*L + E*(r**2 + a**2) - (a**2 + (r - 2)*r)*(K_val + r**2)))
    
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
    acc = spin_curvature_acc(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi)

    #Compute the evolution of the constants of motion
    E_dot, L_dot, K_dot = ELK_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi)
    Q_dot = K_dot - 2*(L - a*E)*(L_dot - a*E_dot)


    #Compute the phase evolution
    qr_dot = 2*Upsilon_r/dVrdr*(Sigma**2*acc[1] - (dr_dot_dE*E_dot + dr_dot_dL*L_dot + dr_dot_dK*K_dot)) + Upsilon_r
    qz_dot = 2*Upsilon_z/dVzdz*(Sigma**2*acc[2]*y - (dz_dot_dE*E_dot + dz_dot_dL*L_dot + dz_dot_dK*K_dot)) + Upsilon_z
    

    return qr_dot, qz_dot

def qi_to_chi(lam, dlam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi, qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None, upsilon_psi=None):
    """
    Compute evolution of Kerr geodesic constants of motion from Lynch 2022 Eq. (B1)-(B5)
    
    Parameters:
        lam (float): Mino time λ
        dlam (float): Mino time step dλ
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
    #Compute qr_dot and qz_dot
    qr_dot, qz_dot = phase_spin_evolution(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi)

    #Compute the action-angle variable
    qr = qr0 + qr_dot*dlam 
    qz = qz0 + qz_dot*dlam

    #Calculate the constants of motion
    E, L, Q = constants_of_motion(a, p, e, x)
    K_val = Q + (a * E - L)**2
    beta = a**2*(1 - E**2)
    
    # Compute moduli k_r and k_z (Eq. 18, 19)
    k_r = np.sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))
    k_z = np.sqrt(a**2 * (1 - E**2) * z1**2 / z2**2)
    

    #Calculate the radial and polar roots
    r1, r2, r3, r4 = stable_radial_roots(a, p, e, x)
    zm, zp = stable_polar_roots(a, p, e, x)
    z1, z2 = np.sqrt(zm), a*np.sqrt((1 - E**2)*zp)

    
    #Here, we define some complementary function
    sn_r, cn_r, dn_r, ph_r = ellipj(qr/np.pi*ellipk(k_r), k_r)
    sn_z, cn_z, dn_z, ph_z = ellipj(2*(qz+np.pi/2)/np.pi*ellipk(k_z), k_z)

    #Compute the radial and angular coordinates
    r = (r3*(r1 - r2)*sn_r**2 - r2*(r1 - r3))/((r1 - r2)*sn_r**2 - (r1 - r3))
    z = zm*sn_z

    chi_r = np.arccos((1 - p/r)/e)
    chi_theta = sn_z
    return chi_r, chi_theta

def killing_vectors(self, r, theta):
        """Compute time-like and axial Killing vectors (contravariant)"""
        xi_t = np.array([1.0, 0.0, 0.0, 0.0])  # ∂/∂t
        xi_phi = np.array([0.0, 0.0, 0.0, 1.0])  # ∂/∂φ
        return xi_t, xi_phi
    
def killing_vectors_covariant(self, r, theta):
    """Compute covariant forms of Killing vectors"""
    xi_t, xi_phi = self.killing_vectors(r, theta)
    g = spacetime.metric(0, r, theta, 0)
    xi_t_cov = np.dot(g, xi_t)
    xi_phi_cov = np.dot(g, xi_phi)
    return xi_t_cov, xi_phi_cov

def conformal_killing_yano_tensor(self, r, theta):
    """Compute Killing-Yano tensor (covariant form)"""
    a = self.a
    M = self.M if self.M is not None else 1.0
    
    # Initialize tensor components
    f = np.zeros((4, 4))
    
    # Components based on Eq. (4) from the paper
    f[0, 1] = -r  # f_tr
    f[1, 0] = r   # f_rt
    f[0, 2] = -a**2 * np.cos(theta)  # f_tθ
    f[2, 0] = a**2 * np.cos(theta)   # f_θt
    f[1, 3] = a * r * np.sin(theta)**2  # f_rφ
    f[3, 1] = -a * r * np.sin(theta)**2  # f_φr
    f[2, 3] = a * np.cos(theta) * (r**2 + a**2)  # f_θφ
    f[3, 2] = -a * np.cos(theta) * (r**2 + a**2)  # f_φθ
    
    return f

def killing_yano_tensor(self, r, theta):
    """
    Compute the Killing-Yano tensor Y from the conformal Killing-Yano tensor f.

    Args:
        r (float): Radial coordinate
        theta (float): Angular coordinate (in radians)

    Returns:
        numpy.ndarray: Killing-Yano tensor Y (4x4, covariant components)
    """
    # Compute conformal Killing-Yano tensor f (covariant form)
    f_cov = self.conformal_killing_yano_tensor(r, theta)
    
    # Compute metric using the existing class method
    g_cov = self.metric(0, r, theta, 0)  # t and phi can be set to 0 since metric doesn't depend on them
    
    # Compute inverse metric
    g_inv = np.linalg.inv(g_cov)
    
    # Compute determinant factor: sqrt(|g|) = Sigma * sin(theta)
    Sigma = r**2 + self.a**2 * np.cos(theta)**2
    sin_theta = np.sin(theta)
    sqrt_g = Sigma * sin_theta  # Known analytical expression for Kerr metric
    
    # Raise indices of f_cov: f^{ab} = g^{ac} g^{bd} f_{cd}
    f_contra = np.einsum('ac,bd,cd->ab', g_inv, g_inv, f_cov)
    
    # Define Levi-Civita symbol (permutation symbol)
    levicivita = np.zeros((4, 4, 4, 4))
    
    # Set non-zero components (±1 for even/odd permutations of [0,1,2,3])
    permutations = [
        (0,1,2,3, 1), (0,1,3,2, -1), (0,2,1,3, -1), (0,2,3,1, 1), 
        (0,3,1,2, 1), (0,3,2,1, -1), (1,0,2,3, -1), (1,0,3,2, 1),
        (1,2,0,3, 1), (1,2,3,0, -1), (1,3,0,2, -1), (1,3,2,0, 1),
        (2,0,1,3, 1), (2,0,3,1, -1), (2,1,0,3, -1), (2,1,3,0, 1),
        (2,3,0,1, 1), (2,3,1,0, -1), (3,0,1,2, -1), (3,0,2,1, 1),
        (3,1,0,2, 1), (3,1,2,0, -1), (3,2,0,1, -1), (3,2,1,0, 1)
    ]
    
    for mu, nu, rho, sigma, val in permutations:
        levicivita[mu, nu, rho, sigma] = val
    
    # Compute Y_{\mu\nu} = (1/2) * \epsilon_{\mu\nu\rho\sigma} f^{\rho\sigma}
    Y = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                for sigma in range(4):
                    # Use Levi-Civita tensor: \epsilon_{\mu\nu\rho\sigma} = sqrt_g * levicivita[\mu,\nu,\rho,\sigma]
                    Y[mu, nu] += 0.5 * sqrt_g * levicivita[mu, nu, rho, sigma] * f_contra[rho, sigma]

    return Y
        

def killing_tensor(self, r, theta):
    """Compute Killing tensor from Killing-Yano tensor"""
    Y = self.killing_yano_tensor(r, theta)
    g_inv = self.metric_inv(0, r, theta, 0)
    
    # Raise index: f^α_β = g^{αμ} f_{μβ}
    Y_raised = np.einsum('ab,cb->ca', g_inv, Y)
    
    # Compute Killing tensor: K_{μν} = f_{μα} f^α_ν
    K = np.einsum('ma,na->mn', Y, Y_raised)
    return K

def killing_vector_derivatives(r, theta, a):
    """
    Computes covariant derivatives of the t- and φ-Killing vectors in Kerr spacetime.
    
    Parameters
    ----------
    r : float
        Radial coordinate
    theta : float
        Polar angle in radians
    a : float
        Kerr spin parameter
        
    Returns
    -------
    tuple of numpy.ndarrays
        (covariant_derivative_t, covariant_derivative_phi) - both 4x4 matrices
    """
    # Precompute common terms
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    sin_2theta = sin(2 * theta)
    cos_2theta = cos(2 * theta)
    
    a2 = a**2
    r2 = r**2
    a2_cos2 = a2 * cos_theta**2
    denom = r2 + a2_cos2
    denom2 = denom**2
    r2_a2 = r2 + a2

    Dkphi_tr = -(r2 - a2_cos2)/denom2
    Dkphi_rt = -Dkphi_tr
    Dkphi_rphi = a*(a2_cos2 - r2)*sin_theta**2/denom2
    Dkphi_phir = -Dkphi_rphi
    Dkphi_ttheta = (a2 * r * sin_2theta)/denom2
    Dkphi_thetat = -Dkphi_ttheta 
    Dkphi_thetaphi = a*r*r2_a2*sin_2theta/denom2
    Dkphi_phitheta = -Dkphi_thetaphi  

    # Covariant derivative of t-Killing vector
    cov_t = np.array([
        [0, Dkphi_tr, Dkphi_ttheta, 0],
        [Dkphi_rt, 0, 0, Dkphi_rphi],
        [Dkphi_thetat, 0, 0, Dkphi_thetaphi],
        [0, Dkphi_phir, Dkphi_phitheta, 0]
    ])
    
    # Covariant derivative of φ-Killing vector
    term1 = -r * r2_a2 * (r2 + a2 * cos_2theta) * sin_theta**2
    term2 = (a2 * r * (r - a2) + a2**2 * (r - 1) * cos_theta**2) * sin_theta**4
    term3 = - term1
    term4 = - term2
    denom3 = a2 + 2*r2 + a2 * cos_2theta

    Dkt_tr = a*(r2 - a2_cos2)*sin_theta**2/denom2
    Dkt_rt = -Dkt_tr
    Dkt_ttheta = -a*r*r2_a2*sin_2theta/denom2
    Dkt_thetat = -Dkt_ttheta
    Dkt_rphi = (term1 + term2)/denom2
    Dkt_phir = -Dkt_rphi
    Dkt_thetaphi = -(a2 + (r - 2)*r)*cos_theta*sin_theta - (4*r*r2_a2**2*sin_2theta)/denom3**2
    Dkt_phitheta = -Dkt_thetaphi

    cov_phi = np.array([
        [0, Dkt_tr, Dkt_ttheta, 0],
        [Dkt_rt, 0, 0, Dkt_rphi],
        [Dkt_thetat, 0, 0, Dkt_thetaphi],
        [0, Dkt_phir, Dkt_phitheta, 0]
    ])
    
    return cov_t, cov_phi

def transform_tensor_to_z_coordinates(tensor, theta):
    """
    Transform a tensor from (t,r,θ,φ) coordinates to (t,r,z,φ) coordinates.
    
    Parameters:
        tensor: 4x4 tensor in (t,r,θ,φ) basis
        theta: current polar angle
        
    Returns:
        Tensor in (t,r,z,φ) basis
    """
    # Transformation matrix: ∂x^α/∂x^{μ'}
    # x_old = (t, r, θ, φ)
    # x_new = (t, r, z, φ)
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-10:
        # Avoid division by zero at poles
        sin_theta = 1e-10 if sin_theta >= 0 else -1e-10
    
    # Transformation matrix: ∂x_old/∂x_new
    # Note: z = cosθ ⇒ ∂θ/∂z = -1/sinθ
    J = np.array([
        [1, 0, 0, 0],                 # ∂t/∂t'
        [0, 1, 0, 0],                 # ∂r/∂r'
        [0, 0, -1/sin_theta, 0],     # ∂θ/∂z'
        [0, 0, 0, 1]                  # ∂φ/∂φ'
    ])
    
    # Tensor transformation: T_{μ'ν'} = J^α_{μ'} J^β_{ν'} T_{αβ}
    transformed = np.zeros_like(tensor)
    for mu in range(4):
        for nu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    transformed[mu, nu] += J[alpha, mu] * J[beta, nu] * tensor[alpha, beta]
    return transformed

def spin_correction_terms(lam, q, a, p, e, x, s, phi_s, theta_s, 
                          qr0=0, qz0=0, qpsi0=0, upsilon_r=None, upsilon_z=None):
    """
    Compute the spin-correction terms for energy and angular momentum.
    
    Parameters:
        lam (float): Mino time λ
        q (float): Mass ratio μ/M
        a (float): Black hole spin parameter
        p (float): Semi-latus rectum
        e (float): Orbital eccentricity
        x (float): Inclination parameter (cosι)
        s (float): Specific spin magnitude
        phi_s (float): Azimuthal angle for spin orientation
        theta_s (float): Polar angle for spin orientation
        qr0, qz0, qpsi0 (float): Initial phases (default 0)
        upsilon_r, upsilon_z (float): Radial and polar frequencies (optional)
        
    Returns:
        tuple: (correction_E, correction_Jz) - spin corrections for energy and angular momentum
    """
    # Compute frequencies if not provided
    if upsilon_r is None or upsilon_z is None:
        upsilon_r, upsilon_theta, _, _ = mino_frequencies(a, p, e, x)
    else:
        upsilon_theta = upsilon_z  # upsilon_z is the polar frequency
        
    # Compute orbital angles at Mino time λ
    chi_r = upsilon_r * lam + qr0
    chi_phi = upsilon_theta * lam + qz0
    
    # Compute position (r, θ)
    sinI = np.sqrt(1 - x**2)
    r = p / (1 + e * np.cos(chi_r))
    z = sinI * np.cos(chi_phi)
    theta = np.arccos(z)
    
    # Compute covariant derivatives of Killing vectors
    cov_t, cov_phi = killing_vector_derivatives(r, theta, a)
    
    # Get SPECIFIC spin tensor (s^{μν} = S^{μν}/μ)
    spin_tensor_value = spin_tensor(lam, q, a, p, e, x, s, phi_s, theta_s, chi_r, chi_phi)
    specific_spin_tensor = spin_tensor_value / q  # Convert to specific spin tensor

    # CORRECTED CONTRACTION: ξ_{μ;ν} s^{μν} with proper index order
    term_t = 0
    term_phi = 0
    for mu in range(4):
        for nu in range(4):
            # Note: s^{μν} is antisymmetric (s^{μν} = -s^{νμ})
            term_t += cov_t[mu, nu] * specific_spin_tensor[mu, nu]
            term_phi += cov_phi[mu, nu] * specific_spin_tensor[mu, nu]
    
    # CORRECTED SIGNS based on Eq.(5a)
    correction_E = 0.5 * term_t
    correction_Jz = -0.5 * term_phi  # Negative sign for angular momentum
    
    return correction_E, correction_Jz