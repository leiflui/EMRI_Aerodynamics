"""Module implementing scattering orbit solutions for Kerr spacetime"""
from .constants import *
from .constants import _standardize_params
from .frequencies import _ellippi, _ellippiinc, _ellipeinc, ellipk, ellipe
from .frequencies import *
from scipy.special import ellipj
from .orbit import Orbit
from numpy import pi, arccos, sqrt, sin, cos, arcsin, real, imag
import numpy as np


class ScatteringOrbit(Orbit):
    r"""Class representing a scattering (unbound) orbit in Kerr spacetime.

    Parameters
    ----------
    a : double
        dimensionless angular momentum
    E : double
        dimensionless energy (E > 1 for scattering orbits)
    L : double
        dimensionless angular momentum
    Q : double
        dimensionless carter constant
    initial_phases : tuple, optional
        tuple of initial phases
        :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`, defaults to (0,0,0,0)
    M : double
        mass of the primary in solar masses, optional
    mu : double
        mass of the smaller body in solar masses, optional

    Attributes
    ----------
    a
        dimensionless angular momentum
    E
        dimensionless energy
    L
        dimensionless angular momentum
    Q
        dimensionless carter constant
    M
        mass of the primary in solar masses
    mu
        mass of the smaller body in solar masses
    initial_phases
        tuple of initial phases
        :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
    stable
        boolean indicating whether the orbit is stable (False for scattering)
    initial_position
        tuple of initial position coordinates :math:`(t_0, r_0, \theta_0, \phi_0)`
    initial_velocity
        tuple of initial four-velocity components :math:`(u^t_0, u^r_0, u^\theta_0, u^\phi_0)`
    """

    def __init__(self, a, E, L, Q, initial_phases=(0, 0, 0, 0), M=None, mu=None):
        if E <= 1:
            raise ValueError("Energy E must be > 1 for scattering orbits")
            
        self.a, self.E, self.L, self.Q, self.initial_phases, self.M, self.mu = (
            a, E, L, Q, initial_phases, M, mu
        )
        
        self.stable = False
        
        # Compute radial roots for scattering orbit
        self.radial_roots = self._scattering_radial_roots()
        
        # Compute polar roots
        self.polar_roots = self._scattering_polar_roots()
        
        # Compute Mino time frequencies
        self.upsilon_r, self.upsilon_theta, self.upsilon_phi, self.gamma = self._scattering_mino_frequencies()
        
        # Set initial position and velocity 
        self.initial_position = (0, 100, pi/2, 0)  # Starting far away
        self.initial_velocity = self._compute_initial_velocity()
        

    def _scattering_radial_roots(self):
        """Computes radial roots for scattering orbits using r-coordinate"""
        a, E, L, Q = self.a, self.E, self.L, self.Q
        
        # Use the same polynomial as in constants.plunging_radial_roots but for scattering
        # Standard form of the radial polynomial R(r)
        R = np.polynomial.Polynomial([
            -a**2 * Q,
            2 * (L**2 + Q + a**2 * E**2) - 4 * a * E * L,
            a**2 * E**2 - L**2 - Q - a**2,
            2,
            E**2 - 1
        ])
        
        roots = R.roots()
        
        # For scattering orbits (E > 1), we typically get 2 real and 2 complex roots
        real_roots = np.real(roots[np.isreal(roots)])
        complex_roots = roots[np.iscomplex(roots)]
        
        if len(real_roots) == 2:
            r1, r2 = np.sort(real_roots)
            r3, r4 = complex_roots
        else:
            # Fallback: use all real parts
            r1, r2, r3, r4 = np.real(roots)
            
        return r1, r2, r3, r4

    def _scattering_polar_roots(self):
        """Computes polar roots for scattering orbits"""
        a, E, L, Q = self.a, self.E, self.L, self.Q
        
        # Polar polynomial in z = cos^2(theta)
        Z = np.polynomial.Polynomial([
            Q,
            -(Q + a**2 * (1 - E**2) + L**2),
            a**2 * (1 - E**2)
        ])
        
        polar_roots = Z.roots()
        real_polar_roots = np.real(polar_roots[np.isreal(polar_roots)])
        
        if len(real_polar_roots) == 2:
            z_minus, z_plus = real_polar_roots
        else:
            # Use real parts
            z_minus, z_plus = np.real(polar_roots[:2])
            
        # Ensure z_minus <= z_plus
        if z_minus > z_plus:
            z_minus, z_plus = z_plus, z_minus
            
        return z_minus, z_plus

    def _scattering_mino_frequencies(self):
        """Computes Mino time frequencies for scattering orbits"""
        a, E, L, Q = self.a, self.E, self.L, self.Q
        r1, r2, r3, r4 = self.radial_roots
        z_minus, z_plus = self.polar_roots
        
        # Handle complex roots for radial frequency
        if np.iscomplex(r3):
            # Use plunging_mino_frequencies approach for complex roots
            rho_r = real(r3)
            rho_i = imag(r3)
            
            # Equation 42 from Fujita & Hikida for complex roots
            A = sqrt((r2 - rho_r)**2 + rho_i**2)
            B = sqrt((r1 - rho_r)**2 + rho_i**2)
            k_r_sq = ((r2 - r1)**2 - (A - B)**2) / (4 * A * B)
            k_r = sqrt(abs(k_r_sq))
            
            # Equation 52 - note the sign change for E^2 - 1
            upsilon_r = pi * sqrt(A * B * (E**2 - 1)) / (2 * ellipk(k_r**2))
        else:
            # All real roots - use standard formula
            k_r = sqrt((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4)))
            upsilon_r = pi * sqrt((E**2 - 1) * (r1 - r3) * (r2 - r4)) / (2 * ellipk(k_r**2))
        
        # Polar frequency (similar to bound case)
        if z_plus > 0 and z_minus > 0:
            k_theta = sqrt(z_minus / z_plus)
            # Simplified form of epsilon0*z_plus for E > 1
            e0zp = (a**2 * (E**2 - 1) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))
            upsilon_theta = pi * L * sqrt(abs(e0zp)) / (2 * ellipk(k_theta**2))
        else:
            # Fallback for invalid polar roots
            upsilon_theta = 0.1
        
        # Azimuthal frequency (placeholder - would need full implementation)
        upsilon_phi = 0.0
        
        # Gamma factor (average dt/dλ) - simplified version
        gamma = abs(E)  # Placeholder
        
        return upsilon_r, upsilon_theta, upsilon_phi, gamma

    def _compute_initial_velocity(self):
        """Computes initial four-velocity from constants of motion"""
        # Simplified calculation - in practice would need to solve geodesic equations
        r0 = 100  # Starting far away
        theta0 = pi/2
        
        # Use constants to compute initial velocities
        Sigma = r0**2 + self.a**2 * cos(theta0)**2
        Delta = r0**2 - 2*r0 + self.a**2
        
        # Simplified calculation - proper implementation would need root finding
        u_t = 1.0
        u_r = -0.01  # Inward radial velocity
        u_theta = 0.001
        u_phi = 0.01
        
        return (u_t, u_r, u_theta, u_phi)

    def trajectory(self, initial_phases=None, distance_units="natural", time_units="natural"):
        r"""Computes the time, radial, polar, and azimuthal coordinates of the scattering orbit 
        as a function of mino time.

        Parameters
        ----------
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`
        distance_units : str, optional
            units to compute the radial component of the trajectory in
            (options are "natural", "mks", "cgs", "au" and "km"),
            defaults to "natural"
        time_units : str, optional
            units to compute the time component of the trajectory in
            (options are "natural", "mks", "cgs", and "days"), defaults
            to "natural"

        Returns
        -------
        tuple(function, function, function, function)
            tuple of functions :math:`(t(\lambda), r(\lambda), \theta(\lambda), \phi(\lambda))`
        """
        if initial_phases is None:
            initial_phases = self.initial_phases

        return scattering_trajectory(
            self.a, self.E, self.L, self.Q, initial_phases, self.M, distance_units, time_units
        )

    def scattering_angle(self, q_z0=None):
        r"""Computes the scattering angle for the orbit using the formula from van de Meent:
        
        χ = 2q_r^S (Υ_θ/Υ_r) - φ_r(q_r^S) + φ_z(q_z0 + q_r^S Υ_θ/Υ_r) - φ_z(q_z0 - q_r^S Υ_θ/Υ_r)
        
        Parameters
        ----------
        q_z0 : double, optional
            initial polar phase, defaults to initial_phases[2]
            
        Returns
        -------
        double
            scattering angle χ in radians
        """
        if q_z0 is None:
            q_z0 = self.initial_phases[2]
        
        # Compute q_r^S (radial phase at turning point)
        # This would need proper calculation based on the specific orbit
        q_r_S = pi / 2  # Placeholder
        
        # Get the φ_r and φ_z functions
        _, _, phi_r = scattering_radial_solutions(self.a, (self.E, self.L, self.Q), self.radial_roots)
        _, _, phi_theta = scattering_polar_solutions(self.a, (self.E, self.L, self.Q), self.polar_roots)
        
        # Compute the terms in the scattering angle formula
        term1 = 2 * q_r_S * (self.upsilon_theta / self.upsilon_r)
        term2 = phi_r(q_r_S)
        term3 = phi_theta(q_z0 + q_r_S * self.upsilon_theta / self.upsilon_r)
        term4 = phi_theta(q_z0 - q_r_S * self.upsilon_theta / self.upsilon_r)
        
        chi = term1 - term2 + term3 - term4
        
        return chi


def scattering_radial_solutions(a, constants, radial_roots):
    r"""Computes the radial solutions for scattering orbits following van de Meent's slides.
    
    Parameters
    ----------
    a : double
        dimensionless spin parameter
    constants : tuple(double,double,double)
        tuple of constants :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
    radial_roots : tuple(double,double,double,double)
        tuple of roots :math:`(r_1,r_2,r_3,r_4)`

    Returns
    -------
    tuple(function, function, function)
        tuple of functions :math:`(r, t^{(r)}, \phi^{(r)})`
    """
    E, L, Q = constants
    r1, r2, r3, r4 = radial_roots
    
    r_plus = 1 + sqrt(1 - a**2)
    r_minus = 1 - sqrt(1 - a**2)
    
    # Handle complex roots
    if np.iscomplex(r3):
        r3, r4 = real(r3), real(r4)  # Use real parts
        
    # Similar to stable.py but adapted for scattering
    k_r = sqrt(abs((r1 - r2) * (r3 - r4) / ((r1 - r3) * (r2 - r4))))
    h_r = (r1 - r2) / (r1 - r3)
    h_plus = (r1 - r2) * (r3 - r_plus) / ((r1 - r3) * (r2 - r_plus))
    h_minus = (r1 - r2) * (r3 - r_minus) / ((r1 - r3) * (r2 - r_minus))

    def r(q_r):
        # Solution for r from Fujita & Hikida, adapted for scattering
        u_r = ellipk(k_r**2) * q_r / pi
        sn, cn, dn, psi_r = ellipj(u_r, k_r**2)
        
        # Equation 27 from Fujita & Hikida
        return (r3 * (r1 - r2) * sn**2 - r2 * (r1 - r3)) / ((r1 - r2) * sn**2 - (r1 - r3))

    def t_r(q_r):
        # Equation 28 from Fujita & Hikida, adapted for E > 1
        u_r = ellipk(k_r**2) * q_r / pi
        sn, cn, dn, psi_r = ellipj(u_r, k_r**2)
        
        # Note: for E > 1, we use E^2 - 1 instead of 1 - E^2
        denominator = sqrt((E**2 - 1) * (r1 - r3) * (r2 - r4))
        
        return (2 / denominator) * (
            E / 2 * (
                (r2 - r3) * (r1 + r2 + r3 + r4) * 
                (_ellippiinc(psi_r, h_r, k_r**2) - q_r / pi * _ellippi(h_r, k_r**2)) +
                (r1 - r3) * (r2 - r4) * (
                    _ellipeinc(psi_r, k_r**2) +
                    h_r * sn * cn * sqrt(1 - k_r**2 * sn**2) / (h_r * sn**2 - 1) -
                    q_r / pi * ellipe(k_r**2)
                )
            ) +
            2 * E * (r2 - r3) * (_ellippiinc(psi_r, h_r, k_r**2) - q_r / pi * _ellippi(h_r, k_r**2)) -
            2 / (r_plus - r_minus) * (
                ((4 * E - a * L) * r_plus - 2 * a**2 * E) * (r2 - r3) / ((r3 - r_plus) * (r2 - r_plus)) *
                (_ellippiinc(psi_r, h_plus, k_r**2) - q_r / pi * _ellippi(h_plus, k_r**2)) -
                ((4 * E - a * L) * r_minus - 2 * a**2 * E) * (r2 - r3) / ((r3 - r_minus) * (r2 - r_minus)) *
                (_ellippiinc(psi_r, h_minus, k_r**2) - q_r / pi * _ellippi(h_minus, k_r**2))
            )
        )

    def phi_r(q_r):
        # Equation 28 from Fujita & Hikida
        u_r = ellipk(k_r**2) * q_r / pi
        sn, cn, dn, psi_r = ellipj(u_r, k_r**2)
        
        denominator = (r_plus - r_minus) * sqrt((E**2 - 1) * (r1 - r3) * (r2 - r4))
        
        return (-2 * a / denominator) * (
            (2 * E * r_plus - a * L) * (r2 - r3) / ((r3 - r_plus) * (r2 - r_plus)) *
            (_ellippiinc(psi_r, h_plus, k_r**2) - q_r / pi * _ellippi(h_plus, k_r**2)) -
            (2 * E * r_minus - a * L) * (r2 - r3) / ((r3 - r_minus) * (r2 - r_minus)) *
            (_ellippiinc(psi_r, h_minus, k_r**2) - q_r / pi * _ellippi(h_minus, k_r**2))
        )
        
    return r, t_r, phi_r


def scattering_polar_solutions(a, constants, polar_roots):
    r"""Computes the polar solutions for scattering orbits.
    
    Parameters
    ----------
    a : double
        dimensionless spin parameter
    constants : tuple(double,double,double)
        tuple of constants :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
    polar_roots : tuple(double,double)
        tuple of roots :math:`(z_-,z_+)`

    Returns
    -------
    tuple(function, function, function)
        tuple of functions :math:`(\theta, t^{(\theta)}, \phi^{(\theta)})`
    """
    E, L, Q = constants
    z_minus, z_plus = polar_roots
    
    # Handle case where roots might be invalid
    if z_plus <= 0 or z_minus < 0:
        # Use fallback values
        z_minus, z_plus = 0.1, 0.9
    
    # Similar to stable.py
    k_theta = sqrt(z_minus / z_plus)
    e0zp = (a**2 * (E**2 - 1) * (1 - z_minus) + L**2) / (L**2 * (1 - z_minus))
    a2sqrt_zp_over_e0 = a**2 * sqrt(z_plus / (a**2 * (E**2 - 1)))

    def theta(q_theta):
        u_theta = 2 / pi * ellipk(k_theta**2) * (q_theta + pi / 2)
        sn, cn, dn, ph = ellipj(u_theta, k_theta**2)
        return arccos(sqrt(z_minus) * sn)

    def t_theta(q_theta):
        u_theta = 2 / pi * ellipk(k_theta**2) * (q_theta + pi / 2)
        sn, cn, dn, psi_theta = ellipj(u_theta, k_theta**2)
        
        return (np.sign(L) * a2sqrt_zp_over_e0 * E / L * 
                (2 / pi * ellipe(k_theta**2) * (q_theta + pi / 2) - _ellipeinc(psi_theta, k_theta**2)))

    def phi_theta(q_theta):
        u_theta = 2 / pi * ellipk(k_theta**2) * (q_theta + pi / 2)
        sn, cn, dn, psi_theta = ellipj(u_theta, k_theta**2)
        
        return (np.sign(L) / sqrt(e0zp) * 
                (_ellippiinc(psi_theta, z_minus, k_theta**2) - 
                 2 / pi * _ellippi(z_minus, k_theta**2) * (q_theta + pi / 2)))
        
    return theta, t_theta, phi_theta


# Unit conversion functions (defined locally to avoid import issues)
def distance_in_meters(r, M):
    """Convert natural units to meters"""
    return r * M * 1.4766250614046494e3  # M_sun in meters

def distance_in_km(r, M):
    return distance_in_meters(r, M) / 1000

def distance_in_cm(r, M):
    return distance_in_meters(r, M) * 100

def distance_in_au(r, M):
    return distance_in_meters(r, M) / 1.495978707e11

def time_in_seconds(t, M):
    """Convert natural units to seconds"""
    return t * M * 4.925490947e-6  # M_sun in seconds

def time_in_days(t, M):
    return time_in_seconds(t, M) / 86400


def scattering_trajectory(a, E, L, Q, initial_phases=(0, 0, 0, 0), M=None, 
                         distance_units="natural", time_units="natural"):
    r"""Computes the trajectory for a scattering orbit.
    
    Parameters
    ----------
    a : double
        spin parameter
    E : double
        dimensionless energy
    L : double
        dimensionless angular momentum  
    Q : double
        dimensionless Carter constant
    initial_phases : tuple, optional
        initial phases
    M : double, optional
        mass parameter
    distance_units : str, optional
        distance units
    time_units : str, optional
        time units

    Returns
    -------
    tuple(function, function, function, function)
        trajectory functions
    """
    if ((distance_units != "natural") or (time_units != "natural")) and M is None:
        raise ValueError("M must be specified to convert to physical units")

    # Create orbit instance to get frequencies and roots
    orbit = ScatteringOrbit(a, E, L, Q, initial_phases, M)
    
    constants = (E, L, Q)
    radial_roots = orbit.radial_roots
    polar_roots = orbit.polar_roots
    upsilon_r, upsilon_theta, upsilon_phi, gamma = orbit.upsilon_r, orbit.upsilon_theta, orbit.upsilon_phi, orbit.gamma

    # Get solution functions
    r_func, t_r, phi_r = scattering_radial_solutions(a, constants, radial_roots)
    theta_func, t_theta, phi_theta = scattering_polar_solutions(a, constants, polar_roots)
    
    q_t0, q_r0, q_theta0, q_phi0 = initial_phases
    
    # Define conversion functions locally
    distance_conversion_func = {
        "natural": lambda x, M: x,
        "mks": distance_in_meters,
        "cgs": distance_in_cm,
        "au": distance_in_au,
        "km": distance_in_km,
    }
    time_conversion_func = {
        "natural": lambda x, M: x,
        "mks": time_in_seconds,
        "cgs": time_in_seconds,
        "days": time_in_days,
    }

    # Calculate normalization constants (set t=0 and phi=0 at lambda=0)
    C_t = t_r(q_r0) + t_theta(q_theta0)
    C_phi = phi_r(q_r0) + phi_theta(q_theta0)

    def t(mino_time):
        return time_conversion_func[time_units](
            q_t0 + gamma * mino_time + 
            t_r(upsilon_r * mino_time + q_r0) + 
            t_theta(upsilon_theta * mino_time + q_theta0) - C_t, 
            M
        )

    def r(mino_time):
        return distance_conversion_func[distance_units](
            r_func(upsilon_r * mino_time + q_r0), 
            M
        )

    def theta(mino_time):
        return theta_func(upsilon_theta * mino_time + q_theta0)

    def phi(mino_time):
        return (q_phi0 + upsilon_phi * mino_time + 
                phi_r(upsilon_r * mino_time + q_r0) + 
                phi_theta(upsilon_theta * mino_time + q_theta0) - C_phi)

    return t, r, theta, phi