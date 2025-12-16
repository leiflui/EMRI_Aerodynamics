"""Module containing the KerrSpacetime class"""
from .constants import *
from numpy import cos, sin
import numpy as np


class KerrSpacetime:
    """Class representing spacetime around a black hole with mass :math:`M` 
    and spin parameter :math:`a`

    Parameters
    ----------
    a : double
        dimensionless angular momentum
    M : double
        mass of the black hole

    Attributes
    ----------
    a
        dimensionless angular momentum
    M
        mass of the black hole
    inner_horizon
        radius of the inner horizon
    outer_horizon
        radius of the outer horizon
    """

    def __init__(self, a, M=None):
        self.a, self.M = a, M
        self.inner_horizon = 1 - sqrt(1 - self.a**2)
        self.outer_horizon = 1 + sqrt(1 - self.a**2)

    def separatrix(self, e, x):
        """Computes the value of p at the separatrix for a given value of e and x

        Parameters
        ----------
        e : double
            orbital eccentricity
        x : double
            cosine of the orbital inclination

        Returns
        -------
        double
        """
        return separatrix(self.a, e, x)

    def is_stable(self, p, e, x):
        """Determines whether a given orbit is stable or not

        Parameters
        ----------
        p : double
            dimensionless semi-latus rectum
        e : double
            orbital eccentricity
        x : double
            cosine of the orbital inclination

        Returns
        -------
        bool
        """
        return is_stable(self.a, p, e, x)

    def metric(self, t, r, theta, phi):
        """Returns the matrix representation of the metric at a given point 
        expressed in Boyer-Lindquist coordinates.

        Parameters
        ----------
        t : double
            time coordinate
        r : double
            radial coordinate
        theta : double
            polar coordinate
        phi : double
            azimuthal coordinate

        Returns
        -------
        numpy.ndarray
        """
        a = self.a
        sigma = r**2 + a**2 * cos(theta) ** 2
        delta = r**2 - 2 * r + a**2
        # fmt: off
        return np.array(
            [[-(1-2*r/sigma),               0,              0,      -2*a*r*sin(theta)**2/sigma],
             [0,                            sigma/delta,    0,      0],
             [0,                            0,              sigma,  0],
             [-2*a*r*sin(theta)**2/sigma,   0,              0,      sin(theta)**2*(r**2+a**2+2*a**2*r*sin(theta)**2/sigma)]
             ]
        )
        # fmt: on
    
    def metric_inv(self, t, r, theta, phi):
        """Returns the matrix representation of the metric at a given point 
        expressed in Boyer-Lindquist coordinates.

        Parameters
        ----------
        t : double
            time coordinate
        r : double
            radial coordinate
        theta : double
            polar coordinate
        phi : double
            azimuthal coordinate

        Returns
        -------
        numpy.ndarray
        """
        a = self.a
        sigma = r**2 + a**2 * cos(theta) ** 2
        delta = r**2 - 2 * r + a**2
        # fmt: off
        return np.array(
            [[-1/delta*(r**2 + a**2 + 2*r*a**2/sigma*sin(theta)**2),               0,              0,      -2*a*r/(delta*sigma)],
             [0,                            delta/sigma,    0,      0],
             [0,                            0,              1/sigma,  0],
             [-2*a*r/(delta*sigma),   0,              0,      (delta - a**2*sin(theta)**2)/(sigma*delta*sin(theta)**2)]
             ]
        )
        # fmt: on

    def norm(self, t, r, theta, phi, v):
        """Computes the norm of a vector at a given point in spacetime 
        expressed in Boyer-Lindquist coordinates

        Parameters
        ----------
        t : double
            time coordinate
        r : double
            radial coordinate
        theta : double
            polar coordinate
        phi : double
            azimuthal coordinate
        v : array_like
            vector to compute the norm of

        Returns
        -------
        double
            norm of v
        """
        return np.dot(v, np.dot(self.metric(t, r, theta, phi), v))
    def kerr_riemann(M, a, r, theta):
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
            R[0, 0, 1, 2] = (4*a**2*M**2*r*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                            + a**2*np.cos(2*theta))**3)
            R[0, 1, 0, 1] = (M*r*(-27*a**4 - 4*a**2*r**2 + 16*r**4 - 4*a**2*(6*a**2 + 7*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/((a**2 
                            + r*(-2*M + r))*(a**2 + 2*r**2  + a**2*np.cos(2*theta))**3)
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
            R[1, 0, 1, 3] = -((8*a*M*r*(3*a**2 + r*(-4*M + 3*r))*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 
                            + a**2*np.cos(2*theta))**4)
            R[1, 0, 2, 3] = (a*M*(a**2 + r*(-2*M + r))*(9*a**4 - 52*a**2*r**2 - 48*r**4 + 4*a**2*(2*a**2 + 5*r**2)*np.cos(2*theta) 
                            - a**4*np.cos(4*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
            R[1, 2, 0, 3] = (2*a*M*(a**2 + r*(-2*M + r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3
            R[1, 2, 1, 2] = (M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/(2*(r**2 + a**2*np.cos(theta)**2)**2)
            R[1, 3, 1, 3] = (8*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*(2*a**4 + r**4 + a**2*r*(-2*M + 3*r) - a**2*(a**2 + r*(-2*M 
                            + r))*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
            R[1, 3, 2, 3] = -((24*a**2*M*(a**2 + r**2)*(a**2 + r*(-2*M + r))*np.cos(theta)*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(theta)**3)/(a**2 
                            + 2*r**2 + a**2*np.cos(2*theta))**4)
            R[2, 0, 0, 1] = (12*a**2*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
            R[2, 0, 0, 2] = (8*M*r*(-3*a**2 + 2*r**2 - 3*a**2*np.cos(2*theta))*(-2*a**2 + (2*M - r)*r + a**2*np.cos(2*theta)))/(a**2 + 2*r**2 
                            + a**2*np.cos(2*theta))**4
            R[2, 0, 1, 3] = -((2*a*M*(-3*a**4 + 22*a**2*r**2 + 12*r**4 - 2*a**2*(a**2 + 7*r**2)*np.cos(2*theta) + a**4*np.cos(4*theta))*np.sin(2*theta))/(a**2 
                            + 2*r**2 + a**2*np.cos(2*theta))**4)
            R[2, 0, 2, 3] = (8*a*M*r*(3*a**2 + r*(-2*M + 3*r))*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2
                            + a**2*np.cos(2*theta))**4
            R[2, 1, 0, 3] =-((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3)
            R[2, 1, 1, 2] = -((M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/(2*(a**2 + r*(-2*M + r))*(r**2 + a**2*np.cos(theta)**2)**2))
            R[2, 3, 0, 2] = -((8*a*M*r*(3*a**2 + r*(-2*M + 3*r))*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 
                            + a**2*np.cos(2*theta))**4)
            R[2, 3, 1, 3] = -((24*a**2*M*(a**2 + r**2)*np.cos(theta)*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(theta)**3)/(a**2 + 2*r**2 
                            + a**2*np.cos(2*theta))**4)
            R[2, 3, 2, 3] = -((4*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*(5*a**4 + 4*r**4 + a**2*r*(-2*M + 9*r) - a**2*(a**2 + r*(-2*M 
                            + r))*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4)
            R[3, 0, 0, 3] = (4*M*r*(-3*a**2 + 2*r**2 - 3*a**2*np.cos(2*theta))*(-a**2 + 4*M*r - 2*r**2 - a**2*np.cos(2*theta)))/(a**2 + 2*r**2 
                            + a**2*np.cos(2*theta))**4
            R[3, 0, 1, 2] = -((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(a**2 + 2*r*(-2*M + r) + a**2*np.cos(2*theta))*cot_theta )/((a**2 + r*(-2*M 
                            + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3))
            R[3, 1, 0, 1] = -((12*a*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3))
            R[3, 1, 0, 2] = -((4*a*M*(-2*a**2 + (2*M - r)*r + a**2*np.cos(2*theta))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*cot_theta )/((a**2 + r*(-2*M 
                            + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3))
            R[3, 1, 1, 3] = (2*M*r*(-9*a**4 + 2*a**2*r**2 + 4*r**4 - 2*a**2*(3*a**2 + 5*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/((a**2 + r*(-2*M 
                            + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3)
            R[3, 1, 2, 3] = (2*a**2*M*(3*a**2 + r*(-2*M + 3*r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                            + a**2*np.cos(2*theta))**3)
            R[3, 2, 0, 1] = -((2*a*M*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*(-5*a**2 + 8*M*r - 4*r**2 + a**2*np.cos(2*theta))*cot_theta )/((a**2 + r*(-2*M 
                            + r))*(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3))
            R[3, 2, 0, 2] = (12*a*M*r*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta)))/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**3
            R[3, 2, 1, 3] = (2*a**2*M*(3*a**2 + r*(-4*M + 3*r))*(a**2 - 6*r**2 + a**2*np.cos(2*theta))*np.sin(2*theta))/((a**2 + r*(-2*M + r))*(a**2 + 2*r**2 
                            + a**2*np.cos(2*theta))**3)
            R[3, 2, 2, 3] = -((M*r*(-27*a**4 - 4*a**2*r**2 + 16*r**4 - 4*a**2*(6*a**2 + 7*r**2)*np.cos(2*theta) + 3*a**4*np.cos(4*theta)))/(a**2 + 2*r**2 
                            + a**2*np.cos(2*theta))**3)
            R[3, 3, 0, 3] = (16*a*M**2*r**2*(3*a**2 - 2*r**2 + 3*a**2*np.cos(2*theta))*np.sin(theta)**2)/(a**2 + 2*r**2 + a**2*np.cos(2*theta))**4
            
        
            #Non-zero components by symmetry of Riemann tensor
            R[0, 0, 1, 2] = - R[0, 0, 1, 2]
            R[0, 0, 3, 0] = - R[0, 0, 0, 3]
            R[0, 1, 2, 0] = - R[0, 1, 0, 2]  
            R[0, 1, 3, 1] = - R[0, 1, 1, 3] 
            R[0, 1, 3, 2] = - R[0, 1, 2, 3]
            R[0, 2, 0, 1] = - R[0, 0, 1, 2] - R[0, 1, 2, 0]
            R[0, 2, 1, 0] = - R[0, 2, 0, 1]
            R[0, 2, 2, 0] = - R[0, 2, 0, 2]
            R[0, 2, 3, 1] = - R[0, 2, 1, 3]
            R[0, 2, 3, 2] = - R[0, 2, 2, 3]
            R[0, 3, 1, 2] = - R[0, 1, 2, 3] - R[0, 2, 1, 3]
            R[0, 3, 1, 2] = - R[0, 3, 2, 1]
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
            R[3, 3, 2, 1] = R[3, 3, 1, 2]
            R[3, 3, 3, 0] = - R[3, 3, 0, 3] - R[3, 0, 3, 3]    
            return R

    def four_velocity(self, t, r, theta, phi, constants, upsilon_r, upsilon_theta, initial_phases):
        r"""Computes the four velocity of a given trajectory

        Parameters
        ----------
        t : function
            time component of trajectory
        r : function
            radial component of trajectory
        theta : function
            polar component of trajectory
        phi : function
            azimuthal component of trajectory
        constants : tuple(double, double, double)
            tuple of constants of motion in the form 
            :math:`(\mathcal{E},\mathcal{L},\mathcal{Q})`
        upsilon_r : double
            radial frequency
        upsilon_theta : double
            polar frequency
        initial_phases : tuple, optional
            tuple of initial phases
            :math:`(q_{t_0},q_{r_0},q_{\theta_0},q_{\phi_0})`

        Returns
        -------
        tuple(function, function, function, function)
            components of the four velocity (i.e.
            :math:`(u^t,u^r,u^\theta,u^\phi)`)
        """
        a = self.a
        E, L, Q = constants
        q_t0, q_r0, q_theta0, q_phi0 = initial_phases

        # radial polynomial
        R = lambda r: (E * (r**2 + a**2) - a * L) ** 2 - (r**2 - 2 * r + a**2) * (
            r**2 + (a * E - L) ** 2 + Q
        )
        # polar polynomial
        Z = (
            lambda z: Q
            - (Q + a**2 * (1 - E**2) + L**2) * z**2
            + a**2 * (1 - E**2) * z**4
        )
        
        def u_t(time):
            delta = r(time) ** 2 - 2 * r(time) + a**2
            sigma = r(time) ** 2 + a**2 * cos(theta(time)) ** 2
            return (
                1
                / sigma
                * (
                    (r(time) ** 2 + a**2) / delta * (E * (r(time) ** 2 + a**2) - a * L)
                    - a**2 * E * (1 - cos(theta(time)) ** 2)
                    + a * L
                )
            )

        def u_r(time):
            q_r = upsilon_r * time + q_r0
            sigma = r(time) ** 2 + a**2 * cos(theta(time)) ** 2
            return np.copysign(1, sin(q_r)) * sqrt(abs(R(r(time)))) / sigma

        def u_theta(time):
            q_theta = upsilon_theta * time + q_theta0
            sigma = r(time) ** 2 + a**2 * cos(theta(time)) ** 2
            return (
                np.copysign(1, sin(q_theta))
                * sqrt(abs(Z(cos(theta(time)))))
                / (sigma * sin(theta(time)))
            )

        def u_phi(time):
            sigma = r(time) ** 2 + a**2 * cos(theta(time)) ** 2
            delta = r(time) ** 2 - 2 * r(time) + a**2
            return (
                1
                / sigma
                * (
                    a / delta * (E * (r(time) ** 2 + a**2) - a * L)
                    + L / (1 - cos(theta(time)) ** 2)
                    - a * E
                )
            )

        return u_t, u_r, u_theta, u_phi

    def killing_vectors(self, r, theta):
        """Compute time-like and axial Killing vectors (contravariant)"""
        xi_t = np.array([1.0, 0.0, 0.0, 0.0])  # ∂/∂t
        xi_phi = np.array([0.0, 0.0, 0.0, 1.0])  # ∂/∂φ
        return xi_t, xi_phi
    
    def killing_vectors_covariant(self, r, theta):
        """Compute covariant forms of Killing vectors"""
        xi_t, xi_phi = self.killing_vectors(r, theta)
        g = self.metric(0, r, theta, 0)
        xi_t_cov = np.dot(g, xi_t)
        xi_phi_cov = np.dot(g, xi_phi)
        return xi_t_cov, xi_phi_cov
    
    def killing_yano_tensor(self, r, theta):
        """
        Compute the Killing-Yano tensor Y in Kerr spacetime.
        
        Based on Eq. (4) and the definition Y_{μν} = (1/2) ε_{μνρσ} f^{ρσ}
        but we can use the known analytic form.
        """
        a = self.a
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        Y = np.zeros((4, 4))
        
        # Y_{tr} = -a cosθ
        Y[0, 1] = -a * cos_theta
        Y[1, 0] = a * cos_theta
        
        # Y_{tθ} = a r sinθ
        Y[0, 2] = a * r * sin_theta
        Y[2, 0] = -a * r * sin_theta
        
        # Y_{rφ} = -a^2 cosθ sin^2θ
        Y[1, 3] = -a**2 * cos_theta * sin_theta**2
        Y[3, 1] = a**2 * cos_theta * sin_theta**2
        
        # Y_{θφ} = r sinθ (r^2 + a^2)
        Y[2, 3] = r * sin_theta * (r**2 + a**2)
        Y[3, 2] = -r * sin_theta * (r**2 + a**2)
        
        return Y
    
    def killing_yano_tensor_derivative(self, r, theta):
        """
        Compute the covariant derivative of the Killing-Yano tensor Y.
        
        Using the relation: ∇_α Y_{μν} = -ε_{μνασ} ξ^σ
        where ξ is the timelike Killing vector (∂/∂t).
        
        Args:
            r (float): Radial coordinate
            theta (float): Angular coordinate (in radians)
            
        Returns:
            numpy.ndarray: Covariant derivative ∇Y (4×4×4, indices: [α, μ, ν])
        """
        # Get the timelike Killing vector (contravariant)
        xi_t, _ = self.killing_vectors(r, theta)
        
        # Compute sqrt(-g) = Σ sinθ
        a = self.a
        Sigma = r**2 + a**2 * np.cos(theta)**2
        sin_theta = np.sin(theta)
        sqrt_g = Sigma * sin_theta
        
        # Define Levi-Civita symbol
        levicivita = np.zeros((4, 4, 4, 4))
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
        
        # Initialize the covariant derivative tensor
        nabla_Y = np.zeros((4, 4, 4))
        
        # Compute using the formula: ∇_α Y_{μν} = -ε_{μνασ} ξ^σ
        for alpha in range(4):
            for mu in range(4):
                for nu in range(4):
                    for sigma in range(4):
                        epsilon_munu_alpha_sigma = sqrt_g * levicivita[mu, nu, alpha, sigma]
                        nabla_Y[alpha, mu, nu] -= epsilon_munu_alpha_sigma * xi_t[sigma]
        
        return nabla_Y
    
    def killing_tensor(self, r, theta):
        """Compute Killing tensor from Killing-Yano tensor: K_{μν} = Y_{μρ} Y^ρ_ν"""
        # Get Killing-Yano tensor
        Y = self.killing_yano_tensor(r, theta)
        
        # Get inverse metric
        g_inv = self.metric_inv(0, r, theta, 0)  # t and phi can be set to 0
        
        # Raise index: Y^ρ_ν = g^{ρσ} Y_{σν}
        Y_rho_nu = np.einsum('rs,sn->rn', g_inv, Y)
        
        # Compute Killing tensor: K_{μν} = Y_{μρ} Y^ρ_ν
        K = np.einsum('mr,rn->mn', Y, Y_rho_nu)
        
        return -K

        
    def killing_vector_derivatives(self, r, theta):
        """
        Computes covariant derivatives of the t- and φ-Killing vectors in Kerr spacetime.
        
        Parameters:
            r (float): Radial coordinate
            theta (float): Polar angle in radians
            
        Returns:
            tuple: (covariant_derivative_t, covariant_derivative_phi) - both 4x4 matrices
        """
        a = self.a
        M = self.M if self.M is not None else 1.0
        
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
    
        # Components for t-Killing vector derivative
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
        
        # Components for φ-Killing vector derivative
        term1 = -r * r2_a2 * (r2 + a2 * cos_2theta) * sin_theta**2
        term2 = (a2 * r * (r - a2) + a2**2 * (r - M) * cos_theta**2) * sin_theta**4
        denom3 = a2 + 2*r2 + a2 * cos_2theta
    
        Dkt_tr = a*(r2 - a2_cos2)*sin_theta**2/denom2
        Dkt_rt = -Dkt_tr
        Dkt_ttheta = -a*r*r2_a2*sin_2theta/denom2
        Dkt_thetat = -Dkt_ttheta
        Dkt_rphi = (term1 + term2)/denom2
        Dkt_phir = -Dkt_rphi
        Dkt_thetaphi = -(a2 + (r - 2*M)*r)*cos_theta*sin_theta - (4*r*r2_a2**2*sin_2theta)/denom3**2
        Dkt_phitheta = -Dkt_thetaphi
    
        # Covariant derivative of φ-Killing vector
        cov_phi = np.array([
            [0, Dkt_tr, Dkt_ttheta, 0],
            [Dkt_rt, 0, 0, Dkt_rphi],
            [Dkt_thetat, 0, 0, Dkt_thetaphi],
            [0, Dkt_phir, Dkt_phitheta, 0]
        ])
        
        return cov_t, cov_phi
