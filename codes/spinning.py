import numpy as np
from scipy.integrate import cumtrapz
from .constants import *
from .constants import _standardize_params
from .frequencies import _ellippi, _ellippiinc, _ellipeinc
from .frequencies import *
from scipy.special import ellipj
from .orbit import Orbit
from .stable import StableOrbit
from numpy import pi, arccos, cos, sin, sqrt, sign
from kerrgeopy.spacetime import KerrSpacetime
from .osculating_geodesics_funcs_test import *

class SpinningOrbit(StableOrbit):
    def __init__(self, a, p, e, x, s, theta_s, phi_s0, initial_phases=(0, 0, 0, 0), M=None, mu=None):
        a, x = _standardize_params(a, x)
        super().__init__(a, p, e, x, initial_phases, M, mu)
        
        # Store spin parameters
        self.s = s
        self.theta_s = theta_s
        self.phi_s0 = phi_s0
        self.s_parallel = s * np.cos(theta_s)
        self.s_perp = s * np.sin(theta_s)
        self.sqrtK = None  # Will be set after constants are computed
        
        # Create Kerr spacetime object
        self.KerrSpacetime = KerrSpacetime(a=self.a, M=self.M)
        
        # Compute spinning particle constants
        self.compute_spinning_constants()
        sqrtK = np.sqrt(self.K_spin)
        self.sqrtK = sqrtK  # Store for later use
        
        # Compute shifted constants for virtual geodesic
        E_tilde = self.E_spin + self.s_parallel * (1 - self.E_spin**2) / (2 * sqrtK)
        L_tilde = self.L_spin + self.s_parallel * (a - self.L_spin * self.E_spin / 2) / sqrtK
        K_tilde = self.K_spin + self.s_parallel * (3*a*(self.L_spin - a*self.E_spin) - self.K_spin*self.E_spin) / sqrtK
        Q_tilde = K_tilde - (L_tilde - a*E_tilde)**2
        
        # Ensure virtual orbit is stable
        try:
            # First try with original parameters
            self.virtual_orbit = StableOrbit.from_constants(a, E_tilde, L_tilde, Q_tilde, initial_phases, M, mu)
        except ValueError:
            # If unstable, try increasing p slightly
            stable_found = False
            for adjustment in np.linspace(0.01, 0.5, 50):
                try:
                    self.virtual_orbit = StableOrbit.from_constants(
                        a, 
                        E_tilde, 
                        L_tilde, 
                        Q_tilde * (1 + adjustment),
                        initial_phases, 
                        M, 
                        mu
                    )
                    stable_found = True
                    break
                except ValueError:
                    continue
            
            if not stable_found:
                raise ValueError("Could not find stable virtual orbit with given parameters")
        
        self.K_tilde = K_tilde

    def compute_spinning_constants(self):
        """Compute constants of motion for spinning particle using Eqs. (5a-c)"""
        # Get initial position and velocity from geodesic
        t0, r0, theta0, phi0 = self.initial_position
        u_t0, u_r0, u_theta0, u_phi0 = self.initial_velocity
        
        # Compute spin tensor
        S_mu_nu = self.compute_spin_tensor(t0, r0, theta0, phi0, 
                                           u_t0, u_r0, u_theta0, u_phi0)
        
        # Use as specific spin tensor
        s_mu_nu = S_mu_nu
        
        # Get Killing vectors and their derivatives
        xi_t_cov, xi_phi_cov = self.KerrSpacetime.killing_vectors_covariant(r0, theta0)
        xi_t_cov_deriv, xi_phi_cov_deriv = self.KerrSpacetime.killing_vector_derivatives(r0, theta0)
        
        
        # Compute four-velocity vector
        u_mu = np.array([u_t0, u_r0, u_theta0, u_phi0])
        
        # Check norm
        norm_u = self.KerrSpacetime.norm(0, r0, theta0, 0, u_mu)
        
        # Compute geodesic constants from Killing vectors (for verification)
        E_geo_killing = -np.dot(xi_t_cov, u_mu)
        L_geo_killing = np.dot(xi_phi_cov, u_mu)
        
        # Compute E (Eq. 5a)
        term1_E = -np.dot(xi_t_cov, u_mu)
        term2_E = 0.5 * np.einsum('ij,ij', xi_t_cov_deriv, s_mu_nu)
        self.E_spin = term1_E + term2_E
        
        # Compute L (Eq. 5b)
        term1_L = np.dot(xi_phi_cov, u_mu)
        term2_L = -0.5 * np.einsum('ij,ij', xi_phi_cov_deriv, s_mu_nu)
        self.L_spin = term1_L + term2_L
        
        # Compute Carter constant K (Eq. 5c)
        # First, compute geodesic Carter constant from orbital parameters
        K_geo = self.Q + (self.L - self.a * self.E)**2
        
        # Now compute using Killing tensor
        K_mu_nu = self.KerrSpacetime.killing_tensor(r0, theta0)
        term1_K = np.einsum('ij,i,j', K_mu_nu, u_mu, u_mu)
        
        # Get Killing-Yano tensor and its derivative
        Y_mu_nu = self.KerrSpacetime.killing_yano_tensor(r0, theta0)
        
        # Compute covariant derivative of Y using analytic formula
        nabla_Y = self.KerrSpacetime.killing_yano_tensor_derivative(r0, theta0)
        
        # Get inverse metric
        g_inv = self.KerrSpacetime.metric_inv(t0, r0, theta0, phi0)
        
        # Raise indices of spin tensor
        s_contrav = np.einsum('ia,jb,ab->ij', g_inv, g_inv, s_mu_nu)
        
        # Raise first index of Y: Y^α_β = g^{αμ} Y_{μβ}
        Y_raised = np.einsum('ab,cb->ca', g_inv, Y_mu_nu)
        
        # Compute spin correction to K: 4 * u^μ s^{ρσ} Y^κ_{[μ} ∇_κ Y_{σ]ρ}
        # Break into two terms: term2a = 2 * u^μ s^{ρσ} Y^κ_μ ∇_κ Y_{σρ}
        #                   term2b = -2 * u^μ s^{ρσ} Y^κ_σ ∇_κ Y_{μρ}
        
        term2a_K = 2.0 * np.einsum('i,ij,ki,kjl->', u_mu, s_contrav, Y_raised, nabla_Y)
        term2b_K = -2.0 * np.einsum('i,ij,kj,kil->', u_mu, s_contrav, Y_raised, nabla_Y)
        term2_K = term2a_K + term2b_K
        
        
        self.K_spin = term1_K + term2_K
        self.Q_spin = self.K_spin - (self.L_spin - self.a * self.E_spin)**2
        
        return


    def killing_vector_derivatives(r, theta, a, M=1):
        """
        Computes covariant derivatives of the t- and φ-Killing vectors in Kerr spacetime.
        
        Parameters:
            r (float): Radial coordinate
            theta (float): Polar angle in radians
            a (float): Kerr spin parameter
            M (float): Black hole mass (default=1)
            
        Returns:
            tuple: (covariant_derivative_t, covariant_derivative_phi) - both 4x4 matrices
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
    
        # Scale terms by M
        M2 = M**2
        Mr = M * r
        Mr2 = M * r2
    
        Dkphi_tr = -(r2 - a2_cos2)/denom2
        Dkphi_rt = -Dkphi_tr
        Dkphi_rphi = a*(a2_cos2 - r2)*sin_theta**2/denom2
        Dkphi_phir = -Dkphi_rphi
        Dkphi_ttheta = (a2 * r * sin_2theta)/denom2
        Dkphi_thetat = -Dkphi_ttheta 
        Dkphi_thetaphi = a*r*r2_a2*sin_2theta/denom2
        Dkphi_phitheta = -Dkphi_thetaphi  
    
        # Covariant derivative of t-Killing vector
        cov_t = array([
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
    
        cov_phi = array([
            [0, Dkt_tr, Dkt_ttheta, 0],
            [Dkt_rt, 0, 0, Dkt_rphi],
            [Dkt_thetat, 0, 0, Dkt_thetaphi],
            [0, Dkt_phir, Dkt_phitheta, 0]
        ])
        
        return cov_t, cov_phi

    def compute_spin_tensor(self, t, r, theta, phi, u_t, u_r, u_theta, u_phi):
        """Compute spin tensor using Eq. (2) from the paper"""
        # Compute metric
        g_mu_nu = self.KerrSpacetime.metric(t, r, theta, phi)
        det_g = np.linalg.det(g_mu_nu)
        sqrt_g = np.sqrt(np.abs(det_g))
        
        # Compute Levi-Civita tensor (only non-zero permutations)
        epsilon = np.zeros((4, 4, 4, 4))
        
        # Even permutations
        epsilon[0, 1, 2, 3] = sqrt_g
        epsilon[1, 2, 3, 0] = sqrt_g
        epsilon[2, 3, 0, 1] = sqrt_g
        epsilon[3, 0, 1, 2] = sqrt_g
        epsilon[0, 2, 3, 1] = sqrt_g
        epsilon[1, 3, 0, 2] = sqrt_g
        epsilon[2, 0, 1, 3] = sqrt_g
        epsilon[3, 1, 2, 0] = sqrt_g
        
        # Odd permutations (negative)
        epsilon[1, 0, 2, 3] = -sqrt_g
        epsilon[0, 2, 1, 3] = -sqrt_g
        epsilon[0, 1, 3, 2] = -sqrt_g
        epsilon[2, 1, 0, 3] = -sqrt_g
        epsilon[1, 2, 0, 3] = -sqrt_g
        epsilon[2, 0, 3, 1] = -sqrt_g
        epsilon[0, 3, 2, 1] = -sqrt_g
        epsilon[3, 2, 1, 0] = -sqrt_g
        
        # Compute spin vector in Marck tetrad basis
        # Note: This is an approximation; for exact solution, should use Marck tetrad
        s_mu = np.zeros(4)
        
        # Project spin onto coordinate basis
        # s_parallel component along orbital angular momentum direction
        # s_perp component in orbital plane
        u_mu_array = np.array([u_t, u_r, u_theta, u_phi])
        
        # Simple projection - in practice should use Marck tetrad
        s_mu = (self.s_parallel * u_mu_array + 
                self.s_perp * np.array([0, np.cos(self.phi_s0), np.sin(self.phi_s0), 0]))
        
        # Normalize spin vector to have correct magnitude
        norm_sq = np.einsum('i,ij,j', s_mu, g_mu_nu, s_mu)
        if norm_sq > 0:
            s_mu = s_mu * (self.s / np.sqrt(np.abs(norm_sq)))
        
        # Compute spin tensor: S^{μν} = ε^{μνρσ} u_ρ s_σ
        u_mu_lower = np.einsum('ij,j->i', g_mu_nu, u_mu_array)
        S_mu_nu = np.zeros((4, 4))
        
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        S_mu_nu[mu, nu] += epsilon[mu, nu, rho, sigma] * u_mu_lower[rho] * s_mu[sigma]
        
        # Ensure antisymmetry
        S_mu_nu = 0.5 * (S_mu_nu - S_mu_nu.T)
        
        return S_mu_nu
    def displacement_vector(self, pos, vel, psi):
        """Compute full displacement vector δx^μ (Eqs. S12-S14)"""
        t, r, theta, phi = pos
        u_t, u_r, u_theta, u_phi = vel
        a = self.a
        M = self.M
        
        # Extract constants from virtual geodesic
        E_tilde = self.virtual_orbit.E
        L_tilde = self.virtual_orbit.L
        K_tilde = self.K_tilde
        
        # Precompute trigonometric terms
        sin_theta = np.sin(theta)
        z = np.cos(theta)
        z2 = z**2
        
        # Compute fundamental quantities (Eqs. S5, S6)
        Delta = r**2 - 2*M*r + a**2
        Sigma = r**2 + a**2 * z2
        P_r = (r**2 + a**2) * E_tilde - a * L_tilde
        P_z = L_tilde - a * E_tilde * (1 - z2)
        
        # Derivatives w.r.t Mino time (dλ)
        dr_dlambda = Sigma * u_r
        dz_dlambda = Sigma * (-sin_theta * u_theta)  # dz/dλ = d(cosθ)/dλ
        
        # Compute Ξ (Eq. S11)
        Xi = np.sqrt((K_tilde - a**2 * z2) / (K_tilde + r**2))
        Xi_inv = 1.0 / Xi if Xi != 0 else 0.0
        
        # Precompute common factors
        sqrtK = np.sqrt(K_tilde)
        K_Sigma = sqrtK * Sigma
        K_Sigma_inv = 1.0 / K_Sigma if K_Sigma != 0 else 0.0
        arz = a * r * z
        
        # Compute δx1 components (Eq. S12)
        dx1_t = a * K_Sigma_inv * (
            (a**2 + r**2)/Delta * z * Xi * dr_dlambda + 
            r * Xi_inv * dz_dlambda
        )
        dx1_r = a * z * Xi * P_r * K_Sigma_inv
        dx1_z = -r * Xi_inv * P_z * K_Sigma_inv
        dx1_theta = -dx1_z / sin_theta  # ∂/∂z → ∂/∂θ
        dx1_phi = K_Sigma_inv * (
            a**2 * z/Delta * Xi * dr_dlambda + 
            r/(1 - z2) * Xi_inv * dz_dlambda
        )
        
        # Compute δx2 components (Eq. S13)
        factor2 = -arz / (K_tilde * Sigma)
        dx2_t = factor2 * (
            (r**2 + a**2)/Delta * Xi * P_r + 
            a * Xi_inv * P_z
        )
        dx2_r = factor2 * Xi * dr_dlambda
        dx2_z = factor2 * Xi_inv * dz_dlambda
        dx2_theta = -dx2_z / sin_theta  # ∂/∂z → ∂/∂θ
        dx2_phi = factor2 * (
            a/Delta * Xi * P_r + 
            1/(1 - z2) * Xi_inv * P_z
        )
        
        # Compute δx3 components (Eq. S14)
        dx3_t = K_Sigma_inv * (
            r * (r**2 + a**2)/Delta * dr_dlambda - 
            a**2 * z * dz_dlambda
        )
        dx3_r = r * P_r * K_Sigma_inv
        dx3_z = a * z * P_z * K_Sigma_inv
        dx3_theta = -dx3_z / sin_theta  # ∂/∂z → ∂/∂θ
        dx3_phi = a * K_Sigma_inv * (
            r/Delta * dr_dlambda - 
            z/(1 - z2) * dz_dlambda
        )
        
        # Combine components (Eq. 13)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        dx_t = (self.s_parallel * dx3_t + 
                self.s_perp * (dx1_t * cos_psi + dx2_t * sin_psi))
        dx_r = (self.s_parallel * dx3_r + 
                self.s_perp * (dx1_r * cos_psi + dx2_r * sin_psi))
        dx_theta = (self.s_parallel * dx3_theta + 
                    self.s_perp * (dx1_theta * cos_psi + dx2_theta * sin_psi))
        dx_phi = (self.s_parallel * dx3_phi + 
                  self.s_perp * (dx1_phi * cos_psi + dx2_phi * sin_psi))
        
        return np.array([dx_t, dx_r, dx_theta, dx_phi])
    
    def compute_psi(self, lambda_array):
        """Compute spin precession phase ψ(λ) (Eq. S4)"""
        # Get virtual geodesic constants
        E = self.virtual_orbit.E
        L = self.virtual_orbit.L
        K = self.K_tilde
        sqrtK = np.sqrt(K)
        
        # Get virtual geodesic trajectory
        _, r_func, theta_func, _ = self.virtual_orbit.trajectory()
        r_vals = r_func(lambda_array)
        theta_vals = theta_func(lambda_array)
        z_vals = np.cos(theta_vals)
        
        # Compute P_r and P_z (Eq. S6)
        P_r = (r_vals**2 + self.a**2) * E - self.a * L
        P_z = L - self.a * E * (1 - z_vals**2)
        
        # Compute denominators safely
        denom_r = K + r_vals**2
        denom_z = K - (self.a * z_vals)**2
        denom_r[denom_r == 0] = 1e-16
        denom_z[denom_z == 0] = 1e-16
        
        # Compute Ψ_r and Ψ_z (Eq. S4)
        Psi_r = sqrtK * P_r / denom_r
        Psi_z = self.a * sqrtK * P_z / denom_z
        
        # Integrate dψ/dλ = -(Ψ_r + Ψ_z)
        dpsi_dlambda = -(Psi_r + Psi_z)
        psi = self.phi_s0 + np.cumsum(dpsi_dlambda) * (lambda_array[1] - lambda_array[0])
        
        return psi

    def compute_trajectory(self, lambda_array, distance_units="natural", time_units="natural"):
        """Compute trajectory of spinning particle"""
        # Virtual geodesic trajectory and four-velocity
        t_g_func, r_g_func, theta_g_func, phi_g_func = self.virtual_orbit.trajectory()
        u_t_func, u_r_func, u_theta_func, u_phi_func = self.virtual_orbit.four_velocity()
        
        t_g = t_g_func(lambda_array)
        r_g = r_g_func(lambda_array)
        theta_g = theta_g_func(lambda_array)
        phi_g = phi_g_func(lambda_array)
        u_t = u_t_func(lambda_array)
        u_r = u_r_func(lambda_array)
        u_theta = u_theta_func(lambda_array)
        u_phi = u_phi_func(lambda_array)
        
        # Compute Σ_g and τ_g (Eq. 24)
        Sigma_g = r_g**2 + (self.a * np.cos(theta_g))**2
        tau_g = np.cumsum(Sigma_g) * (lambda_array[1] - lambda_array[0])
        
        # Compute spin precession phase
        psi = self.compute_psi(lambda_array)
        
        # Initialize displacement vector
        delta_x = np.zeros((len(lambda_array), 4))
        
        # Compute δx^μ at each point
        for i, lam in enumerate(lambda_array):
            pos = [t_g[i], r_g[i], theta_g[i], phi_g[i]]
            vel = [u_t[i], u_r[i], u_theta[i], u_phi[i]]
            delta_x[i] = self.displacement_vector(pos, vel, psi[i])
        
        # Apply transformation (Eqs. 25a-b)
        t_spin = t_g - (3*self.s_parallel/(2*self.sqrtK)) * tau_g - delta_x[:, 0]
        r_spin = r_g - delta_x[:, 1]
        theta_spin = theta_g - delta_x[:, 2]
        phi_spin = phi_g - delta_x[:, 3]
        
        # Unit conversions
        if distance_units != "natural" or time_units != "natural":
            if self.M is None:
                raise ValueError("M required for unit conversion")
            t_spin = time_in_seconds(t_spin, self.M) if time_units == "mks" else t_spin
            r_spin = distance_in_meters(r_spin, self.M) if distance_units == "mks" else r_spin
        
        return t_spin, r_spin, theta_spin, phi_spin