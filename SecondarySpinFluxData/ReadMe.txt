====================================================================
IMPORTANT
====================================================================

This dataset was produced through a collaboration between Lisa 
Drummond, Viktor Skoupy, Phil Lynch, Francisco Duque and additional 
collaborators.

The dataset contained in this directory is preliminary and not public; 
it should not be redistributed or shared.

If this dataset is used, it must be properly referenced.
Appropriate credit shouldbe given to the collaborators listed above, 
either by citing the relevant publication describing the dataset
or by referencing the official public release (when one exists).
====================================================================

# Schwarzschild Equatorial Secondary-Spin Flux Grids

This directory contains tabulated flux data used to interpolate 
secondary-spin corrections for eccentric equatorial inspirals 
in Schwarzschild spacetime.

## Files

- `fluxes_schw_inner_Edot_Ldot_deltaTildeEdot_deltaTildeLdot.dat`
- `fluxes_schw_outer_Edot_Ldot_deltaTildeEdot_deltaTildeLdot.dat`

The two files represent two interpolation regions in the parameter space. 

## File format

The columns are:

- **col 0**: `u`
- **col 1**: `w`
- **col 3**: `p` (semi-latus rectum)
- **col 4**: `e` (eccentricity)
- **col 7**: stored energy-flux quantity
- **col 8**: stored angular-momentum-flux quantity

The reader sorts rows lexicographically by `(u, w)` and reshapes the
flux data into a 2D grid with shape

`(len(unique(u)), len(unique(w)))`.

## Sign convention

The file values in columns 7 and 8 should be negated on input
when used to generate an inspiral:

- `Edot = -data[:, 7]`
- `Ldot = -data[:, 8]`

## How the data are used

The tabulated `Edot` and `Ldot` values are divided by post-Newtonian
reference fluxes,

- `_E1dotPN(p, e)`
- `_L1dotPN(p, e)`

and interpolated in `(u, w)`.

The code also converts `(Edot, Ldot)` to `(pdot, edot)` using
`ELdot_to_PEdot_Jacobian(...)`, then divides by PN reference rates

- `_p1dot_PN(p, e)`
- `_e1dot_PN(p, e)`

to build interpolants for `pdot` and `edot`.

====================================================================
Example code to unpack grid and interpolate fluxes
====================================================================

# PN Rescaling Expressions

def _E1dotPN(p, e):
    return  p**(-5.5)*((1-e**2)**(3/2)) * (1 + (73/24)*e**2 + (37/96)*e**4)

def _L1dotPN(p, e):
    return p**(-4)*((1-e**2)**(3/2)) * (1 + (7/8)*e**2)

def _p1dot_PN(p, e):
    return p**(-3.5) * ((1 - e**2)**(3/2)) * (1 + (7/8)*e**2)

def _e1dot_PN(p, e):
    return p**(-4.5) * ((1 - e**2)**(3/2)) * (304 + 121*e**2)


DELTAPMINSPIN = 0.02540147545182414

# Modified Flux Trajectory with Secondary Spin Corrections
class KerrEccEqFluxSecSpin(KerrEccEqFlux):

    def __init__(
        self,
        *args,
        use_ELQ: bool = False,
        downsample=None,
        flux_output_convention="pex",
        **kwargs,
    ):
        super().__init__(*args, use_ELQ=use_ELQ, downsample=downsample, **kwargs)

        self.flux_output_convention = flux_output_convention

        # Interpolating inner flux data

        fpA = 'fluxes_schw_inner_Edot_Ldot_deltaTildeEdot_deltaTildeLdot.dat'

        innerData = np.loadtxt(fpA)
        innerDataSorted = innerData[np.lexsort((innerData[:,1], innerData[:,0]))]

        u_inner = innerDataSorted[:, 0]
        w_inner = innerDataSorted[:, 1]
        p_inner = innerDataSorted[:, 3]
        e_inner = innerDataSorted[:, 4]
        a_inner = np.zeros(len(u_inner))
        x_inner = np.ones(len(u_inner))
        Edot_inner = -innerDataSorted[:, 7]
        Ldot_inner = -innerDataSorted[:, 8]
        # extract unqie values of u and w
        us = np.unique(u_inner)
        ws = np.unique(w_inner)

        Edot_grid = Edot_inner.reshape(len(us), len(ws))
        Ldot_grid = Ldot_inner.reshape(len(us), len(ws))
        Edot_PN = _E1dotPN(p_inner,e_inner).reshape(len(us), len(ws))
        Ldot_PN = _L1dotPN(p_inner,e_inner).reshape(len(us), len(ws))

        self.Edot_spin_interp_A = BicubicSpline(us, ws, Edot_grid/Edot_PN)
        self.Ldot_spin_interp_A = BicubicSpline(us, ws, Ldot_grid/Ldot_PN)
        out_pdot_edot = np.asarray(
                    [
                        ELdot_to_PEdot_Jacobian(
                            a_inner[i],
                            p_inner[i],
                            e_inner[i],
                            x_inner[i],
                            Edot_inner[i],
                            Ldot_inner[i],
                        )
                        for i in range(Edot_inner.size)
                    ]
        )


        pdot_inner = (out_pdot_edot[:, 0]).reshape(us.size, ws.size)
        edot_inner = (out_pdot_edot[:, 1]).reshape(us.size, ws.size)

        pdot_PN = _p1dot_PN(p_inner,e_inner).reshape(us.size, ws.size)
        edot_PN = _e1dot_PN(p_inner,e_inner).reshape(us.size, ws.size)

        self.pdot_spin_interp_A = BicubicSpline(us, ws, pdot_inner/pdot_PN)
        self.edot_spin_interp_A = BicubicSpline(us, ws, edot_inner/edot_PN)


        # Interpolating outer flux data

        fpB = 'fluxes_schw_outer_Edot_Ldot_deltaTildeEdot_deltaTildeLdot.dat'

        outerData = np.loadtxt(fpB)
        outerDataSorted = outerData[np.lexsort((outerData[:,1], outerData[:,0]))]
        u_outer = outerDataSorted[:, 0]
        w_outer = outerDataSorted[:, 1]
        p_outer = outerDataSorted[:, 3]
        e_outer = outerDataSorted[:, 4]
        a_outer = np.zeros(len(u_outer))
        x_outer = np.ones(len(u_outer))
        Edot_outer = -outerDataSorted[:, 7]
        Ldot_outer = -outerDataSorted[:, 8]

        # extract unique values of U and W
        Us = np.unique(outerDataSorted[:, 0])
        Ws = np.unique(outerDataSorted[:, 1])

        Edot_grid = Edot_outer.reshape(len(Us), len(Ws))
        Ldot_grid = Ldot_outer.reshape(len(Us), len(Ws))

        Edot_PN = _E1dotPN(p_outer,e_outer).reshape(len(Us), len(Ws))
        Ldot_PN = _L1dotPN(p_outer,e_outer).reshape(len(Us), len(Ws))

        self.Edot_spin_interp_B = BicubicSpline(Us, Ws, Edot_grid/Edot_PN)
        self.Ldot_spin_interp_B = BicubicSpline(Us, Ws, Ldot_grid/Ldot_PN)

        out_pdot_edot = np.asarray(
                    [
                        ELdot_to_PEdot_Jacobian(
                            a_outer[i],
                            p_outer[i],
                            e_outer[i],
                            x_outer[i],
                            Edot_outer[i],
                            Ldot_outer[i],
                        )
                        for i in range(Edot_outer.size)
                    ]
        )


        pdot_outer = (out_pdot_edot[:, 0]).reshape(Us.size, Ws.size)
        edot_outer = (out_pdot_edot[:, 1]).reshape(Us.size, Ws.size)

        pdot_PN = _p1dot_PN(p_outer,e_outer).reshape(Us.size, Ws.size)
        edot_PN = _e1dot_PN(p_outer,e_outer).reshape(Us.size, Ws.size)

        self.pdot_spin_interp_B = BicubicSpline(Us, Ws, pdot_outer/pdot_PN)
        self.edot_spin_interp_B = BicubicSpline(Us, Ws, edot_outer/edot_PN)

    @property
    def separatrix_buffer_dist(self):
        return 1.5 * DELTAPMINSPIN

    @property
    def separatrix_buffer_dist_grid(self):
        return DELTAPMINSPIN
    
    def interpolate_spin_flux_grids(
            self,
            p: float,
            e: float,
            x: float = 1,
            a: float = 0,
            pLSO: float = None,
        ) -> tuple[float]:
            if pLSO is None:
                pLSO = 6 + 2*e

            edge_buffer = -1e-8

            # handle xI = -1 case
            if x == -1:
                a_in = -a
            else:
                a_in = a

            u, w, _, z, in_region_A = _kerrecceq_flux_forward_map(a_in, p, e, 1.0, pLSO)

            if u < edge_buffer or u > 1 - edge_buffer or np.isnan(u):
                raise ValueError("Interpolation: p out of bounds.")
            if w < edge_buffer:
                raise TrajectoryOffGridException("Interpolation: e out of bounds.")
            if w > 1 - edge_buffer:
                if self.integrate_backwards:
                    raise ValueError("Interpolation: e out of bounds.")
                else:
                    raise TrajectoryOffGridException("Interpolation: e out of bounds.")

            if z < edge_buffer or z > 1 - edge_buffer:
                raise TrajectoryOffGridException("Interpolation: a out of bounds.")

            if self.flux_output_convention == "ELQ":
                E1dotPN = _E1dotPN(p, e)
                L1dotPN = _L1dotPN(p, e)
                if in_region_A:
                    Edot = self.Edot_spin_interp_A(u, w) * E1dotPN
                    Ldot = self.Ldot_spin_interp_A(u, w) * L1dotPN
                else:
                    Edot = self.Edot_spin_interp_B(u, w) * E1dotPN
                    Ldot = self.Ldot_spin_interp_B(u, w) * L1dotPN
                if a_in < 0:
                    Ldot *= -1

                return Edot, Ldot

            else:
                #risco = get_separatrix(a_in, 0.0, 1.0)
                #p_sep = pLSO
                p1dotPN = _p1dot_PN(p, e)
                e1dotPN = _e1dot_PN(p, e)
                if in_region_A:
                    pdot = self.pdot_spin_interp_A(u, w) * p1dotPN
                    edot = self.edot_spin_interp_A(u, w) * e1dotPN
                else:
                    pdot = self.pdot_spin_interp_B(u, w) * p1dotPN
                    edot = self.edot_spin_interp_B(u, w) * e1dotPN

                return pdot, edot

    def modify_rhs(self, ydot, y):
        # in-place modification of the derivatives
        s = self.additional_args[0]
        nu = self.massratio

        if self.use_ELQ:
            E, L, Q = y[:3]
            p, e, x = ELQ_to_pex(self.a, E, L, Q)
        else:
            p, e, x = y[:3]

        # Will also add in spin corrections to frequencies

        ydot0, ydot1 = self.interpolate_spin_flux_grids(
            p, e, x, a=self.a, pLSO=self.p_sep_cache
        )
        ydot[0] = ydot[0] + s*nu*ydot0
        ydot[1] = ydot[1] + s*nu*ydot1
        #ydot[0] = ydot0
        #ydot[1] = ydot1
        # xdot still zero for equatorial orbits

        OmegaPhi1, OmegaTheta1, OmegaR1 = get_fundamental_frequencies_SkoupyWitzany_spin_corrections(
            self.a, p, e, x
        )

        #ydot[3] = OmegaPhi1
        #ydot[4] = OmegaTheta1
        #ydot[5] = OmegaR1
        ydot[3] = ydot[3] + s*nu*OmegaPhi1
        ydot[4] = ydot[4] + s*nu*OmegaTheta1
        ydot[5] = ydot[5] + s*nu*OmegaR1
