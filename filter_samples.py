import numpy as np
import scipy
from scipy.interpolate import interp1d
import pQCD
from pQCD import pQCD



def energyd(n, P, epsi=None, n0=0.08, p0=0.3829,e0=75.9712):

    use_default = (epsi is None)

    #-----------------------------------
    # initialise
    #-----------------------------------

    if use_default:

        n_new = np.concatenate(([n0], n))

        # add the low-density pressure point
        P = np.concatenate(
            ( np.full((*P.shape[:-1], 1), p0), P),
            axis=-1)

        eps_next = np.zeros(
            (*P.shape[:-1], len(n_new))
        )

        eps_next[..., 0] = e0

        n_use = n_new

    else:

        eps_next = np.zeros(
            (*P.shape[:-1], len(n))
        )

        eps_next[..., 0] = epsi

        n_use = n

    #-----------------------------------
    # integrate
    #-----------------------------------

    for i in range(len(n_use)-1):

        dn = n_use[i+1] - n_use[i]

        integral = 0.5*dn*(
            P[..., i]/n_use[i]**2
            +
            P[..., i+1]/n_use[i+1]**2
        )

        eps_next[..., i+1] = (

            n_use[i+1]*(
                eps_next[..., i]/n_use[i]
                +
                integral
            )

        )

    #-----------------------------------
    # remove the artificial first point
    #-----------------------------------

    if use_default:

        return eps_next[...,1:]

    return eps_next

def monotonic_mask(P):

    return np.all(np.diff(P,axis=-1)>=0,axis=-1)


def cs2(P,E):

    dP=np.diff(P,axis=-1)
    dE=np.diff(E,axis=-1)

    return dP/dE

NX=1000

np.random.seed(10)

print(NX,"NX")

Xs=np.exp(
        np.random.uniform(
            np.log(1/2),
            np.log(2),
            NX
        )
)

pQCD_models=[pQCD(x) for x in Xs]


def pqcd_constraint(

        n,
        P,
        E,
        density

):

    # allows density=1.28 or density=(N_samples,)
    if np.isscalar(density):

        density=np.full(
                len(P),
                density
            )


    P_atX=np.array([

            np.interp(
                density[i],
                n,
                P[i]
            )

            for i in range(len(P))

    ])/1000.


    E_atX=np.array([

            np.interp(
                density[i],
                n,
                E[i]
            )

            for i in range(len(E))

    ])/1000.


    weight=np.zeros(
                (NX,len(P))
            )


    for i,pQCDX in enumerate(pQCD_models):

        weight[i]=(

            pQCDX.constraints(

                    e0=E_atX,
                    p0=P_atX,
                    n0=density

                ).astype(int))


    return weight.mean(axis=0)

PQCD_DENSITIES=[
        0.35,
        0.40,
        0.50,
        0.70,
        1.00,
        1.28
]


def physical_filter(

        P,
        n,
        nc=None,
        pqcd=True,
        causal=True,
        monotonic=True,
        pqcd_threshold=0.0,
        epsi=None,
        n0=0.08,
        p0=0.3829,
        e0=75.9712

):


    mask=np.ones(
            P.shape[:-1],
            dtype=bool
        )

    N0=len(mask)


    #----------------------------------
    # monotonicity
    #----------------------------------

    if monotonic:

        mask &= monotonic_mask(P)

    N1=mask.sum()


    if not np.any(mask):

        return mask


    #----------------------------------
    # causality
    #----------------------------------

    if causal:

        P_phys=P[mask]

        E=energyd(
                n,
                P_phys,
                epsi=epsi,
                e0=e0,
                p0=p0,
                n0=n0
            )


        cs=cs2(
                P_phys,
                E
            )


        mask_causal=np.all(

                (cs<=1)
                &
                (cs>=0),

                axis=-1

            )


        mask[mask]=mask_causal


    N2=mask.sum()


    #----------------------------------
    # pQCD
    #----------------------------------

    if pqcd:

        P_phys=P[mask]


        E=energyd(
                n,
                P_phys,
                epsi=epsi,
                e0=e0,
                p0=p0,
                n0=n0
            )


        # use the same density for every sample
        if nc is None:

            density=1.28

        # every EOS has its own central density
        else:

            density=np.repeat(
                        nc[:,None],
                        P.shape[1],
                        axis=1
                    )[mask]


        ll=pqcd_constraint(

                n,
                P_phys,
                E,
                density

            )


        mask_pqcd=(

                ll>pqcd_threshold

            )


        mask[mask]=mask_pqcd


    N3=mask.sum()


    return mask,{

        "begining":N0,
        "monotonic":N1,
        "causal":N2,
        "pqcd":N3

    }