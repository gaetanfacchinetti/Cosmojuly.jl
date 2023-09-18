
module FSLModel

#include("./Halos.jl")

import Main.Cosmojuly.BackgroundCosmo: BkgCosmology, planck18_bkg
import Main.Cosmojuly.PowerSpectrum: Cosmology, planck18

export subhalo_mass_function_template
export mass_function_merger_tree

#############################################################
# Defnitions of basic functions

@doc raw""" 
    subhalo_mass_function_template(x, γ1, α1, γ2, α2, β, ζ)

Template function for the subhalo mass function:

``m_Δ^{\rm host} \frac{\partial N(m_Δ^{\rm sub}, z=0)}{\partial m_Δ^{\rm sub}} = \left(\gamma_1 x^{-\alpha_1} + \gamma_2 x^{-\alpha_2}\right)  e^{-\beta x^\zeta}``

The first argument, `x::Real`, is the ratio of the subhalo over the host mass ``m_Δ^{\rm sub} / m_Δ^{\rm host}.``
"""
function subhalo_mass_function_template(x::Real, γ1::Real,  α1::Real, γ2::Real, α2::Real, β::Real, ζ::Real)
    return (γ1*x^(-α1) + γ2*x^(-α2)) * exp(-β * x^ζ )
end


@doc raw"""
    mass_function_merger_tree(mΔ_sub, mΔ_host)

Example of subhalo mass function fitted on merger tree results
(Facchinetti et al., in prep.)

# Arguments
- `mΔ_sub::Real` : subhalo virial mass (in Msun)
- `mΔ_host::Real`: host virial mass (in Msun)

# Returns
- ``\frac{\partial N(m_Δ^{\rm sub}, z=0)}{\partial m_Δ^{\rm sub}}``
"""
function mass_function_merger_tree(mΔ_sub::Real, mΔ_host::Real) 
    return subhalo_mass_function_template(mΔ_sub / mΔ_host, 0.019, 1.94, 0.464, 1.58, 24.0, 3.4)/mΔ_host
end


function pdf_concentration(cΔ::Real, mΔ::Real; median::Function = median_concentration_SCP12, std::Function = m -> 0.14 * log(10.0))
   
    σ_c = std(mΔ)
    median_c = median(mΔ)

    Kc = 0.5 * erfc(-log(median_c) / (sqrt(2.0) * σ_c))

    return 1.0 / Kc / cΔ / sqrt(2.0 * π) / σ_c * exp(-(log(cΔ) - log(median_c)) / sqrt(2.0) / σ_c^2)
end


#m200 in Msol
function median_concentration_SCP12(mΔ::Real, h::Real)
    cn::Vector = [37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7]
    mΔ_min = (mΔ > 7.24e-10) ? mΔ : 7.24e-10
    return sum(cn .* log(mΔ_min * h).^ (0:5))
end

median_concentration_SCP12(mΔ::Real, bkg_cosmo::BkgCosmology = planck18_bkg) = median_concentration_SCP12(mΔ, bkg_cosmo.h)
median_concentration_SCP12(mΔ::Real, cosmo::Cosmology = planck18) = median_concentration_SCP12(mΔ, cosmo.bkg.h)


#############################################################

# Model of galaxies

struct Bulge{T<:Real}
    ρ0b::T
    r0::T
    q::T
    rcut::T
end

abstract type Disc{T<:Real} end

struct StellarDisc{T<:Real} <: Disc{T}
    Σ0::T
    zd::T
    Rd::T
end

# Definition of the profiles
density_spherical_BG02() = 0
density_exponential_disc(R::Real, z::Real; Σ0::Real, Rd::Real, zd::Real) = Σ0/(2.0*zd)*exp(-abs(z)/zd - R/Rd)
density_sech_disc(R::Real, z::Real; Σ0::Real, Rd::Real, Rm::Real, zd::Real)  = Σ0/(4.0*zd)*exp(-Rm/R - R/Rd)*(sech(z/(2.0*zd)))^2

density_exponential_disc(R::Real, z::Real; stellar_disc::StellarDisc) = density_exponential_disc(R, z, Σ0 = stellar_disc.Σ0, Rd = stellar_disc.Rd, zd = stellar_disc.zd)

struct GasDisc{T<:Real} <: Disc{T}
    Σ0::T
    zd::T
    Rm::T
    Td::T
end

struct DiscGalaxy
    halo::Halo
    thick_stellar_disk::StellarDisk
    thin_stellar_disk::StellarDisk
end




end # module FSL
