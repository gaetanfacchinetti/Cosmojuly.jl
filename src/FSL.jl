
module FSLModel

include("./Halos.jl")

import QuadGK: quadgk

import Main.Cosmojuly.BackgroundCosmo: BkgCosmology, planck18_bkg
import Main.Cosmojuly.PowerSpectrum: Cosmology, planck18
import Main.Cosmojuly.Halos: Halo, HaloProfile, NFWProfile, αβγProfile

import Main.Cosmojuly.Halos.ρ as ρ_halo
import Main.Cosmojuly.Halos.μ as μ_halo

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

export density_HI, density_baryons_spherical

# Definition of the profiles
function density_spherical_BG02(r::Real, z::Real; ρ0b::Real, r0::Real, q::Real, α::Real, rcut::Real) 
    rp = sqrt(r^2 + (z/q)^2)
    return ρ0b/(1+rp/r0)^α * exp(-(rp/rcut)^2)
end

density_exponential_disc(r::Real, z::Real; σ0::Real, rd::Real, zd::Real) = σ0/(2.0*zd)*exp(-abs(z)/zd - r/rd)
density_sech_disc(r::Real, z::Real; σ0::Real, rd::Real, rm::Real, zd::Real) = σ0/(4.0*zd)*exp(-rm/r - r/rd)*(sech(z/(2.0*zd)))^2

abstract type GalaxyModel end
abstract type MM17Model <: GalaxyModel end
abstract type MM17Gamma1 <: MM17Model end
abstract type MM17Gamma0 <: MM17Model end
abstract type MM17GammaFree <: MM17Model end

# we take the median value of the MCMC
halo(::Type{MM17Gamma1})    = Halo_from_ρs_and_rs(αβγProfile(1, 3, 1), 9.24412866426226e+15, 1.86e-2)
halo(::Type{MM17Gamma0})    = Halo_from_ρs_and_rs(αβγProfile(1, 3, 0), 9.086059744049174e+16, 7.7e-3)
halo(::Type{MM17GammaFree}) = Halo_from_ρs_and_rs(αβγProfile(1.0, 3.0, 0.79), 1.5300738158389776e+16, 1.54e-2)

halo(::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = halo(T)
density_dm(r::Real, ::Type{T} = MM17Gamma1) where {T <: MM17Model} = ρ_halo(r, halo(T))

density_HI(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = density_sech_disc(r, z, σ0=5.31e+13, rd=8.5e-5, rm=4.0e-3, zd=7.0e-3)
density_H2(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = density_sech_disc(r, z, σ0=2.180e+15, rd=4.5e-5, rm=1.2e-2, zd=1.5e-3)
density_bulge(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = density_spherical_BG02(r, z, ρ0b = 9.73e+19, r0 = 7.5e-5, q = 0.5, α = 1.8, rcut = 2.1e-3)
density_thick_stellar_disc(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = density_exponential_disc(r, z, σ0=1.487e+14, rd =9.0e-4, zd=3.29e-3)
density_thin_stellar_disc(r::Real, z::Real, ::Type{MM17Gamma1})    = density_exponential_disc(r, z, σ0=8.87e+14, rd=3.0e-4, zd=2.53e-3)
density_thin_stellar_disc(r::Real, z::Real, ::Type{MM17Gamma0})    = density_exponential_disc(r, z, σ0=8.87e+14, rd=3.0e-4, zd=2.36e-3)
density_thin_stellar_disc(r::Real, z::Real, ::Type{MM17GammaFree}) = density_exponential_disc(r, z, σ0=8.87e+14, rd=3.0e-4, zd=2.51e-3)
density_thin_stellar_disc(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:MM17Model} = density_thin_stellar_disc(r, z, T)

density_baryons(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = density_HI(r, z, T) + density_H2(r, z, T) + density_bulge(r, z, T) + density_thick_stellar_disc(r, z, T) + density_thin_stellar_disc(r, z, T)
density_ISM(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = density_HI(r, z, T) + density_H2(r, z, T)

function der_density_baryons_spherical(xp::Real, r::Real, ::Type{T}) where {T<:MM17Model}
    y = sqrt(1 - xp^2)
    return xp/y * (density_baryons(r * xp, -y, T) + density_baryons(r * xp, y, T))
end

density_baryons_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = quadgk(xp -> der_density_baryons_spherical(xp, r, T), 0, 1, rtol=1e-4)[1] 

#mvir(::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = m_halo(, halo(T)) -> need to implement


end # module FSL
