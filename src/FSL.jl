
module FSLModel

include("./Halos.jl")

import QuadGK: quadgk

import Main.Cosmojuly.BackgroundCosmo: BkgCosmology, planck18_bkg
import Main.Cosmojuly.PowerSpectrum: Cosmology, planck18
import Main.Cosmojuly.Halos: Halo, HaloProfile, NFWProfile, αβγProfile, halo_from_ρs_and_rs, m_halo, ρ_halo, μ_halo

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

#pdf_position(r::Real, ::Type{T}) where {T<:GalaxyModel} = 4 * π  * r^2 * ρ_dm(r, T) / mΔ(galactic_halo(T))


#############################################################

# Model of galaxies

export ρ_HI, ρ_baryons_spherical, galactic_halo, m_baryons_spherical, μ_baryons_spherical
export galactic_profile, ρ_dm, m_galaxy_spherical, μ_galaxy_spherical, ρ_galaxy_spherical

# Definition of the profiles
function ρ_spherical_BG02(r::Real, z::Real; ρ0b::Real, r0::Real, q::Real, α::Real, rcut::Real) 
    rp = sqrt(r^2 + (z/q)^2)
    return ρ0b/(1+rp/r0)^α * exp(-(rp/rcut)^2)
end

ρ_exponential_disc(r::Real, z::Real; σ0::Real, rd::Real, zd::Real) = σ0/(2.0*zd)*exp(-abs(z)/zd - r/rd)
ρ_sech_disc(r::Real, z::Real; σ0::Real, rd::Real, rm::Real, zd::Real) = σ0/(4.0*zd)*exp(-rm/r - r/rd)*(sech(z/(2.0*zd)))^2

abstract type GalaxyModel end
abstract type MM17Model <: GalaxyModel end
abstract type MM17Gamma1 <: MM17Model end
abstract type MM17Gamma0 <: MM17Model end
abstract type MM17GammaFree <: MM17Model end

# we take the median value of the MCMC
galactic_profile(::Type{MM17Gamma1})    = αβγProfile(1, 3, 1)
galactic_profile(::Type{MM17Gamma0})    = αβγProfile(1, 3, 0)
galactic_profile(::Type{MM17GammaFree}) = αβγProfile(1, 3, 0.79)

galactic_halo(::Type{MM17Gamma1})    = halo_from_ρs_and_rs(galactic_profile(MM17Gamma1), 9.24412866426226e+15, 1.86e-2)
galactic_halo(::Type{MM17Gamma0})    = halo_from_ρs_and_rs(galactic_profile(MM17Gamma0), 9.086059744049174e+16, 7.7e-3)
galactic_halo(::Type{MM17GammaFree}) = halo_from_ρs_and_rs(galactic_profile(MM17GammaFree), 1.5300738158389776e+16, 1.54e-2)

galactic_profile(::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = galactic_profile(T)
galactic_halo(::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = galactic_halo(T)
ρ_dm(r::Real, ::Type{T} = MM17Gamma1) where {T <: MM17Model} = ρ_halo(r, galactic_halo(T))

ρ_HI(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_sech_disc(r, z, σ0=5.31e+13, rd=8.5e-5, rm=4.0e-3, zd=7.0e-3)
ρ_H2(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_sech_disc(r, z, σ0=2.180e+15, rd=4.5e-5, rm=1.2e-2, zd=1.5e-3)
ρ_bulge(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_spherical_BG02(r, z, ρ0b = 9.73e+19, r0 = 7.5e-5, q = 0.5, α = 1.8, rcut = 2.1e-3)
ρ_thick_stellar_disc(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_exponential_disc(r, z, σ0=1.487e+14, rd =9.0e-4, zd=3.29e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{MM17Gamma1})    = ρ_exponential_disc(r, z, σ0=8.87e+14, rd=3.0e-4, zd=2.53e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{MM17Gamma0})    = ρ_exponential_disc(r, z, σ0=8.87e+14, rd=3.0e-4, zd=2.36e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{MM17GammaFree}) = ρ_exponential_disc(r, z, σ0=8.87e+14, rd=3.0e-4, zd=2.51e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:MM17Model} = ρ_thin_stellar_disc(r, z, T)

ρ_baryons(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = ρ_HI(r, z, T) + ρ_H2(r, z, T) + ρ_bulge(r, z, T) + ρ_thick_stellar_disc(r, z, T) + ρ_thin_stellar_disc(r, z, T)
ρ_ISM(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = ρ_HI(r, z, T) + ρ_H2(r, z, T)

function der_ρ_baryons_spherical(xp::Real, r::Real, ::Type{T}) where {T<:MM17Model}
    y = sqrt(1 - xp^2)
    return xp/y * (ρ_baryons(r * xp, -y, T) + ρ_baryons(r * xp, y, T))
end

ρ_baryons_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = quadgk(xp -> der_ρ_baryons_spherical(xp, r, T), 0, 1, rtol=1e-4)[1] 
m_baryons_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = 4.0 * π * quadgk(rp -> rp^2 * ρ_baryons_spherical(rp, T), 0, r, rtol=1e-3)[1] 
μ_baryons_spherical(x::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = m_baryons_spherical(x * galactic_halo(T).rs, T)/(4*π*galactic_halo(T).ρs * (galactic_halo(T).rs)^3)

ρ_galaxy_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = ρ_baryons_spherical(r, T) + ρ_halo(r, galactic_halo(T))
m_galaxy_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = m_baryons_spherical(r, T) + m_halo(r, galactic_halo(T))
μ_galaxy_spherical(x::Real, ::Type{T} = MM17Gamma1) where {T<:GalaxyModel} = μ_baryons_spherical(x, T) + μ_halo(x, galactic_profile(T))

end # module FSL
