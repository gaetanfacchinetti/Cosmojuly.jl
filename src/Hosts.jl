
module Hosts

include("./Halos.jl")

import QuadGK: quadgk

import Main.Cosmojuly.BackgroundCosmo: BkgCosmology, planck18_bkg
import Main.Cosmojuly.PowerSpectrum: Cosmology, planck18
import Main.Cosmojuly.Halos: Halo, HaloProfile, nfwProfile, αβγProfile, halo_from_ρs_and_rs, m_halo, ρ_halo, μ_halo, coreProfile

export ρ_HI, ρ_baryons_spherical, host_halo, m_baryons_spherical, μ_baryons_spherical
export host_profile, ρ_dm, m_host_spherical, μ_host_spherical, ρ_host_spherical
export HostModel, MM17Model, MM17Gamma1, MM17Gamma0, MM17GammaFree

# Definition of the profiles
function ρ_spherical_BG02(r::Real, z::Real; ρ0b::Real, r0::Real, q::Real, α::Real, rcut::Real) 
    rp = sqrt(r^2 + (z/q)^2)
    return ρ0b/(1+rp/r0)^α * exp(-(rp/rcut)^2)
end

ρ_exponential_disc(r::Real, z::Real; σ0::Real, rd::Real, zd::Real) = σ0/(2.0*zd)*exp(-abs(z)/zd - r/rd)
ρ_sech_disc(r::Real, z::Real; σ0::Real, rd::Real, rm::Real, zd::Real) = σ0/(4.0*zd)*exp(-rm/r - r/rd)*(sech(z/(2.0*zd)))^2

abstract type HostModel end

abstract type MM17Model <: HostModel end
abstract type MM17Gamma1 <: MM17Model end
abstract type MM17Gamma0 <: MM17Model end
abstract type MM17GammaFree <: MM17Model end

abstract type DMOnlyMM17Model <: HostModel end
abstract type DMOnlyMM17Gamma1 <: DMOnlyMM17Model end
abstract type DMOnlyMM17Gamma0 <: DMOnlyMM17Model end
abstract type DMOnlyMM17GammaFree <: DMOnlyMM17Model end

# we take the median value of the MCMC
host_profile(::Type{MM17Gamma1})    = nfwProfile
host_profile(::Type{MM17Gamma0})    = coreProfile
host_profile(::Type{MM17GammaFree}) = αβγProfile(1, 3, 0.79)

host_profile(::Type{DMOnlyMM17Gamma1})    = nfwProfile
host_profile(::Type{DMOnlyMM17Gamma0})    = coreProfile
host_profile(::Type{DMOnlyMM17GammaFree}) = αβγProfile(1, 3, 0.79)

host_halo(::Type{MM17Gamma1})    = halo_from_ρs_and_rs(host_profile(MM17Gamma1), 9.24412866426226e+15, 1.86e-2)
host_halo(::Type{MM17Gamma0})    = halo_from_ρs_and_rs(host_profile(MM17Gamma0), 9.086059744049174e+16, 7.7e-3)
host_halo(::Type{MM17GammaFree}) = halo_from_ρs_and_rs(host_profile(MM17GammaFree), 1.5300738158389776e+16, 1.54e-2)

host_halo(::Type{DMOnlyMM17Gamma1})    = halo_from_ρs_and_rs(host_profile(MM17Gamma1), 9.24412866426226e+15, 1.86e-2)
host_halo(::Type{DMOnlyMM17Gamma0})    = halo_from_ρs_and_rs(host_profile(MM17Gamma0), 9.086059744049174e+16, 7.7e-3)
host_halo(::Type{DMOnlyMM17GammaFree}) = halo_from_ρs_and_rs(host_profile(MM17GammaFree), 1.5300738158389776e+16, 1.54e-2)

host_profile(::Type{T} = MM17Gamma1) where {T<:HostModel} = host_profile(T)
host_halo(::Type{T} = MM17Gamma1) where {T<:HostModel} = host_halo(T)
ρ_dm(r::Real, ::Type{T} = MM17Gamma1) where {T <:HostModel} = ρ_halo(r, host_halo(T))

ρ_HI(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_sech_disc(r, z, σ0=5.31e+13, rd=8.5e-5, rm=4.0e-3, zd=7.0e-3)
ρ_H2(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_sech_disc(r, z, σ0=2.180e+15, rd=4.5e-5, rm=1.2e-2, zd=1.5e-3)
ρ_bulge(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_spherical_BG02(r, z, ρ0b = 9.73e+19, r0 = 7.5e-5, q = 0.5, α = 1.8, rcut = 2.1e-3)
ρ_thick_stellar_disc(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_exponential_disc(r, z, σ0=1.487e+14, rd =9.0e-4, zd=3.29e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{MM17Gamma1})    = ρ_exponential_disc(r, z, σ0=8.87e+14, rd=3.0e-4, zd=2.53e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{MM17Gamma0})    = ρ_exponential_disc(r, z, σ0=8.87e+14, rd=3.0e-4, zd=2.36e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{MM17GammaFree}) = ρ_exponential_disc(r, z, σ0=8.87e+14, rd=3.0e-4, zd=2.51e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:MM17Model} = ρ_thin_stellar_disc(r, z, T)

ρ_HI(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
ρ_H2(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
ρ_bulge(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
ρ_thick_stellar_disc(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0

ρ_baryons(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = ρ_HI(r, z, T) + ρ_H2(r, z, T) + ρ_bulge(r, z, T) + ρ_thick_stellar_disc(r, z, T) + ρ_thin_stellar_disc(r, z, T)
ρ_ISM(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = ρ_HI(r, z, T) + ρ_H2(r, z, T)

function der_ρ_baryons_spherical(xp::Real, r::Real, ::Type{T}) where {T<:MM17Model}
    y = sqrt(1 - xp^2)
    return xp/y * (ρ_baryons(r * xp, -y, T) + ρ_baryons(r * xp, y, T))
end
der_ρ_baryons_spherical(xp::Real, r::Real, ::Type{T}) where {T<:DMOnlyMM17Model} = 0

ρ_baryons_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = quadgk(xp -> der_ρ_baryons_spherical(xp, r, T), 0, 1, rtol=1e-4)[1] 
m_baryons_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = 4.0 * π * quadgk(rp -> rp^2 * ρ_baryons_spherical(rp, T), 0, r, rtol=1e-3)[1] 
μ_baryons_spherical(x::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = m_baryons_spherical(x * host_halo(T).rs, T)/(4*π*host_halo(T).ρs * (host_halo(T).rs)^3)

ρ_baryons_spherical(r::Real, ::Type{T}) where {T<:DMOnlyMM17Model} = 0.0
m_baryons_spherical(r::Real, ::Type{T}) where {T<:DMOnlyMM17Model} = 0.0
μ_baryons_spherical(r::Real, ::Type{T}) where {T<:DMOnlyMM17Model} = 0.0

ρ_host_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = ρ_baryons_spherical(r, T) + ρ_halo(r, host_halo(T))
m_host_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = m_baryons_spherical(r, T) + m_halo(r, host_halo(T))
μ_host_spherical(x::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = μ_baryons_spherical(x, T) + μ_halo(x, host_profile(T))

## Possibility to interpolate the model
function _interpolate_host(::Type{T}) where {T<:HostModel}
    """ change that to a save function """

    rs = host_halo(T).rs
    log10_r_arr = range(log10(1e-5 * rs), log10(1e+5 * rs), 1000)

    log10ρ_host = interpolate((log10_r_arr,), log10.(ρ_host_spherical(10.0.^log10_r_arr, T)),  Gridded(Linear()))
    log10ρ_baryons = interpolate((log10_r_arr,), log10.(ρ_baryons_spherical(10.0.^log10_r_arr, T)),  Gridded(Linear()))
    log10m_host = interpolate((log10_r_arr,), log10.(m_host_spherical(10.0.^log10_r_arr, T)),  Gridded(Linear()))
    log10m_baryons = interpolate((log10_r_arr,), log10.(m_baryons_spherical(10.0.^log10_r_arr, T)),  Gridded(Linear()))

    ρ_host_spherical(r::Real) = 10.0^log10ρ_host(log10(r))

end

end # module Hosts
