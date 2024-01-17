
module Hosts

include("./Halos.jl")

import QuadGK: quadgk
using JLD2,  Interpolations
import Unitful: km, s, Gyr, K, Myr, NoUnits
import UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

import Main.Cosmojuly.BackgroundCosmo: BkgCosmology, planck18_bkg, lookback_redshift, δt_s, z_to_a
import Main.Cosmojuly.PowerSpectrum: Cosmology, planck18
import Main.Cosmojuly.Halos: Halo, HaloProfile, nfwProfile, αβγProfile, halo_from_ρs_and_rs, m_halo, ρ_halo, μ_halo, coreProfile

export ρ_HI, ρ_H2, ρ_ISM, ρ_baryons, ρ_baryons_spherical, host_halo, m_baryons_spherical, μ_baryons_spherical
export host_profile, ρ_dm, m_host_spherical, μ_host_spherical, ρ_host_spherical, tidal_radius
export HostModel, MM17Model, MM17Gamma1, MM17Gamma0, MM17GammaFree, DMOnlyMM17Gamma0
export _save_host, _load_host

# Definition of the profiles
function ρ_spherical_BG02(r::Real, z::Real; ρ0b::Real, r0::Real, q::Real, α::Real, rcut::Real) 
    rp = sqrt(r^2 + (z/q)^2)
    return ρ0b/((1+rp/r0)^α) * exp(-(rp/rcut)^2)
end

ρ_exponential_disc(r::Real, z::Real; σ0::Real, rd::Real, zd::Real) = σ0/(2.0*zd)*exp(-abs(z)/zd - r/rd)
σ_exponential_disc(r::Real; σ0::Real, rd::Real) = σ0 * exp(-r/rd)
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

name_model(::Type{MM17Gamma1}) = "MM17Gamma1"
name_model(::Type{MM17Gamma0}) = "MM17Gamma0"
name_model(::Type{MM17GammaFree}) = "MM17GammaFree"
name_model(::Type{DMOnlyMM17Gamma0}) = "DMOnlyMM17Gamma0"
name_model(::Type{T}) where {T<:HostModel} = name_model(T)

tidal_radius(::Type{T}) where {T<:MM17Model} = 0.5 # in Mpc
tidal_radius(::Type{T}) where {T<:DMOnlyMM17Model} = 0.5 # in Mpc

export age_host

""" age of the host in s """
function age(z::Real = 0, ::Type{T} = MM17Gamma1, cosmo::BkgCosmology = planck18_bkg; kws...) where {T<:MM17Model} 
    z_max = lookback_redshift((1e+4 * (Myr / s) |> NoUnits))
    return δt_s(z_to_a(z_max), z_to_a(z), cosmo; kws...)
end

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

ρ_HI(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_sech_disc(r, z, σ0=5.31e+13, zd=8.5e-5, rm=4.0e-3, rd=7.0e-3)
ρ_H2(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_sech_disc(r, z, σ0=2.180e+15, zd=4.5e-5, rm=1.2e-2, rd=1.5e-3)
ρ_bulge(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_spherical_BG02(r, z, ρ0b = 9.73e+19, r0 = 7.5e-5, q = 0.5, α = 1.8, rcut = 2.1e-3)
ρ_thick_stellar_disc(r::Real, z::Real, ::Type{T}) where {T <: MM17Model} = ρ_exponential_disc(r, z, σ0=1.487e+14, zd =9.0e-4, rd=3.29e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{MM17Gamma1})    = ρ_exponential_disc(r, z, σ0=8.87e+14, zd=3.0e-4, rd=2.53e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{MM17Gamma0})    = ρ_exponential_disc(r, z, σ0=8.87e+14, zd=3.0e-4, rd=2.36e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{MM17GammaFree}) = ρ_exponential_disc(r, z, σ0=8.87e+14, zd=3.0e-4, rd=2.51e-3)
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:MM17Model} = ρ_thin_stellar_disc(r, z, T)

σ_thick_stellar_disc(r::Real, ::Type{T}) where {T <: MM17Model} = σ_exponential_disc(r, σ0=1.487e+14, rd=3.29e-3)
σ_thin_stellar_disc(r::Real, ::Type{MM17Gamma1})    = σ_exponential_disc(r, σ0=8.87e+14, rd=2.53e-3)
σ_thin_stellar_disc(r::Real, ::Type{MM17Gamma0})    = σ_exponential_disc(r, σ0=8.87e+14, rd=2.36e-3)
σ_thin_stellar_disc(r::Real, ::Type{MM17GammaFree}) = σ_exponential_disc(r, σ0=8.87e+14, rd=2.51e-3)
σ_thin_stellar_disc(r::Real, ::Type{T} = MM17Gamma1) where {T<:MM17Model} = σ_thin_stellar_disc(r, T)

ρ_HI(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
ρ_H2(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
ρ_bulge(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
ρ_thick_stellar_disc(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
ρ_thin_stellar_disc(r::Real, z::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
σ_thick_stellar_disc(r::Real, ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0
σ_thin_stellar_disc(r::Real,  ::Type{T}) where {T <: DMOnlyMM17Model}  = 0.0

ρ_ISM(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = ρ_HI(r, z, T) + ρ_H2(r, z, T)
ρ_stellar_disc(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = ρ_thick_stellar_disc(r, z, T) + ρ_thin_stellar_disc(r, z, T)
ρ_baryons(r::Real, z::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = ρ_ISM(r, z, T) + ρ_bulge(r, z, T) + ρ_stellar_disc(r, z, T) 
σ_stellar_disc(r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = σ_thick_stellar_disc(r, T) + σ_thin_stellar_disc(r, T)


###########################
## SPHERICISED QUANTITIES
export circular_velocity, circular_period, number_circular_orbits, velocity_dispersion_spherical


function der_ρ_baryons_spherical(xp::Real, r::Real, ::Type{T}) where {T<:MM17Model}
    y = sqrt(1 - xp^2)
    return xp^2/y * (ρ_baryons(r * xp, - r * y, T) + ρ_baryons(r * xp, r * y, T))/2.0
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


""" circular velocity in km/s for `r` in Mpc """
circular_velocity(r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = sqrt(G_NEWTON * m_host_spherical(r, T) * Msun / (r * Mpc)) / (km/s) |> NoUnits
circular_velocity(r::Real, h::Halo) = sqrt(G_NEWTON * m_halo(r, h) * Msun / (r * Mpc)) / (km/s) |> NoUnits # should be moves to Halos.jl

""" circular period in s for `r` in Mpc """
circular_period(r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = 2.0 * π * r * Mpc / circular_velocity(r, T) / km  |> NoUnits

export number_circular_orbits

""" number or circular orbits with `r` in Mpc """
number_circular_orbits(r::Real, z::Real = 0, ::Type{T} = MM17Gamma1, cosmo::BkgCosmology = planck18_bkg; kws...) where {T<:HostModel} = floor(Int, age(z, T, cosmo, kws...) / circular_period(r, T))

""" Jeans dispersion (km / s)"""
velocity_dispersion_spherical(r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = sqrt(G_NEWTON * Msun / Mpc / ρ_dm(r, T) *  quadgk(rp -> ρ_dm(rp, T) * m_host_spherical(rp, T)/rp^2, r, tidal_radius(T), rtol=1e-3)[1]) / (km / s)  |> NoUnits 

###########################



function _save_host(::Type{T}) where {T<:HostModel}
    
    rs = host_halo(T).rs
    r_array = 10.0.^range(log10(1e-6 * rs), log10(1e+6 * rs), 1000)

    # -------------------------------------------
    # Checking if the file does not already exist
    hash_value = hash(name_model(T))
    filenames = readdir("../cache/hosts/")

    file = "host_" * string(hash_value, base=16) * ".jld2" 
 
    if file in filenames
        existing_data = jldopen("../cache/hosts/host_" * string(hash_value, base=16) * ".jld2")

        if existing_data["r"] == r_array
            @info "| file to save is already cached"
            return nothing
        end
    end
    # -------------------------------------------

    ρ_host    =  ρ_host_spherical.(r_array, T)
    ρ_baryons = ρ_baryons_spherical.(r_array, T)
    m_host    = m_host_spherical.(r_array, T)
    m_baryons = m_baryons_spherical.(r_array, T)
    
    jldsave("../cache/hosts/host_" * string(hash_value, base=16) * ".jld2"; 
        r = r_array, ρ_host = ρ_host, ρ_baryons = ρ_baryons, m_host = m_host, m_baryons = m_baryons)

    return true

end


## Possibility to interpolate the model
function _load_host(::Type{T}) where {T<:HostModel}
    """ change that to a save function """

    hash_value = hash(name_model(T))
    filenames = readdir("../cache/hosts/")

    file = "host_" * string(hash_value, base=16) * ".jld2" 


    if file in filenames

        data = jldopen("../cache/hosts/" * file)
        r_array = data["r"]
        ρ_host = data["ρ_host"]
        ρ_baryons = data["ρ_baryons"]
        m_host = data["m_host"]
        m_baryons = data["m_baryons"]

    end

    log10ρ_host = interpolate((log10.(r_array),), log10.(ρ_host),  Gridded(Linear()))
    log10ρ_baryons = interpolate((log10.(r_array),), log10.(ρ_baryons),  Gridded(Linear()))
    log10m_host = interpolate((log10.(r_array),), log10.(m_host),  Gridded(Linear()))
    log10m_baryons = interpolate((log10.(r_array),), log10.(m_baryons),  Gridded(Linear()))

    ρ_host_spherical(r::Real) = 10.0^log10ρ_host(log10(r))
    ρ_baryons_spherical(r::Real) = 10.0^log10ρ_baryons(log10(r))
    m_host_spherical(r::Real) = 10.0^log10m_host(log10(r))
    m_baryons_spherical(r::Real) = 10.0^log10m_baryons(log10(r))

    return ρ_host_spherical, ρ_baryons_spherical, m_host_spherical, m_baryons_spherical

end



#######################

end # module Hosts
