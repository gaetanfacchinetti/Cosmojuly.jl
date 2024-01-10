module BackgroundCosmo

using QuadGK, Roots, Unitful
import Unitful: km, s, Gyr, K
import UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

export planck18_bkg

export BkgCosmology, FLRW
export planck18_bkg, edsPlanck18_bkg, convert_cosmo, FlatFLRW
export hubble_constant
export Ω, Ω_vs_a
export z_eq_mr, z_eq_Λm, z_to_a, a_to_z, universe_age, temperature_CMB_K, k_eq_mr_Mpc
export growth_factor, growth_factor_Carroll
export lookback_time, lookback_redshift


@doc raw"""
    BkgCosmology{T<:Real}

Abstract type: generic background cosmology 
"""
abstract type BkgCosmology{T<:Real} end

@doc raw"""
    FLRW{T<:Real} <: BkgCosmology{T}

Abstract type: Generic background cosmology of type FLRW
"""
abstract type FLRW{T<:Real} <: BkgCosmology{T} end

@doc raw"""
    FlatFLRW{T<:Real} <: FLRW{T}

Defines a flat FLRW cosmology
"""
struct FlatFLRW{T<:Real} <: FLRW{T}
    
    # Hubble parameter
    h::T 

    # Abundances of the different components
    Ω_χ0::T
    Ω_b0::T
    Ω_m0::T
    Ω_r0::T
    Ω_γ0::T
    Ω_ν0::T
    Ω_Λ0::T

    # Derived quantities
    z_eq_mr::T
    z_eq_Λm::T
    k_eq_mr_Mpc::T
    
    # Quantity with units
    ρ_c0::T
    T0_CMB_K::T

end


""" Convert cosmo object attributes to another type """
convert_cosmo(::Type{T}, cosmo::FlatFLRW = planck18_bkg) where {T<:Real} = FlatFLRW{T}([convert(T, getfield(cosmo, field)) for field in fieldnames(typeof(cosmo))]...)


# First definition of the abundances
Ω_m(z::Real, Ω_m0::Real, Ω_r0::Real, Ω_Λ0::Real) = Ω_m0 * (1+z)^3 / (Ω_m0 * (1+z)^3 + Ω_r0 * (1+z)^4 + Ω_Λ0)
Ω_r(z::Real, Ω_m0::Real, Ω_r0::Real, Ω_Λ0::Real) = Ω_r0 * (1+z)^4 / (Ω_m0 * (1+z)^3 + Ω_r0 * (1+z)^4 + Ω_Λ0)
Ω_Λ(z::Real, Ω_m0::Real, Ω_r0::Real, Ω_Λ0::Real) = Ω_Λ0  / (Ω_m0 * (1+z)^3 + Ω_r0 * (1+z)^4 + Ω_Λ0)

@doc raw"""
    FlatFLRW(h, Ω_χ0, Ω_b0; T0_CMB_K = 2.72548, Neff = 3.04)

Creates a FlatFLRW instance

# Arguments
- `h::Real` : hubble parameter (dimensionless)
- `Ω_χ0::Real`: cold dark matter abundance today (dimensionless)
- `Ω_b0::Real`: baryon abundance today (dimensionless)
- `T0_CMB_K::Real`: temperature of the CMB today (in Kelvin)
- `Neff::Real`: effective number of neutrinos (dimensionless)
"""
function FlatFLRW(h::Real, Ω_χ0::Real, Ω_b0::Real; T0_CMB_K::Real = 2.72548, Neff::Real = 3.04, EdS::Bool = false)
    
    # Derived abundances
    Ω_γ0 = EdS ? 0.0 : 4.48131e-7 * T0_CMB_K^4 / h^2
    Ω_ν0 = EdS ? 0.0 : Neff * Ω_γ0 * (7 / 8) * (4 / 11)^(4 / 3)
    Ω_r0 = EdS ? 0.0 : Ω_γ0 + Ω_ν0
    Ω_m0 = Ω_χ0 + Ω_b0
    Ω_Λ0 = 1 - Ω_m0 - Ω_r0
 
    ρ_c0 =  3/(8*π*G_NEWTON) * (100 * h * km / s / Mpc )^2 / (Msun / Mpc^3)

    z_eq_mr = 0.0
    z_eq_Λm = 0.0

    try
        z_eq_mr = EdS ? NaN : exp(find_zero( y -> Ω_r(exp(y), Ω_m0, Ω_r0, Ω_Λ0) - Ω_m(exp(y), Ω_m0, Ω_r0, Ω_Λ0), (-10, 10), Bisection())) 
        z_eq_Λm = exp(find_zero( y -> Ω_Λ(exp(y), Ω_m0, Ω_r0, Ω_Λ0) - Ω_m(exp(y), Ω_m0, Ω_r0, Ω_Λ0), (-10, 10), Bisection())) 
    catch e
        println("Impossible to definez z_eq_mr and/or z_eq_Λm for this cosmology")
        println("Error: ", e)
    end

    k_eq_mr_Mpc =  EdS ? NaN : Ω_r0 / Ω_m0 * (100. * h * sqrt(Ω_m0 * (1+z_eq_mr)^3 + Ω_r0 * (1+z_eq_mr)^4 + Ω_Λ0) * km / s / c_0) |> NoUnits

    return FlatFLRW(promote(h, Ω_χ0, Ω_b0, Ω_m0, Ω_r0, Ω_γ0, Ω_ν0, Ω_Λ0, z_eq_mr, z_eq_Λm, k_eq_mr_Mpc, ρ_c0, T0_CMB_K)...) 
end

#########################################################
# Predfinition of some cosmologies
"""
    planck18_bkg::FlatFLRW = FlatFLRW(0.6736, 0.26447, 0.04930)

Flat FLRW cosmology with [*Planck18*](https://arxiv.org/abs/1807.06209) parameters
"""
const planck18_bkg::FlatFLRW = FlatFLRW(0.6736, 0.26447, 0.04930)

"""
    edsplanck18_bkg::FlatFLRW  = FlatFLRW(0.6736, 0.3, 0) 

Flat FLRW cosmology with [*Planck18*](https://arxiv.org/abs/1807.06209) parameters and no baryons
"""
const edsPlanck18_bkg::FlatFLRW  = FlatFLRW(0.6736, 0.3, 0, EdS = true) 
#########################################################

#########################################################
# Definition of the densities

""" CMB temperature (in K) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
temperature_CMB_K(z::Real = 0, cosmo::BkgCosmology = planck18_bkg)::Real  = cosmo.T0_CMB_K * (1+z)

""" Hubble constant H0 (in km/s/Mpc) for the cosmology `cosmo` """
hubble_constant(cosmo::BkgCosmology = planck18_bkg)::Real = 100 * cosmo.h

""" Hubble evolution (no dimension) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
hubble_evolution(z::Real = 0, cosmo::BkgCosmology = planck18_bkg)::Real = sqrt(cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)

""" Hubble rate H(z) (in km/s/Mpc) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
hubble_rate(z::Real = 0, cosmo::BkgCosmology = planck18_bkg):: Real = hubble_evolution(z, cosmo) .* hubble_constant(cosmo) 
#########################################################

export Species, Radiation, Photons, Neutrinos, Matter, ColdDarkMatter, Baryons, DarkEnergy

abstract type Species end

abstract type Radiation <: Species end
abstract type Photons <: Species end
abstract type Neutrinos <: Species end
abstract type Matter <: Species end
abstract type ColdDarkMatter <: Species end
abstract type Baryons <: Species end
abstract type DarkEnergy <: Species end

#########################################################
# Definition of the densities
# All densities are in units of Msun / Mpc^3 

export ρ_b_Msun_Mpc3, ρ_Msun_Mpc3

""" Critical density (in Msun/Mpc^3) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
ρ_c_Msun_Mpc3(z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = cosmo.ρ_c0 * (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)

""" Radiation density (in Msun/Mpc^3) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
ρ_r_Msun_Mpc3(z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = cosmo.Ω_r0 * cosmo.ρ_c0 * (1+z)^4

""" Photon density (in Msun/Mpc^3) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
ρ_γ_Msun_Mpc3(z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = cosmo.Ω_γ0 * cosmo.ρ_c0 * (1+z)^4

""" Neutrino density (in Msun/Mpc^3) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
ρ_ν_Msun_Mpc3(z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = cosmo.Ω_ν0 * cosmo.ρ_c0 * (1+z)^4

""" Matter density (in Msun/Mpc^3) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
ρ_m_Msun_Mpc3(z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = cosmo.Ω_m0 * cosmo.ρ_c0 * (1+z)^3

""" Cold dark matter density (in Msun/Mpc^3) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
ρ_χ_Msun_Mpc3(z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = cosmo.Ω_χ0 * cosmo.ρ_c0 * (1+z)^3

""" Baryon density (in Msun/Mpc^3) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
ρ_b_Msun_Mpc3(z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = cosmo.Ω_b0 * cosmo.ρ_c0 * (1+z)^3

""" Cosmological constant density (in Msun/Mpc^3) of the Universe at redshift `z` (by default z=0) for the cosmology `cosmo` """
ρ_Λ_Msun_Mpc3(z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = cosmo.Ω_Λ0 * cosmo.ρ_c0


ρ_Msun_Mpc3(::Type{Radiation}, z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = ρ_r_Msun_Mpc3(z, cosmo)
ρ_Msun_Mpc3(::Type{Photons}, z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = ρ_γ_Msun_Mpc3(z, cosmo)
ρ_Msun_Mpc3(::Type{Neutrinos}, z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = ρ_ν_Msun_Mpc3(z, cosmo)
ρ_Msun_Mpc3(::Type{Matter}, z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = ρ_m_Msun_Mpc3(z, cosmo)
ρ_Msun_Mpc3(::Type{Baryons}, z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = ρ_b_Msun_Mpc3(z, cosmo)
ρ_Msun_Mpc3(::Type{DarkEnergy}, z::Real = 0, cosmo::BkgCosmology = planck18_bkg) = ρ_Λ_Msun_Mpc3(z, cosmo)

@doc raw"""
    ρ_Msun_Mpc3(T, z = 0, cosmo = planck18_bkg) where {T<:Species}

Energy density of species `T` at redshift `z` and for the background cosmology `cosmo`
"""
ρ_Msun_Mpc3(::Type{T}, z::Real = 0.0, cosmo::BkgCosmology = planck18_bkg) where {T<:Species} = ρ_Msun_Mpc3(T, z, cosmo)
#########################################################



#########################################################
# Definition of the abundances
Ω_r(z::Real = 0, cosmo::BkgCosmology = planck18_bkg)::Real = (cosmo.Ω_r0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_γ(z::Real = 0, cosmo::BkgCosmology = planck18_bkg)::Real = (cosmo.Ω_γ0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_ν(z::Real = 0, cosmo::BkgCosmology = planck18_bkg)::Real = (cosmo.Ω_ν0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_m(z::Real = 0, cosmo::BkgCosmology = planck18_bkg)::Real = (cosmo.Ω_m0 * (1+z)^3) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_χ(z::Real = 0, cosmo::BkgCosmology = planck18_bkg)::Real = (cosmo.Ω_χ0 * (1+z)^3) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_b(z::Real = 0, cosmo::BkgCosmology = planck18_bkg)::Real = (cosmo.Ω_b0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_Λ(z::Real = 0, cosmo::BkgCosmology = planck18_bkg)::Real = cosmo.Ω_Λ0 / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)

Ω(::Type{Radiation}, z::Real = 0.0, cosmo::BkgCosmology = planck18_bkg) = (cosmo.Ω_r0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω(::Type{Photons}, z::Real = 0.0, cosmo::BkgCosmology = planck18_bkg)   = (cosmo.Ω_γ0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω(::Type{Neutrinos}, z::Real = 0.0, cosmo::BkgCosmology = planck18_bkg) = (cosmo.Ω_ν0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω(::Type{Matter}, z::Real = 0.0, cosmo::BkgCosmology = planck18_bkg)         = (cosmo.Ω_m0 * (1+z)^3) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω(::Type{ColdDarkMatter}, z::Real = 0.0, cosmo::BkgCosmology = planck18_bkg) = (cosmo.Ω_χ0 * (1+z)^3) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω(::Type{Baryons}, z::Real = 0.0, cosmo::BkgCosmology = planck18_bkg)    = (cosmo.Ω_b0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω(::Type{DarkEnergy}, z::Real = 0.0, cosmo::BkgCosmology = planck18_bkg) =  cosmo.Ω_Λ0 / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)

@doc raw"""
    Ω(T, z = 0, cosmo = planck18_bkg) where {T<:Species}

Abundance of species `T` at redshift `z` and for the background cosmology `cosmo`
"""
Ω(::Type{T}, z::Real = 0.0, cosmo::BkgCosmology = planck18_bkg) where {T<:Species} = Ω(T, z, cosmo)
Ω_vs_a(::Type{T}, a::Real = 1.0,  cosmo::BkgCosmology = planck18_bkg) where {T<:Species} = Ω(T, a_to_z(a), cosmo)
#########################################################

k_eq_mr_Mpc(cosmo::BkgCosmology)::Real = cosmo.k_eq_mr_Mpc

#########################################################
# Definition of specific time quantities
z_to_a(z::Real)::Real = 1 /(1 + z)
a_to_z(a::Real)::Real = 1 / a - 1
z_eq_mr(cosmo::BkgCosmology = planck18_bkg) = cosmo.z_eq_mr
z_eq_Λm(cosmo::BkgCosmology = planck18_bkg) = cosmo.z_eq_Λm
a_eq_mr(cosmo::BkgCosmology = planck18_bkg) = z_to_a(cosmo.z_eq_mr)
a_eq_Λm(cosmo::BkgCosmology = planck18_bkg) = z_to_a(cosmo.z_eq_Λm)
δt_s(a0::Real, a1::Real, cosmo::BkgCosmology = planck18_bkg; kws...) = QuadGK.quadgk(a -> 1.0 / hubble_evolution(a_to_z(a), cosmo) / a, a0, a1, rtol=1e-3; kws...)[1] / (hubble_constant(cosmo) * km / Mpc) |> NoUnits
""" age of the universe (s)"""
universe_age(z::Real=0, cosmo::BkgCosmology = planck18_bkg; kws...) = δt_s(0, z_to_a(z), cosmo; kws...)
""" lookback time of the Universe (s) """
lookback_time(z::Real, cosmo::BkgCosmology = planck18_bkg; kws...) = δt_s(z_to_a(z), 1, cosmo; kws...)
""" lookback redshift of the Universe for t in (s) """
lookback_redshift(t::Real, cosmo::BkgCosmology = planck18_bkg; kws...) = exp(find_zero(lnz -> lookback_time(exp(lnz), cosmo; kws...)-t, (-5, 10), Bisection())) 
#########################################################


#########################################################
# Functions related to perturbations
"""
    growth_factor(z, cosmo)

    Exact growth factor in a matter-Λ Universe computed from an integral
    Carroll et al. 1992 (Mo and White p. 172)
    Corresponds to D1(a=1) with the definition of Dodelson 2003

# Arguments
- z: redshift
- cosmo: background cosmology (default Planck18)
"""
function growth_factor(z::Real, cosmo::BkgCosmology = planck18_bkg, kws...)::Real
    norm = 2.5 * cosmo.Ω_m0 * sqrt(cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_Λ0)
    return norm * QuadGK.quadgk(a -> (cosmo.Ω_m0 * a^(-1) + cosmo.Ω_Λ0 * a^2)^(-3/2), 0, z_to_a(z), rtol=1e-6; kws...)[1]
end

"""
    growth_factor_Carroll(z, cosmo)

    Approximate growth factor in a matter-Λ Universe (faster than growth_factor)
    Carroll et al. 1992 (Mo and White p. 172)
    Corresponds to D1(a=1) with the definition of Dodelson 2003

# Arguments
- z: redshift
- cosmo: background cosmology (default Planck18)
"""
function growth_factor_Carroll(z::Real, cosmo=BkgCosmology = planck18_bkg)::Real
    # Abundances in a Universe with no radiation
    _Ω_m = cosmo.Ω_m0 * (1+z)^3 / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_Λ0)
    _Ω_Λ = cosmo.Ω_Λ0 / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_Λ0)
    return 2.5*_Ω_m/(_Ω_m^(4.0/7.0) - _Ω_Λ + (1.0 + 0.5*_Ω_m) * (1.0 + 1.0/70.0*_Ω_Λ))/(1+z)
end
#########################################################


end # module BackgroundCosmo
