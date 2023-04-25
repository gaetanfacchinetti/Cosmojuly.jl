module MyCosmology

using Unitful
import Unitful: km, s, Gyr, K
using UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

export BASEPLANCK18, AbstractBaseCosmology

abstract type AbstractBaseCosmology end
abstract type AbstractBaseFlatCosmology <: AbstractBaseCosmology end
abstract type AbstractFlatCosmology <: AbstractBaseFlatCosmology end

# Define a structure for BaseLc
struct BaseFlatLCDM{T <: Real} <: AbstractBaseFlatCosmology
    
    hubble_parameter::T # Hubble parameter
    Ω_c::T
    Ω_b::T
    Ω_m::T
    Ω_r::T
    Ω_γ::T
    Ω_ν::T
    Ω_Λ::T

    # Quantity with units
    T_CMB::Unitful.Temperature{T}

end

# Constructor of the FlatBaseLc
function BaseFlatLCDM(hubble_parameter::Real, Ω_c::Real, Ω_b::Real, T_CMB::Unitful.Temperature = 2.72548 * K, Neff::Real = 3.04)
    
    T_CMB_K = Unitful.ustrip(K, T_CMB) 

    hubble_parameter, Ω_c, Ω_b, T_CMB_K = promote(float(hubble_parameter), float(Ω_c), float(Ω_b), float(T_CMB_K))
    T_CMB = T_CMB_K * K

    # Derived abundances
    Ω_γ = 4.48131e-7 * T_CMB_K^4 / hubble_parameter^2
    Ω_ν = Neff * Ω_γ * (7 / 8) * (4 / 11)^(4 / 3)
    Ω_r = Ω_γ + Ω_ν
    Ω_m = Ω_c + Ω_b
    Ω_Λ = 1 - Ω_m - Ω_r
    
    return BaseFlatLCDM(hubble_parameter, Ω_c, Ω_b, Ω_m, Ω_r, Ω_γ, Ω_ν, Ω_Λ, T_CMB)
end

BASEPLANCK18 = BaseFlatLCDM(0.6736, 0.26447, 0.04930)

# Defining the abundances
Ω_r(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}} = 0) = cosmo.Ω_r .* (1 .+ z).^4
Ω_γ(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}} = 0) = cosmo.Ω_γ .* (1 .+ z).^4
Ω_ν(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}} = 0) = cosmo.Ω_ν .* (1 .+ z).^4
Ω_m(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}} = 0) = cosmo.Ω_m .* (1 .+ z).^3
Ω_c(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}} = 0) = cosmo.Ω_c .* (1 .+ z).^3
Ω_b(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}} = 0) = cosmo.Ω_b .* (1 .+ z).^3
Ω_Λ(cosmo::AbstractBaseCosmology) = cosmo.Ω_Λ

T_CMB(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}}) = cosmo.T_CMB .* (1 .+ z)
hubble_constant(cosmo::AbstractBaseCosmology) = cosmo.hubble_parameter * 100 * km / s / Mpc
hubble_evolution(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}}) = sqrt.(Ω_m(cosmo, z) .+ Ω_r(cosmo, z) .+ Ω_Λ(cosmo))
hubble_rate(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}}) = hubble_evolution(cosmo, z) .* hubble_constant(cosmo)

ρ_c(cosmo::AbstractBaseCosmology, z::Union{Real,Vector{<:Real}} = 0) =  3/(8*π*G_NEWTON) * hubble_rate(cosmo, z).^2 |> Msun / Mpc^3


end # module Cosmology
