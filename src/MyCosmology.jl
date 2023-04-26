module MyCosmology

using QuadGK, Roots, Unitful
import Unitful: km, s, Gyr, K
using UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

export BASEPLANCK18, AbstractBaseCosmology

abstract type AbstractBaseCosmology end
abstract type AbstractBaseFlatCosmology <: AbstractBaseCosmology end
abstract type AbstractFlatCosmology <: AbstractBaseFlatCosmology end

# Define a structure for BaseFlatLCDM
struct BaseFlatLCDM{T <: Real} <: AbstractBaseFlatCosmology
    
    hubble_parameter::T # Hubble parameter

    Ω_χ0::T
    Ω_b0::T
    Ω_m0::T
    Ω_r0::T
    Ω_γ0::T
    Ω_ν0::T
    Ω_Λ0::T
    ρ_c0::Unitful.Density{T}

    # Quantity with units
    T0_CMB::Unitful.Temperature{T}

end

# Constructor of the FlatBaseLCDM
function BaseFlatLCDM(hubble_parameter::Real, Ω_χ0::Real, Ω_b0::Real, T0_CMB::Unitful.Temperature = 2.72548 * K, Neff::Real = 3.04)
    
    T0_CMB_K = Unitful.ustrip(K, T0_CMB) 

    hubble_parameter, Ω_c0, Ω_b0, T0_CMB_K = promote(float(hubble_parameter), float(Ω_χ0), float(Ω_b0), float(T0_CMB_K))
    T0_CMB = T0_CMB_K * K

    # Derived abundances
    Ω_γ0 = 4.48131e-7 * T0_CMB_K^4 / hubble_parameter^2
    Ω_ν0 = Neff * Ω_γ0 * (7 / 8) * (4 / 11)^(4 / 3)
    Ω_r0 = Ω_γ0 + Ω_ν0
    Ω_m0 = Ω_χ0 + Ω_b0
    Ω_Λ0 = 1 - Ω_m0 - Ω_r0
 
    ρ_c0 =  3/(8*π*G_NEWTON) * (hubble_parameter * 100 * km / s / Mpc )^2 |> Msun / Mpc^3
    
    return BaseFlatLCDM(hubble_parameter, Ω_χ0, Ω_b0, Ω_m0, Ω_r0, Ω_γ0, Ω_ν0, Ω_Λ0, ρ_c0, T0_CMB)
    
end

BASEPLANCK18 = BaseFlatLCDM(0.6736, 0.26447, 0.04930)

T_CMB(z::Union{Real,Vector{<:Real}}, cosmo::AbstractBaseCosmology = BASEPLANCK18) = cosmo.T0_CMB .* (1 .+ z)
hubble_constant(cosmo::AbstractBaseCosmology = BASEPLANCK18) = cosmo.hubble_parameter * 100 * km / s / Mpc
hubble_evolution(z::Union{Real,Vector{<:Real}}, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. sqrt(cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
hubble_rate(z::Union{Real,Vector{<:Real}}, cosmo::AbstractBaseCosmology = BASEPLANCK18) = hubble_evolution(z, cosmo) .* hubble_constant(cosmo)

ρ_c(z::Union{Real,Vector{<:Real}}, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. cosmo.ρ_c0 * (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)

# Definition of the densities
ρ_r(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. cosmo.Ω_r0 * cosmo.ρ_c0 * (1+z)^4
ρ_γ(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. cosmo.Ω_γ0 * cosmo.ρ_c0 * (1+z)^4
ρ_ν(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. cosmo.Ω_ν0 * cosmo.ρ_c0 * (1+z)^4
ρ_m(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. cosmo.Ω_m0 * cosmo.ρ_c0 * (1+z)^3
ρ_χ(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. cosmo.Ω_χ0 * cosmo.ρ_c0 * (1+z)^3
ρ_b(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. cosmo.Ω_b0 * cosmo.ρ_c0 * (1+z)^3
ρ_Λ(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = repeat([cosmo.Ω_Λ0 * cosmo.ρ_c0], length(z))

# Definition of the abundances
Ω_r(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. (cosmo.Ω_r0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_γ(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. (cosmo.Ω_γ0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_ν(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. (cosmo.Ω_ν0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_m(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. (cosmo.Ω_m0 * (1+z)^3) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_χ(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. (cosmo.Ω_χ0 * (1+z)^3) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_b(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. (cosmo.Ω_b0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_Λ(z::Union{Real,Vector{<:Real}} = 0, cosmo::AbstractBaseCosmology = BASEPLANCK18) = @. cosmo.Ω_Λ0 / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)

# Definition of specific time quantities
z_to_a(z::Union{Real,Vector{<:Real}}) = 1 ./(1 .+ z)
a_to_z(a::Union{Real,Vector{<:Real}}) = 1 ./ a .- 1
z_eq_mr(cosmo::AbstractBaseCosmology = BASEPLANCK18) = exp(find_zero( y -> Ω_r(exp(y), cosmo) - Ω_m(exp(y), cosmo), (-10, 10), Bisection())) 
z_eq_Λm(cosmo::AbstractBaseCosmology = BASEPLANCK18) = exp(find_zero( y -> Ω_Λ(exp(y), cosmo) - Ω_m(exp(y), cosmo), (-10, 10), Bisection())) 
a_eq_mr(cosmo::AbstractBaseCosmology = BASEPLANCK18) = z_to_a(z_eq_mr(cosmo))
a_eq_Λm(cosmo::AbstractBaseCosmology = BASEPLANCK18) = z_to_a(z_eq_Λr(cosmo))
cosmic_time_difference(a0, a1, cosmo::AbstractBaseCosmology = BASEPLANCK18; kws...) = QuadGK.quadgk(a -> a / hubble_evolution(a_to_z(a), cosmo), a0, a1; kws...)[1] / hubble_constant(cosmo)  |> s
age(z=0, cosmo::AbstractBaseCosmology = BASEPLANCK18; kws...) = cosmic_time_difference(0, z_to_a(z), cosmo; kws...)
lookback_time(z, cosmo::AbstractBaseCosmology = BASEPLANCK18; kws...) = cosmic_time_difference(z_to_a(z), 1, cosmo; kws...)


end # module Cosmology
