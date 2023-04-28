module MyCosmology

using QuadGK, Roots, Unitful
import Unitful: km, s, Gyr, K, Temperature, DimensionlessQuantity, Density
using UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

export FLRWPLANCK18, FLRW

@derived_dimension Mode dimension(1/km)

abstract type Cosmology{T<:Real} end
abstract type FLRW{T} <: Cosmology{T} end

# Define a structure for FlatFLRW
struct FlatFLRW{T} <: FLRW{T}
    
    h::DimensionlessQuantity{T} # Hubble parameter

    Ω_χ0::DimensionlessQuantity{T}
    Ω_b0::DimensionlessQuantity{T}
    Ω_m0::DimensionlessQuantity{T}
    Ω_r0::DimensionlessQuantity{T}
    Ω_γ0::DimensionlessQuantity{T}
    Ω_ν0::DimensionlessQuantity{T}
    Ω_Λ0::DimensionlessQuantity{T}

    z_eq_mr::DimensionlessQuantity{T}
    z_eq_Λm::DimensionlessQuantity{T}
    
    # Quantity with units
    ρ_c0::Density{T}
    T0_CMB::Temperature{T}

end

# First definition of the abundances
Ω_m(z::Real, Ω_m0::Real, Ω_r0::Real, Ω_Λ0::Real) = (Ω_m0 * (1+z)^3) / (Ω_m0 * (1+z)^3 + Ω_r0 * (1+z)^4 + Ω_Λ0)
Ω_r(z::Real, Ω_m0::Real, Ω_r0::Real, Ω_Λ0::Real) = (Ω_r0 * (1+z)^4) / (Ω_m0 * (1+z)^3 + Ω_r0 * (1+z)^4 + Ω_Λ0)
Ω_Λ(z::Real, Ω_m0::Real, Ω_r0::Real, Ω_Λ0::Real) = Ω_Λ0  / (Ω_m0 * (1+z)^3 + Ω_r0 * (1+z)^4 + Ω_Λ0)

# Constructor of the FlatBaseLCDM structure
function FlatFLRW(h::Real, Ω_χ0::Real, Ω_b0::Real; T0_CMB::Unitful.Temperature{<:Real} = 2.72548 * K, Neff::Real = 3.04)
    
    # Derived abundances
    Ω_γ0 = 4.48131e-7 / K^4 * T0_CMB^4 / h^2
    Ω_ν0 = Neff * Ω_γ0 * (7 / 8) * (4 / 11)^(4 / 3)
    Ω_r0 = Ω_γ0 + Ω_ν0
    Ω_m0 = Ω_χ0 + Ω_b0
    Ω_Λ0 = 1 - Ω_m0 - Ω_r0
 
    ρ_c0 =  3/(8*π*G_NEWTON) * (h * 100 * km / s / Mpc )^2 |> Msun / Mpc^3

    z_eq_mr = 0.0
    z_eq_Λm = 0.0

    try
        z_eq_mr = exp(find_zero( y -> Ω_r(exp(y), Ω_m0, Ω_r0, Ω_Λ0) - Ω_m(exp(y), Ω_m0, Ω_r0, Ω_Λ0), (-10, 10), Bisection())) 
        z_eq_Λm = exp(find_zero( y -> Ω_Λ(exp(y), Ω_m0, Ω_r0, Ω_Λ0) - Ω_m(exp(y), Ω_m0, Ω_r0, Ω_Λ0), (-10, 10), Bisection())) 
    catch e
        println("Impossible to definez z_eq_mr and/or z_eq_Λm for this cosmology")
        println("Error: ", e)
    end

    return FlatFLRW(promote(h, Ω_χ0, Ω_b0, Ω_m0, Ω_r0, Ω_γ0, Ω_ν0, Ω_Λ0, z_eq_mr, z_eq_Λm, ρ_c0, T0_CMB)...)
    
end


# global constant defining the cosmology used
const FLRWPLANCK18::FlatFLRW = FlatFLRW(0.6736, 0.26447, 0.04930)
const EDSPLANCK18::FlatFLRW  = FlatFLRW(0.6736, 0.3, 0) 


T_CMB(z::Real, cosmo::Cosmology = FLRWPLANCK18) = cosmo.T0_CMB * (1 .+ z)
hubble_constant(cosmo::Cosmology = FLRWPLANCK18) = cosmo.h * 100 * km / s / Mpc
hubble_evolution(z::Real, cosmo::Cosmology = FLRWPLANCK18) = sqrt(cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
hubble_rate(z::Real, cosmo::Cosmology = FLRWPLANCK18) = hubble_evolution(z, cosmo) .* hubble_constant(cosmo) 

ρ_c(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = cosmo.ρ_c0 * (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)

# Definition of the densities
ρ_r(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = cosmo.Ω_r0 * cosmo.ρ_c0 * (1+z)^4
ρ_γ(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = cosmo.Ω_γ0 * cosmo.ρ_c0 * (1+z)^4
ρ_ν(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = cosmo.Ω_ν0 * cosmo.ρ_c0 * (1+z)^4
ρ_m(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = cosmo.Ω_m0 * cosmo.ρ_c0 * (1+z)^3
ρ_χ(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = cosmo.Ω_χ0 * cosmo.ρ_c0 * (1+z)^3
ρ_b(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = cosmo.Ω_b0 * cosmo.ρ_c0 * (1+z)^3
ρ_Λ(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = cosmo.Ω_Λ0 * cosmo.ρ_c0

Ω_r(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = (cosmo.Ω_r0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_γ(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = (cosmo.Ω_γ0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_ν(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = (cosmo.Ω_ν0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_m(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = (cosmo.Ω_m0 * (1+z)^3) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_χ(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = (cosmo.Ω_χ0 * (1+z)^3) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_b(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = (cosmo.Ω_b0 * (1+z)^4) / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)
Ω_Λ(z::Real = 0, cosmo::Cosmology = FLRWPLANCK18) = cosmo.Ω_Λ0 / (cosmo.Ω_m0 * (1+z)^3 + cosmo.Ω_r0 * (1+z)^4 + cosmo.Ω_Λ0)

# Definition of specific time quantities
z_to_a(z::Real) = 1 ./(1 .+ z)
a_to_z(a::Real) = 1 ./ a .- 1
z_eq_mr(cosmo::Cosmology = FLRWPLANCK18) = cosmo.z_eq_mr
z_eq_Λm(cosmo::Cosmology = FLRWPLANCK18) = cosmo.z_eq_Λm
a_eq_mr(cosmo::Cosmology = FLRWPLANCK18) = z_to_a(cosmo.z_eq_mr)
a_eq_Λm(cosmo::Cosmology = FLRWPLANCK18) = z_to_a(cosmo.z_eq_Λm)
cosmic_time_difference(a0, a1, cosmo::Cosmology = FLRWPLANCK18; kws...) = QuadGK.quadgk(a -> a / hubble_evolution(a_to_z(a), cosmo), a0, a1; kws...)[1] / hubble_constant(cosmo)  |> s
age(z=0, cosmo::Cosmology = FLRWPLANCK18; kws...) = cosmic_time_difference(0, z_to_a(z), cosmo; kws...)
lookback_time(z, cosmo::Cosmology = FLRWPLANCK18; kws...) = cosmic_time_difference(z_to_a(z), 1, cosmo; kws...)


end # module Cosmology
