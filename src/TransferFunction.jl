module TransferFunction

include("../src/MyCosmology.jl")

using Unitful
import Unitful: km, s, Gyr, K, Temperature, DimensionlessQuantity, Density, Volume
using UnitfulAstro: Mpc, Gpc, Msun

import .MyCosmology: FLRWPLANCK18, FLRW, ρ_c, Cosmology

@derived_dimension Mode dimension(1/km)

abstract type AbstractTranferFunction{T<:Real} end

struct TransferFunctionEH98{T, S<:Cosmology{T}} <: AbstractTranferFunction{T} 
    
    with_baryons::Bool
    cosmo::S
    z_eq::DimensionlessQuantity{T}
    k_eq::Mode{T}
    z_drag::DimensionlessQuantity{T}
    soud_horizon::DimensionlessQuantity{T}
    alpha_c::DimensionlessQuantity{T}
    alpha_b::DimensionlessQuantity{T}
    beta_c::DimensionlessQuantity{T}
    beta_b::DimentionlessQuantity{T}
    k_silk::Mode{T}
   
end

#TransferFunction_EH98(with_baryons::Bool, cosmo::Cosmology{<:Real}, z_eq::Real, k_eq::Mode{<:Real}, z_drag::Real)



end # module TransferFunction
