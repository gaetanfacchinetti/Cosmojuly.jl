module TransferFunction

include("../src/MyCosmology.jl")

using Unitful
import Unitful: km, s, Gyr, K, Temperature, DimensionlessQuantity, Density, Volume
using UnitfulAstro: Mpc, Gpc, Msun

import .MyCosmology: FLRWPLANCK18, FLRW, ρ_c, FLRW


abstract type AbstractTranferFunction{T<:Real} end

struct TransferFunction_EH98{T} <: AbstractTranferFunction{T}
    with_baryons::Bool
    cosmology::FLRW{T} 
    z_eq::DimensionlessQuantity{T}
    k_eq::Mode{T}
    z_drag::DimensionlessQuantity{T}

end

end # module TransferFunction
