module PowerSpectrum

include("./MyCosmology.jl")
include("./TransferFunction.jl")


import Main.Cosmojuly.MyCosmology: Cosmology, growth_factor, growth_factor_Carroll, hubble_constant, planck18, ρ_m_Msun_Mpc3
import Main.Cosmojuly.TransferFunction: transfer_function, TransferFunctionModel, EH98_planck18
import Unitful: NoUnits, km, s
import UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0

import QuadGK: quadgk

export curvature_power_spectrum, matter_power_spectrum
export window_function, Window, TopHat, SharpK, Exponential
export radius_from_mass, mass_from_radius, σ²_m, σ²_χ

""" ΛCDM power-law power spectrum (dimensionless) at k_Mpc (in 1/Mpc) """
function power_spectrum_ΛCDM(k_Mpc::Real, amplitude::Real = 1e-10*exp(3.044), index::Real = 0.9649)
    return amplitude * (k_Mpc / 0.05)^(index-1)
end

""" Curvature power spectrum (in Mpc^3) at k_Mpc (in 1/Mpc) """
function curvature_power_spectrum(k_Mpc::Real, power_spectrum::Function = power_spectrum_ΛCDM)
    return 2.0 * pi^2 * power_spectrum(k_Mpc) / k_Mpc^3
end 

""" Matter power spectrum (in Mpc^3) at k_Mpc (in 1/Mpc) """
function matter_power_spectrum( 
    k_Mpc::Real, 
    z::Real = 0.0;
    power_spectrum::Function = power_spectrum_ΛCDM, 
    cosmology::Cosmology{<:Real} = planck18,
    transferFunctionModel::Union{Nothing, <:TransferFunctionModel} = nothing,
    dimensionless = false,
    approximate_growth_factor::Bool = true,
    with_baryons::Bool = true)

    _c_over_H0_Mpc = c_0 / (hubble_constant(cosmology) * km / s / Mpc) / Mpc |> NoUnits 
    _D1_z = approximate_growth_factor ?  growth_factor_Carroll(z, cosmology) : growth_factor(z, cosmology)

    ## Accelerate computation with planck18 cosmology
    ## This can be done for other pre-implemented cosmologies
    if cosmology === planck18
        transferFunctionModel = isnothing(transferFunctionModel) ? EH98_planck18 : transferFunctionModel
    end

    tf = isnothing(transferFunctionModel) ? transfer_function(k_Mpc, cosmology, with_baryons = with_baryons) : transfer_function(k_Mpc, transferFunctionModel, with_baryons = with_baryons)

    prefactor = !dimensionless ? 1 : k_Mpc^3/(2*π^2)

    return prefactor * (4. / 25.) * (_D1_z * k_Mpc^2 * tf * _c_over_H0_Mpc^2 / cosmology.Ω_m0)^2 * curvature_power_spectrum(k_Mpc, power_spectrum) 
end




##############################################
# Quantities that depend on the window function

abstract type Window end
abstract type TopHat <: Window end
abstract type SharpK <: Window end
abstract type Exponential <: Window end

window_function(kR::Real, ::Type{TopHat})      = kR > 1e-3 ? 3.0 * (sin(kR) - kR * cos(kR)) / kR^3 : 1.0 - kR^2/10.0 
window_function(kR::Real, ::Type{SharpK})      = 1 - kR > 0 ? 1.0 : 0.0
window_function(kR::Real, ::Type{Exponential}) = exp(-kR * kR / 2.0)

volume_factor(::Type{TopHat})      = 4.0/3.0 * π 
volume_factor(::Type{SharpK})      = 6.0 * π^2 
volume_factor(::Type{Exponential}) = (2*π)^(3//2)


""" 
    mass_from_lagrangian_radius(R_Mpc, [volume factor, cosmo]) in Msun

# Arguments
- `R_Mpc`: radius in Mpc
- `volume_factor`: volume_factor to relate mass and radius
- `cosmo` : cosmology 
"""
mass_from_radius(R_Mpc::Real, ::Type{T} = TopHat; cosmo::Cosmology = planck18) where {T<:Window}  = volume_factor(T) * ρ_m_Msun_Mpc3(0, cosmo) * R_Mpc^3


""" 
    radius_from_mass(M_Msun, [volume factor, cosmo]) in Msun

# Arguments
- `M_Msun`: radius in Msun
- `volume_factor`: volume_factor to relate mass and radius
- `cosmo` : cosmology 
"""
radius_from_mass(M_Msun::Real, ::Type{T} = TopHat; cosmo::Cosmology = planck18) where {T<:Window} = (M_Msun / volume_factor(T) / ρ_m_Msun_Mpc3(0, cosmo))^(1/3)


function σ²(R_Mpc::Real, ::Type{TopHat}, matter_ps::Function; kws...)
    return log(20.0) - log(R_Mpc) > -10.0 ? quadgk(lnk -> matter_ps(exp(lnk)) * window_function(exp(lnk) * R_Mpc, TopHat)^2, -10.0, log(20.0) - log(R_Mpc), rtol=1e-3; kws...)[1] : 0.0
end

function σ²(R_Mpc::Real, ::Type{SharpK}, matter_ps::Function; kws...)
    return - log(R_Mpc) > -10.0 ? quadgk(lnk -> matter_ps(exp(lnk)), -10.0, -log(R_Mpc), rtol=1e-3; kws...)[1] : 0.0
end


function σ²_m(
    R_Mpc::Real, 
    ::Type{T} = TopHat, 
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transferFunctionModel::Union{Nothing, TransferFunctionModel} = nothing,
    kws...) where {T<:Window}

    return σ²(R_Mpc, T, k->matter_power_spectrum(k, 0; power_spectrum = power_spectrum, cosmology = cosmology, transferFunctionModel = transferFunctionModel, dimensionless=true); kws...)
end

function σ²_χ(
    R_Mpc::Real, 
    ::Type{T} = TopHat, 
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transferFunctionModel::Union{Nothing, TransferFunctionModel} = nothing,
    kws...) where {T<:Window}

    return σ²(R_Mpc, T, k->matter_power_spectrum(k, 0; power_spectrum = power_spectrum, cosmology = cosmology, transferFunctionModel = transferFunctionModel, dimensionless=true, with_baryons=false); kws...)
end







end # module PowerSpectrum
