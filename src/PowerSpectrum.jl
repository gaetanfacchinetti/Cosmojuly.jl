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
export window_function, Window, TopHat, SharpK, Gaussian
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
abstract type Gaussian <: Window end

@doc raw""" 
    window_function(kR, [T])

Give the window function for the product of mode by radius ``k \times R``: `kR::Real` and a certain window type `T<:Window` 
"""
window_function(kR::Real, ::Type{T} = TopHat) where {T<:Window} = window_function(kR, T)

window_function(kR::Real, ::Type{TopHat})      = kR > 1e-3 ? 3.0 * (sin(kR) - kR * cos(kR)) / kR^3 : 1.0 - kR^2/10.0 
window_function(kR::Real, ::Type{SharpK})      = 1 - kR > 0 ? 1.0 : 0.0
window_function(kR::Real, ::Type{Gaussian}) = exp(-kR * kR / 2.0)

dwindow_function_dkR(kR::Real, ::Type{TopHat})   = kR < 0.1 ? -kR/5.0 + kR^3/70.0 : 3*sin(kR) / kR^2 - 9 * (-kR * cos(kR) + sin(kR)) / kR^4
dwindow_function_dkR(kR::Real,  ::Type{SharpK})   = 0
dwindow_function_dkR(kR::Real, ::Type{Gaussian}) = -kR * exp(-kR^2/2.0)


volume_factor(::Type{TopHat})      = 4.0/3.0 * π 
volume_factor(::Type{SharpK})      = 6.0 * π^2 
volume_factor(::Type{Gaussian}) = (2*π)^(3//2)

@doc raw""" 
   volume_factor([T])

Give the volume factor associated to a certain window type `T<:Window` (default is `TopHat`)
"""
volume_factor(::Type{T} = TopHat) where {T<:Window} = volume_factor(T)


""" 
    mass_from_radius(R_Mpc, [T, [cosmo]]) in Msun

Give the Lagrangian mass (in Msun) in terms of the comoving radius R (in Mpc)

# Arguments
- `R_Mpc`: radius in Mpc
- `T`: type of Window (default is `TopHat`)
- `cosmo` : cosmology type (default is `planck18`)
"""
mass_from_radius(R_Mpc::Real, ::Type{T} = TopHat; cosmo::Cosmology = planck18) where {T<:Window}  = volume_factor(T) * ρ_m_Msun_Mpc3(0, cosmo) * R_Mpc^3


""" 
    radius_from_mass(M_Msun, [T, [cosmo]]) in Msun

Give the comoving radius R (in Mpc) in terms of the  Lagrangian mass (in Msun)

# Arguments
- `M_Msun`: radius in Msun
- `T`: type of Window (default is `TopHat`)
- `cosmo` : cosmology type (default is `planck18`)
"""
radius_from_mass(M_Msun::Real, ::Type{T} = TopHat; cosmo::Cosmology = planck18) where {T<:Window} = (M_Msun / volume_factor(T) / ρ_m_Msun_Mpc3(0, cosmo))^(1/3)

σ²(R_Mpc::Real, ::Type{TopHat}, matter_ps::Function; kws...)   = log(20.0) - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)) * window_function(exp(lnk) * R_Mpc, TopHat)^2, -8.0, log(20.0) - log(R_Mpc), rtol=1e-3; kws...)[1] : 0.0
σ²(R_Mpc::Real, ::Type{SharpK}, matter_ps::Function; kws...)   = - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)), -8.0, -log(R_Mpc), rtol=1e-3; kws...)[1] : 0.0
σ²(R_Mpc::Real, ::Type{Gaussian}, matter_ps::Function; kws...) = log(4.0) - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)) * window_function(exp(lnk) * R_Mpc, Gaussian)^2, -8.0, log(4.0) - log(R_Mpc), rtol=1e-3; kws...)[1] : 0.0
σ²(R_Mpc::Real, ::Type{T}, matter_ps::Function; kws...) where {T<:Window} = σ²(R_Mpc, T, matter_ps; kws...)

function σ²(
    R_Mpc::Real, 
    ::Type{T} = TopHat, 
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transferFunctionModel::Union{Nothing, TransferFunctionModel} = nothing;
    kws...) where {T<:Window}

    return σ²(R_Mpc, T, k->matter_power_spectrum(k, 0; power_spectrum = power_spectrum, cosmology = cosmology, transferFunctionModel = transferFunctionModel, dimensionless=true); kws...)
end


σ(R_Mpc::Real, ::Type{T}, matter_ps::Function; kws...) where {T<:Window} = sqrt(σ²(R_Mpc, T, matter_ps; kws...))

function σ(
    R_Mpc::Real, 
    ::Type{T} = TopHat, 
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transferFunctionModel::Union{Nothing, TransferFunctionModel} = nothing;
    kws...) where {T<:Window}

    return σ(R_Mpc, T, power_spectrum, cosmology, TransferFunctionModel; kws...)
end


dσ²_dR(R_Mpc::Real, ::Type{TopHat}, matter_ps::Function; kws...) = log(20.0) - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)) * 2 * window_function(exp(lnk) * R_Mpc, TopHat) * dwindow_function_dkR(exp(lnk) * R_Mpc, TopHat), -8.0, log(20.0) - log(R_Mpc), rtol=1e-3; kws...)[1] : 0.0
dσ²_dR(R_Mpc::Real, ::Type{SharpK}, matter_ps::Function; kws...) = - matter_power_spectrum(1.0/R) / R
dσ²_dR(R_Mpc::Real, ::Type{Gaussian}, matter_ps::Function; kws...) = log(4.0) - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)) * 2 * window_function(exp(lnk) * R_Mpc, Gaussian) * dwindow_function_dkR(exp(lnk) * R_Mpc, Gaussian), -8.0, log(4.0) - log(R_Mpc), rtol=1e-3; kws...)[1] : 0.0
dσ²_dR(R_Mpc::Real, ::Type{T}, matter_ps::Function; kws...) where {T<:Window} = dσ²_dR(R_Mpc, T, matter_ps; kws...)

function dσ²_dR(
    R_Mpc::Real, 
    ::Type{T} = TopHat, 
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transferFunctionModel::Union{Nothing, TransferFunctionModel} = nothing;
    kws...) where {T<:Window}

    return dσ²_dR(R_Mpc, T, power_spectrum, cosmology, TransferFunctionModel; kws...)
end

dσ_dR(R_Mpc::Real, ::Type{T}, matter_ps::Function; kws...) where {T<:Window} = 0.5 * dσ²_dR(R_Mpc, T, matter_ps; kws...) / σ(R_Mpc, T, matter_ps; kws...) 

function dσ_dR(R_Mpc::Real, 
    ::Type{T} = TopHat, 
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transferFunctionModel::Union{Nothing, TransferFunctionModel} = nothing;
    kws...) where {T<:Window}

    return dσ_dR(R_Mpc, T, power_spectrum, cosmology, TransferFunctionModel; kws...)
end



end # module PowerSpectrum
