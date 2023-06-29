module PowerSpectrum

include("./MyCosmology.jl")
include("./TransferFunction.jl")


import Main.Cosmojuly.MyCosmology: Cosmology, growth_factor, growth_factor_Carroll, hubble_constant, planck18
import Main.Cosmojuly.TransferFunction: transfer_function, ParametersTF, parametersEH98_planck18
import Unitful: NoUnits, km, s
import UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0


using QuadGK

export curvature_power_spectrum, matter_power_spectrum
export window_function, Window, Real_space_top_hat, Sharp_k, Exponential

""" ΛCDM power-law power spectrum (dimensionless) at k_Mpc (in 1/Mpc) """
function power_spectrum_ΛCDM(k_Mpc::Real, amplitude::Real = 1e-10*exp(3.044), index::Real = 0.9649)
    return amplitude * (k_Mpc / 0.05)^(index-1)
end

# TO DO
#""" ΛCDM power-law power spectrum (dimensionless) at k_Mpc (in 1/Mpc) with an extra spike inside"""
#function power_spectrum_with_spike(k_Mpc::Real, amplitude_1::Real = 1e-10*exp(3.044), index::Real = 0.9649, amplitude_star::Real = 0.0, k_star::Real = )

""" Curvature power spectrum (in Mpc^3) at k_Mpc (in 1/Mpc) """
function curvature_power_spectrum(k_Mpc::Real, power_spectrum::Function = power_spectrum_ΛCDM, params...)::Real 
    return 2.0 * pi^2 * power_spectrum(k_Mpc, params...) / k_Mpc^3
end 

""" Matter power spectrum (in Mpc^3) at k_Mpc (in 1/Mpc) """
function matter_power_spectrum(k_Mpc::Real, z::Real = 0, 
    power_spectrum::Function    = power_spectrum_ΛCDM, 
    transfer_function::Function = transfer_function, 
    cosmology::Cosmology{<:Real} = planck18,
    params_ps... = (1e-10*exp(3.044), 0.9649)...;
    approximate_growth_factor::Bool = true,
    with_baryons::Bool = true)

    _c_over_H0_Mpc = c_0 / (hubble_constant(cosmology) * km / s / Mpc) / Mpc |> NoUnits 
    _D1_z = approximate_growth_factor ?  growth_factor_Carroll(z, cosmology) : growth_factor(z, cosmology)

    pref = (4. / 25.) * (_D1_z * k_Mpc^2 * transfer_function(k_Mpc, cosmology, with_baryons = with_baryons) * _c_over_H0_Mpc^2 / cosmology.Ω_m0)^2
    return pref * curvature_power_spectrum(k_Mpc, power_spectrum, params_ps...) 
end


""" Matter power spectrum (in Mpc^3) at k_Mpc (in 1/Mpc) optimized for planck18 ΛCDM """
function matter_power_spectrum(k_Mpc::Real, z::Real = 0; approximate_growth_factor::Bool = true, with_baryons::Bool = true)

    _c_over_H0_Mpc = c_0 / (hubble_constant(planck18) * km / s / Mpc) / Mpc |> NoUnits 
    _D1_z = approximate_growth_factor ?  growth_factor_Carroll(z, planck18) : growth_factor(z, planck18)

    pref = (4. / 25.) * (_D1_z * k_Mpc^2 * transfer_function(k_Mpc, parametersEH98_planck18, with_baryons = with_baryons) * _c_over_H0_Mpc^2 / planck18.Ω_m0)^2
    return pref * curvature_power_spectrum(k_Mpc, power_spectrum_ΛCDM, 1e-10*exp(3.044), 0.9649) 
end





##############################################
# Quantities that depend on the window function

abstract type Window end
abstract type TopHat <: Window end
abstract type SharpK <: Window end
abstract type Exponential <: Window end

window_function(kR::Real, ::Type{TopHat})::Real     = kR > 1e-3 ? 3.0 * (sin(kR) - kR * cos(kR)) / kR^3 : 1.0 - kR^2/10.0 
window_function(kR::Real, ::Type{SharpK})::Real     = 1 - kR > 0 ? 1.0 : 0.0
window_function(kR::Real, ::Type{Exponential})::Real = exp(-kR * kR / 2.0)

volume_factor(::Type{TopHat})::Real = 4.0/3.0 * π 
volume_factor(::Type{SharpK})::Real = 6.0 * π^2 
volume_factor(::Type{Exponential})::Real = (2*π)^(3//2)

""" 
    mass_vs_lagrangian_radius(R_Mpc, [volume factor, cosmo]) in Msun

# Arguments
- `R_Mpc`: radius in Mpc
- `volume_factor`: volume_factor to relate mass and radius
- `cosmo` : cosmology 
"""
mass_vs_radius(R_Mpc::Real; volume_factor::Real=6*π^2, cosmo::MyCosmology.Cosmology = planck18)::Real = volume_factor * cosmo.ρ_c0_Msun_Mpc3 * R_Mpc^3


""" 
    radius_vs_mass(M_Msun, [volume factor, cosmo]) in Msun

# Arguments
- `M_Msun`: radius in Msun
- `volume_factor`: volume_factor to relate mass and radius
- `cosmo` : cosmology 
"""
radius_vs_mass(M_Msun::Real; volume_factor::Real=6*π^2, cosmo::MyCosmology.Cosmology = planck18)::Real = (volume_factor * cosmo.ρ_c0_Msun_Mpc3 * M_Msun)^(1/3)


radius_vs_mass(M_Msun::Real, ::Type{T} = TopHat; cosmo::MyCosmology.Cosmology = planck18)::Real where {T<:Window} = (volume_factor(T) * cosmo.ρ_c0_Msun_Mpc3 * M_Msun)^(1/3)
mass_vs_radius(R_Mpc::Real, ::Type{T} = TopHat; cosmo::MyCosmology.Cosmology = planck18)::Real where {T<:Window}  = volume_factor(T) * cosmo.ρ_c0_Msun_Mpc3 * R_Mpc^3

#function sigma2(R_Mpc::Real, ::Type{TopHat}; matter_power_spectrum::Function = matter_power_spectrum) 
#    return QuadGK.quadgk(matter_power_spectrum)
#end


end # module PowerSpectrum
