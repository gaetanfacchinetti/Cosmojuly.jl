module PowerSpectrum

include("./BackgroundCosmo.jl")
include("./TransferFunction.jl")


import Main.Cosmojuly.BackgroundCosmo: BkgCosmology, growth_factor, growth_factor_Carroll, hubble_constant, planck18_bkg, ρ_m_Msun_Mpc3
import Main.Cosmojuly.TransferFunction: transfer_function, TransferFunctionModel, EH98_planck18, EH98
import Unitful: NoUnits, km, s
import UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0

import QuadGK: quadgk

export curvature_power_spectrum, matter_power_spectrum, power_spectrum_ΛCDM
export Cosmology, planck18
export window_function, Window, TopHat, SharpK, Gaussian
export radius_from_mass, mass_from_radius, dradius_dmass, σ², dσ²_dR, σ, dσ_dR
export σ²_vs_M, dσ²_dM, σ_vs_M, dσ_dM

#####################
# COSMOLOGY STRUCTURE
#####################

struct Cosmology
    bkg::BkgCosmology{<:Real}
    power_spectrum::Function
    transfer_function_model::TransferFunctionModel
end

function Cosmology(bkg::BkgCosmology{<:Real}, power_spectrum::Function, ::Type{T} = EH98) where {T<:TransferFunctionModel}
    return Cosmology(bkg, power_spectrum, T(bkg))
end

const planck18::Cosmology = Cosmology(planck18_bkg, k_Mpc->power_spectrum_ΛCDM(k_Mpc, 1e-10*exp(3.044), 0.9649), EH98_planck18)



######################################
# (CURVATURE AND MATTER) POWER SPECTRA
######################################

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
    cosmology::Cosmology = planck18,
    growth_function::Function = growth_factor_Carroll,
    dimensionless = false,
    with_baryons::Bool = true)

    _c_over_H0_Mpc = c_0 / (hubble_constant(cosmology.bkg) * km / s / Mpc) / Mpc |> NoUnits 
    _D1_z = growth_function(z, cosmology.bkg)
    _tf = transfer_function(k_Mpc, cosmology.transfer_function_model, with_baryons = with_baryons)
    _prefactor = !dimensionless ? 1 : k_Mpc^3/(2*π^2)

    return _prefactor * (4. / 25.) * (_D1_z * k_Mpc^2 * _tf * _c_over_H0_Mpc^2 / cosmology.bkg.Ω_m0)^2 * curvature_power_spectrum(k_Mpc, cosmology.power_spectrum) 
end


###############################################
# QUANTITIES THAT DEPEND ON THE WINDOW FUNCTION
###############################################

abstract type Window end
abstract type TopHat <: Window end
abstract type SharpK <: Window end
abstract type Gaussian <: Window end

# -------------------------------
# WINDOW FUNCTION AND DERIVATIVES

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

# --------------
# VOLUME FACTORS

volume_factor(::Type{TopHat})      = 4.0/3.0 * π 
volume_factor(::Type{SharpK})      = 6.0 * π^2 
volume_factor(::Type{Gaussian}) = (2*π)^(3//2)

@doc raw""" 
   volume_factor([T])

Give the volume factor associated to a certain window type `T<:Window` (default is `TopHat`)
"""
volume_factor(::Type{T} = TopHat) where {T<:Window} = volume_factor(T)

# ------------------------------------
# MASS - LAGRANGIAN RADIUS CONVERSIONS

""" 
    mass_from_radius(R_Mpc, [T, [cosmo]]) in Msun

Give the Lagrangian mass (in Msun) in terms of the comoving radius R (in Mpc)

# Arguments
- `R_Mpc`: radius in Mpc
- `T`: type of Window (default is `TopHat`)
- `cosmo` : cosmology type (default is `planck18_bkg`)
"""
mass_from_radius(R_Mpc::Real, ::Type{T} = TopHat, cosmo::BkgCosmology = planck18_bkg) where {T<:Window}  = volume_factor(T) * ρ_m_Msun_Mpc3(0, cosmo) * R_Mpc^3


@doc raw""" 
    radius_from_mass(M_Msun, [T, [cosmo]]) in Msun

Give the comoving radius R (in Mpc) in terms of the  Lagrangian mass (in Msun)

# Arguments
- `M_Msun`: mass in Msun
- `T`: type of Window (default is `TopHat`)
- `cosmo` : cosmology type (default is `planck18_bkg`)
"""
radius_from_mass(M_Msun::Real, ::Type{T} = TopHat, cosmo::BkgCosmology = planck18_bkg) where {T<:Window} = (M_Msun / volume_factor(T) / ρ_m_Msun_Mpc3(0, cosmo))^(1/3)


dradius_dmass(R_Mpc::Real, ::Type{T} = TopHat, cosmo::BkgCosmology = planck18_bkg) where {T<:Window} =  R_Mpc^(-2) / 3.0 / volume_factor(T) / ρ_m_Msun_Mpc3(0, cosmo)

# ---------------------------------------------------------
# SMOOTHED VARIANCE OF THE POWER SPECTRUM (AND DERIVATIVES)


σ²(R_Mpc::Real, ::Type{T} = TopHat; cosmology::Cosmology = planck18) where {T<:Window} = σ²(R_Mpc, T, k->matter_power_spectrum(k, 0, cosmology = cosmology, dimensionless = true))

@doc raw""" 
   σ^2(R_Mpc, T, matter_ps; kws...)

Give the variance of the function `matter_ps` smoothed on region of size `R_Mpc` in Mpc.

``\sigma^2(R) = \int_0^{\infty} \mathcal{P}(k) |W(kR)|^2 {\rm d} \ln k``

# Arguments
- `R_Mpc`: radius in Mpc
- `T`: type of Window 
- `matter_ps`: function
- `kws`: arguments passed to the integration routine `quadgk`
"""
σ²(R_Mpc::Real, ::Type{T}, matter_ps::Function) where {T<:Window} = σ²(R_Mpc, T, matter_ps)
σ²(R_Mpc::Real, ::Type{TopHat}, matter_ps::Function)   = log(20.0) - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)) * window_function(exp(lnk) * R_Mpc, TopHat)^2, -8.0, log(20.0) - log(R_Mpc), rtol=1e-4)[1] : 0.0
σ²(R_Mpc::Real, ::Type{SharpK}, matter_ps::Function)   = - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)), -8.0, -log(R_Mpc), rtol=1e-3)[1] : 0.0
σ²(R_Mpc::Real, ::Type{Gaussian}, matter_ps::Function) = log(4.0) - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)) * window_function(exp(lnk) * R_Mpc, Gaussian)^2, -8.0, log(4.0) - log(R_Mpc), rtol=1e-4)[1] : 0.0

σ(R_Mpc::Real, ::Type{T} = TopHat; cosmology::Cosmology = planck18) where {T<:Window} = σ(R_Mpc, T, k->matter_power_spectrum(k, 0, cosmology = cosmology, dimensionless = true))
σ(R_Mpc::Real, ::Type{T}, matter_ps::Function) where {T<:Window} = sqrt(σ²(R_Mpc, T, matter_ps))

dσ²_dR(R_Mpc::Real, ::Type{TopHat}, matter_ps::Function) = log(20.0) - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)) * 2 * window_function(exp(lnk) * R_Mpc, TopHat) * dwindow_function_dkR(exp(lnk) * R_Mpc, TopHat) * exp(lnk), -8.0, log(20.0) - log(R_Mpc), rtol=1e-4)[1] : 0.0
dσ²_dR(R_Mpc::Real, ::Type{SharpK}, matter_ps::Function) = - matter_ps(1.0 / R_Mpc) / R_Mpc
dσ²_dR(R_Mpc::Real, ::Type{Gaussian}, matter_ps::Function) = log(4.0) - log(R_Mpc) > -8.0 ? quadgk(lnk -> matter_ps(exp(lnk)) * 2 * window_function(exp(lnk) * R_Mpc, Gaussian) * dwindow_function_dkR(exp(lnk) * R_Mpc, Gaussian) * exp(lnk), -8.0, log(4.0) - log(R_Mpc), rtol=1e-4)[1] : 0.0
dσ²_dR(R_Mpc::Real, ::Type{T} = TopHat; cosmology::Cosmology = planck18) where {T<:Window} = dσ²_dR(R_Mpc, T, k->matter_power_spectrum(k, 0, cosmology = cosmology, dimensionless = true))
dσ²_dR(R_Mpc::Real, ::Type{T}, matter_ps::Function) where {T<:Window} = dσ²_dR(R_Mpc, T, matter_ps)

dσ_dR(R_Mpc::Real, ::Type{T} = TopHat; cosmology::Cosmology = planck18) where {T<:Window} = dσ_dR(R_Mpc, T, k->matter_power_spectrum(k, 0, cosmology = cosmology, dimensionless = true))
dσ_dR(R_Mpc::Real, ::Type{T}, matter_ps::Function) where {T<:Window} = 0.5 * dσ²_dR(R_Mpc, T, matter_ps) / σ(R_Mpc, T, matter_ps) 

σ²_vs_M(M_Msun::Real, ::Type{T} = TopHat; cosmology::Cosmology = planck18) where {T<:Window} =  σ²(radius_from_mass(M_Msun, T, cosmology.bkg), T, cosmology = cosmology)
σ_vs_M(M_Msun::Real, ::Type{T} = TopHat; cosmology::Cosmology = planck18) where {T<:Window} =  σ(radius_from_mass(M_Msun, T, cosmology.bkg), T, cosmology = cosmology)

function dσ²_dM(M_Msun::Real, ::Type{T} = TopHat; cosmology::Cosmology = planck18) where {T<:Window} 
    _R_Mpc = radius_from_mass(M_Msun, T, cosmology.bkg)
    return dσ²_dR(_R_Mpc, T, cosmology = cosmology) * dradius_dmass(_R_Mpc, T, cosmology.bkg)
end

function dσ_dM(M_Msun::Real, ::Type{T} = TopHat; cosmology::Cosmology = planck18) where {T<:Window} 
    _R_Mpc = radius_from_mass(M_Msun, T, cosmology.bkg)
    return dσ_dR(_R_Mpc, T, cosmology = cosmology) * dradius_dmass(_R_Mpc, T, cosmology.bkg)
end


end # module PowerSpectrum
