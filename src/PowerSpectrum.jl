module PowerSpectrum

include("../src/MyCosmology.jl")

using Unitful
import Unitful: km, s, Gyr
using UnitfulAstro: Mpc, Gpc, Msun
import .MyCosmology: BASEPLANCK18, AbstractBaseCosmology, ρ_c

export mass_vs_lagrangian_radius

abstract type AbstractPowerSpectrum end

@derived_dimension Mode dimension(1/km)

struct PowerSpectrumLCDM{T<:Real} <: AbstractPowerSpectrum
    amplitude::T
    index::T
    value::Function
end

# Initialise the LCDM power spectrum
PowerSpectrumLCDM(amplitude::Real, index::Real) = PowerSpectrumLCDM(promote(float(amplitude), float(index))..., k::Mode->power_law(amplitude, index, k))
power_law(amplitude::Real, index::Real, k::Mode) = amplitude .* (k / 0.05 * Mpc)^(index-1) ## To be checked


mass_vs_lagrangian_radius(radius::Unitful.Length, volume_factor::Real=6*π^2, cosmo::AbstractBaseCosmology = BASEPLANCK18)::Unitful.Mass = volume_factor *  ρ_c(cosmo) *  radius^3 |> Msun



end # module PowerSpectrum
