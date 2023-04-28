module PowerSpectrum

include("../src/MyCosmology.jl")

using Unitful
import Unitful: km, s, Gyr, K, Temperature, DimensionlessQuantity, Density, Volume
using UnitfulAstro: Mpc, Gpc, Msun
import .MyCosmology: FLRWPLANCK18, FLRW, ρ_c

export mass_vs_lagrangian_radius

abstract type Spectrum{T<:Real} end

@derived_dimension Mode dimension(1/km)

struct PowerSpectrumΛCDM{T} <: Spectrum{T}
    
    # Parameter
    amplitude::T
    index::T

    # Function of the power_spectrum
    dimensionless_curvature_ps::Function
end


# Initialise the LCDM power spectrum
function PowerSpectrumΛCDM(amplitude::Real, index::Real) 
    amplitude, index = promote(float(amplitude), float(index))
    return PowerSpectrumΛCDM(amplitude, index, (k::Mode -> amplitude * (k / 0.05 * Mpc)^(index-1)) )
end

const PSPLANCK18 = PowerSpectrumΛCDM(1e-10*exp(3.044), 0.9649)

dimensionless_curvature_power_spectrum(k::Mode, ps::Spectrum=PSPLANCK18) = ps.dimensionless_curvature_ps(k)
curvature_power_spectrum(k::Mode, ps::Spectrum=PSPLANCK18)::Volume = 2.0 * pi^2 * ps.dimensionless_curvature_power_spectrum(k) * k^(-3)



mass_vs_lagrangian_radius(radius::Unitful.Length, volume_factor::Real=6*π^2, cosmo::FLRW = FLRWPLANCK18)::Unitful.Mass = volume_factor *  ρ_c(cosmo) *  radius^3 |> Msun



end # module PowerSpectrum
