module PowerSpectrum

include("../src/MyCosmology.jl")
include("../src/TransferFunction.jl")


using .MyCosmology
import .TransferFunction as tf

export mass_vs_radius, radius_vs_mass

abstract type Spectrum{T<:Real} end

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
    return PowerSpectrumΛCDM(amplitude, index, (k::Real -> amplitude * (k / 0.05)^(index-1)) )
end

const psPlanck18 = PowerSpectrumΛCDM(1e-10*exp(3.044), 0.9649)

""" Curvature power spectrum (dimensionless) at k_Mpc (in 1/Mpc) for the cosmology `cosmo` """
dimensionless_curvature_power_spectrum(k_Mpc::Real, ps::Spectrum=psPlanck18)::Real = ps.dimensionless_curvature_ps(k_Mpc)

""" Curvature power spectrum (in Mpc^3) at k_Mpc (in 1/Mpc) for the cosmology `cosmo` """
curvature_power_spectrum(k_Mpc::Real, ps::Spectrum=psPlanck18)::Real = 2.0 * pi^2 * ps.dimensionless_curvature_power_spectrum(k_Mpc) * k_Mpc^(-3)

#function matter_power_spectrum(k_Mpc::Real, ps::Spectrum = psPlanck18, p::tf.ParametersTF = parametersEH98_planck18_wi_baryons)
#    return curvature_power_spectrum(k_Mpc, ps) * tf.transfer_function(k_Mpc, p)
#end


##############################################
""" 
    mass_vs_lagrangian_radius(R_Mpc, [volume factor, cosmo]) in Msun

# Arguments
- `R_Mpc`: radius in Mpc
- `volume_factor`: volume_factor to relate mass and radius
- `cosmo` : cosmology 
"""
mass_vs_radius(R_Mpc::Real; volume_factor::Real=6*π^2, cosmo::FLRW = planck18)::Real = volume_factor * cosmo.ρ_c0_Msun_Mpc3 * R_Mpc^3


""" 
    radius_vs_mass(M_Msun, [volume factor, cosmo]) in Msun

# Arguments
- `M_Msun`: radius in Msun
- `volume_factor`: volume_factor to relate mass and radius
- `cosmo` : cosmology 
"""
radius_vs_mass(M_Msun::Real; volume_factor::Real=6*π^2, cosmo::FLRW = planck18)::Real = (volume_factor * cosmo.ρ_c0_Msun_Mpc3 * M_Msun)^(1/3)





end # module PowerSpectrum
