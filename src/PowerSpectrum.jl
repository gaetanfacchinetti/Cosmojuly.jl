module PowerSpectrum

include("./MyCosmology.jl")
include("./TransferFunction.jl")


import Main.Cosmojuly.MyCosmology: Cosmology, growth_factor, growth_factor_Carroll, hubble_constant, planck18
import Main.Cosmojuly.TransferFunction: transfer_function, ParametersTF, parametersEH98_planck18
import Unitful: NoUnits, km, s
import UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0


""" ΛCDM power-law power spectrum (dimensionless) at k_Mpc (in 1/Mpc) """
function power_spectrum_ΛCDM(k_Mpc::Real, amplitude::Real = 1e-10*exp(3.044), index::Real = 0.9649)
    return amplitude * (k_Mpc / 0.05)^(index-1)
end

# TO DO
#""" ΛCDM power-law power spectrum (dimensionless) at k_Mpc (in 1/Mpc) with an extra spike inside"""
#function power_spectrum_with_spike(k_Mpc::Real, amplitude_1::Real = 1e-10*exp(3.044), index::Real = 0.9649, amplitude_star::Real = 0.0, k_star::Real = )

""" Curvature power spectrum (in Mpc^3) at k_Mpc (in 1/Mpc) """
function curvature_power_spectrum(k_Mpc::Real, power_spectrum::Function = power_spectrum_ΛCDM, params::Tuple{Vararg{Real}} = (1e-10*exp(3.044), 0.9649))::Real 
    return 2.0 * pi^2 * power_spectrum(k_Mpc, params...) / k_Mpc^3
end 

""" Matter power spectrum (in Mpc^3) at k_Mpc (in 1/Mpc) """
function matter_power_spectrum(
        k_Mpc::Real, z::Real = 0; 
        power_spectrum::Function = power_spectrum_ΛCDM, 
        transfer_function::Function = transfer_function, 
        params_ps::Tuple{Vararg{Real}} = (1e-10*exp(3.044), 0.9649),
        params_TF::ParametersTF = parametersEH98_planck18, 
        approximate_growth_factor::Bool = true,
        cosmo::Cosmology{<:Real} = planck18,
        with_baryons::Bool = true)

    _c_over_H0_Mpc = c_0 / (hubble_constant(cosmo) * km / s / Mpc) / Mpc |> NoUnits 
    _D1_z = approximate_growth_factor ?  growth_factor_Carroll(z, cosmo) : growth_factor(z, cosmo)

    pref = (4. / 25.) * (_D1_z * k_Mpc^2 * transfer_function(k_Mpc, params_TF, with_baryons = with_baryons) * _c_over_H0_Mpc^2 / cosmo.Ω_m0)^2
    return pref * curvature_power_spectrum(k_Mpc, power_spectrum, params_ps) 
end




##############################################
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





end # module PowerSpectrum
