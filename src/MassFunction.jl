module MassFunction

include("./PowerSpectrum.jl")

using Roots, Random, Interpolations
import QuadGK: quadgk

import Main.Cosmojuly.PowerSpectrum: σ², dσ²_dR, σ, dσ_dR, radius_from_mass, Window, SharpK, TopHat, Gaussian, dradius_dmass, power_spectrum_ΛCDM, mass_from_radius
import Main.Cosmojuly.MyCosmology: Cosmology, growth_factor, growth_factor_Carroll, planck18,  ρ_m_Msun_Mpc3
import Main.Cosmojuly.TransferFunction: transfer_function, TransferFunctionModel, EH98_planck18

export dn_dM
export MassFunctionType, PressSchechter, SethTormen


###########################
# PRESS-SCHECHTER FORMALISM
###########################

abstract type MassFunctionType end
abstract type PressSchechter <: MassFunctionType end
abstract type SethTormen <: MassFunctionType end

f_mass_function(ν::Real, ::Type{PressSchechter}) = sqrt(2.0 / π) * exp(-ν^2 / 2.0)

function f_mass_function(ν::Real, ::Type{SethTormen}) 
    _a = 0.707;
    _νp = sqrt(_a) * ν;
    _q = 0.3;
    _A = 0.3222;

    return _νp / ν * _A * (1.0 + _νp^(-2. * _q)) * sqrt(2.0 * _a / π) * exp(- _νp^2 / 2.)
end

function dn_dM(M_Msun::Real, 
                z::Real = 0,
                ::Type{T} = TopHat,
                ::Type{S} = PressSchechter;
                power_spectrum::Function = power_spectrum_ΛCDM,  
                cosmology::Cosmology{<:Real} = planck18,
                transferFunctionModel::Union{Nothing, TransferFunctionModel} = nothing,
                growth_function::Function = growth_factor_Carroll,
                δ_c = 1.686,
                kws...) where {T<:Window, S<:MassFunctionType}

    _D_z = growth_function(z, cosmology)

    _R  = radius_from_mass(M_Msun, T, cosmology)
    _dR_dM = dradius_dmass(_R, T, cosmology)
    _σ = σ(_R, T, power_spectrum, cosmology, transfer_function_model; kws...) * _D_z
    _dσ_dR = dσ_dR(_R, T,  power_spectrum, cosmology, transfer_function_model; kws...) * _D_z
    _ν = δ_c / _σ

    return  ρ_m_Msun_Mpc3(0, cosmology) / M_Msun * abs(_dσ_dR) / _σ * _dR_dM  * _ν  * f_mass_function(_ν, S)

end


######################
# EXCURSION SET THEORY
######################

export mass_vs_S, mass_fraction_unresolved, mean_number_small_progenitors, draw_mass_with_restrictions
export mean_number_progenitors_below_M, subhalo_mass_function

# In the excursion set framework S=σ² 
# The only valid window function is the SharpK function
# All masses are in solar masses

function mass_vs_S(S::Real;
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transfer_function_model::Union{Nothing, TransferFunctionModel} = nothing,
    kws...)

    return mass_from_radius(10^(find_zero( y -> σ²(10^y, SharpK, power_spectrum, cosmology, transfer_function_model; kws...) - S, (-10, 3), Bisection(), rtol=1e-4)), SharpK, cosmology)
end

function mass_fraction_unresolved(M1::Real, Mres::Real, Δω::Real,
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transfer_function_model::Union{Nothing, TransferFunctionModel} = nothing;
    kws...)

    S1   = σ²(radius_from_mass(M1, SharpK, cosmology), SharpK, power_spectrum, cosmology, transfer_function_model; kws...)
    Sres = σ²(radius_from_mass(Mres, SharpK, cosmology), SharpK, power_spectrum, cosmology, transfer_function_model; kws...)

    return Δω * sqrt(2. / π) / sqrt(Sres - S1)
end


function mean_number_progenitors_below_M(M::Real, M1::Real, Mres::Real, Δω::Real,
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transfer_function_model::Union{Nothing, TransferFunctionModel} = nothing;
    kws...)

    if (M < Mres || M > M1/2.0)
        return 0.0
    end
   
    function _integrand(M2::Real, S1::Real, Δω::Real,
        power_spectrum::Function = power_spectrum_ΛCDM,  
        cosmology::Cosmology{<:Real} = planck18,
        transfer_function_model::Union{Nothing, TransferFunctionModel} = nothing;
        kws...)

            _ΔS = σ²(radius_from_mass(M2, SharpK, cosmology), SharpK, power_spectrum, cosmology, transfer_function_model; kws...) - S1
            _R2 = radius_from_mass(M2, SharpK, cosmology)
            _dS_dM = dσ²_dR(_R2, SharpK, power_spectrum, cosmology, transfer_function_model; kws...) * dradius_dmass(_R2, SharpK, cosmology)
            
            return 1.0 / M2 / _ΔS^(3/2) * abs(_dS_dM)
    end

    _S1 = σ²(radius_from_mass(M1, SharpK, cosmology), SharpK, power_spectrum, cosmology, transfer_function_model; kws...)

    _integral = quadgk(lnM2 -> _integrand(exp(lnM2), _S1, Δω, power_spectrum, cosmology, transfer_function_model; kws...) * exp(lnM2), log(Mres), log(M), rtol=1e-3; kws...)[1]
    return Δω * M1 / sqrt(2.0 * π) * _integral

end


function mean_number_progenitors(M1::Real, Mres::Real, Δω;
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transfer_function_model::Union{Nothing, TransferFunctionModel} = nothing,
    kws...)
    
    if (Mres > M1 / 2.0)
        return 0.0
    end

    return mean_number_progenitors_below_M(M1/2.0, M1, Mres, Δω, power_spectrum, cosmology, transfer_function_model, kws...)
end


function draw_mass_with_restrictions(M1::Real, Mres::Real, Δω::Real;
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transfer_function_model::Union{Nothing, TransferFunctionModel} = nothing,
    kws...)

    function _cumulative(M2::Real, M1::Real, Mres::Real, Δω::Real,
        power_spectrum::Function = power_spectrum_ΛCDM,  
        cosmology::Cosmology{<:Real} = planck18,
        transfer_function_model::Union{Nothing, TransferFunctionModel} = nothing;
        kws...)

        _normalisation = mean_number_progenitors(M1, Mres, Δω; power_spectrum = power_spectrum, cosmology = cosmology, transfer_function_model = transfer_function_model, kws...) 
        return mean_number_progenitors_below_M(M2, M1, Mres, Δω, power_spectrum, cosmology, transfer_function_model, kws...) / _normalisation

    end

    _y = rand(Float64)
    return 10.0^find_zero(z -> _y - _cumulative(exp(z), M1, Mres, Δω, power_spectrum,  cosmology, transfer_function_model; kws...) , (log(Mres), log(M1/2.0)), Bisection(), rtol=1e-2)

end

function draw_mass_with_restrictions(M1::Real, Mres::Real, Δω::Real, interp_mean_number_progenitors_below_M::Function)
    
    function _cumulative(M2::Real, M1::Real, interp_mean_number_progenitors_below_M::Function)
        return 10.0^interp_mean_number_progenitors_below_M(log10(M2), log10(M1)) / 10.0^interp_mean_number_progenitors_below_M(log10(M1/2.), log10(M1)) 
    end

    _y = rand(Float64)
    return 10.0^find_zero(z -> _y - _cumulative(exp(z), M1, Mres, Δω, power_spectrum,  cosmology, transfer_function_model; kws...) , (log(Mres), log(M1/2.0)), Bisection(), rtol=1e-2)

end




function one_step_merger_tree(P::Real, F::Real, M1::Real, Mres::Real, draw_mass::Function)

    _R = rand(Float64)
    array_progenitors = zeros(0)

    if (P < _R || M1 < 2*Mres)
        if (M1 * (1-F) > Mres) 
            append!(array_progenitors, M1 * (1-F)) 
        end 
    else
        M2 = draw_mass()
        if (M1 * (1-F) - M2 > Mres)
            append!(array_progenitors, M1*(1-F) - M2)
        end
        append!(array_progenitors, M2)
    end

    return array_progenitors

end
    

function Δω_approx(Mhost::Real, Mres::Real, 
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transfer_function_model::Union{Nothing, TransferFunctionModel} = nothing;
    kws...) 

    _Rhost = radius_from_mass(Mhost, SharpK, cosmology)
    _dS_dM = dσ²_dR(_Rhost, SharpK, power_spectrum, cosmology, transfer_function_model; kws...) * dradius_dmass(_Rhost, SharpK, cosmology)
    return  sqrt(1e-2 * Mres * abs(_dS_dM) )
end


function subhalo_mass_function(Mhost::Real, Mres::Real;
    power_spectrum::Function = power_spectrum_ΛCDM,  
    cosmology::Cosmology{<:Real} = planck18,
    transfer_function_model::Union{Nothing, TransferFunctionModel} = nothing,
    kws...)

    # initialisation of the values
    the_end = false
    frac_Mhost = 1.0
    Δω = 0.1 *  Δω_approx(Mhost, Mres)

    accretion_redshift = zeros(0)
    subhalo_mass = zeros(0)

    # interpolating the function for the mean number of progenitors below a given mass
    grid = range(log10(Mres),stop=log10(Mhost),length=500)
    y = [log10(mean_number_progenitors_below_M(10.0^logM2, 10.0^logM1, Mres, Δω, power_spectrum,  cosmology, transfer_function_model; kws...)) for logM2 in grid, logM1 in grid]
    replace!(y, Inf=>NaN)
    interp_mean_number_progenitors_below_M = LinearInterpolation((grid, grid), y)

    i = 0

    while(the_end == false)

        the_end = true

        P = 10.0^interp_mean_number_progenitors_below_M(log10(frac_Mhost * Mhost/2.0), log10(frac_Mhost * Mhost))
        F = mass_fraction_unresolved(frac_Mhost * Mhost, Mres, Δω, power_spectrum,  cosmology, transfer_function_model; kws...)
        array_progenitors = one_step_merger_tree(P, F, frac_Mhost * Mhost, Mres, x->draw_mass_with_restrictions(frac_Mhost * Mhost, Mres, Δω, interp_mean_number_progenitors_below_M))

        if (size(array_progenitors)[1] == 1)
            frac_Mhost = array_progenitors[1]/Mhost
            the_end = false
        end

        if (size(array_progenitors)[1]==0)
            frac_Mhost = 0
        end

        if (size(array_progenitors)[1]==2)
            if (array_progenitors[1] > array_progenitors[2])
                frac_Mhost = array_progenitors[1] / Mhost
                append!(subhalo_mass, array_progenitors[2] / Mhost)
            else
                frac_Mhost = array_progenitors[2] / Mhost
                append!(subhalo_mass, array_progenitors[1] / Mhost)
            end

            the_end = false
        end

        if i%10000 == 0
            println(subhalo_mass)
        end

        i = i+1

    end

    return subhalo_mass
end



end # end of module MassFunction