module MassFunction

include("./PowerSpectrum.jl")

using Roots, Random, Interpolations
import QuadGK: quadgk

import Main.Cosmojuly.PowerSpectrum: Cosmology, planck18, σ², dσ²_dR, σ, dσ_dR, radius_from_mass, σ²_vs_M, dσ²_dM, σ_vs_M, dσ_dM
import Main.Cosmojuly.PowerSpectrum: Window, SharpK, TopHat, Gaussian, dradius_dmass, power_spectrum_ΛCDM, mass_from_radius
import Main.Cosmojuly.BackgroundCosmo: BkgCosmology, growth_factor, growth_factor_Carroll, planck18_bkg,  ρ_m_Msun_Mpc3
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
                cosmology::Cosmology = planck18,
                growth_function::Function = growth_factor_Carroll,
                δ_c = 1.686) where {T<:Window, S<:MassFunctionType}

    _D_z = growth_function(z, cosmology.bkg)
    _σ = σ_vs_M(M_Msun, T, cosmology = cosmology) * _D_z
    _dσ_dM = dσ_dM(M_Msun, T,  cosmology = cosmology) * _D_z
    _ν = δ_c / _σ

    return  ρ_m_Msun_Mpc3(0, cosmology.bkg) / M_Msun * abs(_dσ_dM) / _σ  * _ν  * f_mass_function(_ν, S)

end


######################
# EXCURSION SET THEORY
######################

export mass_vs_S, mass_fraction_unresolved, mean_number_small_progenitors, draw_mass_with_restrictions
export mean_number_progenitors_below_M, subhalo_mass_function, cumulative_progenitors, Δω_approx, redshift_array

# In the excursion set framework S=σ² 
# The only valid window function is the SharpK function
# All masses are in solar masses

mass_vs_S(S::Real; cosmology::Cosmology = planck18) = 10^(find_zero(y -> σ²_vs_M(10^y, SharpK, cosmology = cosmology) - S, (-20, 20), Bisection(), rtol=1e-4))
mass_fraction_unresolved(M1::Real, Mres::Real; cosmology::Cosmology = planck18) =  sqrt(2. / π) / sqrt(σ²_vs_M(Mres, SharpK, cosmology=cosmology) - σ²_vs_M(M1, SharpK, cosmology=cosmology))

# Mean number of progenitors between Mres and M in a host of mass M1 per time step Δω
function mean_number_progenitors(M::Real, M1::Real, Mres::Real; cosmology::Cosmology = planck18) 

    if (M < Mres || M > M1 / 2.0)
        return 0.0
    end
   
    function _integrand(M2::Real, S1::Real, cosmology::Cosmology = planck18)

            _ΔS = σ²_vs_M(M2, SharpK, cosmology = cosmology) - S1
            _dS2_dM = dσ²_dM(M2, SharpK, cosmology = cosmology)
            
            return 1.0 / M2 / _ΔS^(3/2) * abs(_dS2_dM)
    end

    _S1 = σ²_vs_M(M1, SharpK, cosmology = cosmology)

    return  M1 / sqrt(2.0 * π)  * quadgk(lnM2 -> _integrand(exp(lnM2), _S1, cosmology) * exp(lnM2), log(Mres), log(M), rtol=1e-3)[1]

end

cumulative_progenitors(M2::Real, M1::Real, Mres::Real; cosmology::Cosmology = planck18) =  M2 > M1/2.0 ? 1.0 : mean_number_progenitors(M2, M1, Mres, cosmology = cosmology) / mean_number_progenitors(M1/2.0, M1, Mres, cosmology = cosmology) 

function draw_mass_with_restrictions(M1::Real, Mres::Real; cosmology::Cosmology = planck18) 
    _y = rand(Float64)
    return 10.0^find_zero(z -> _y - cumulative_progenitors(10.0^z, M1, Mres, cosmology=cosmology), (log10(Mres), log10(M1/2.0)), Bisection(), rtol=1e-2)
end

function one_step_merger_tree(P::Real, F::Real, M1::Real, Mres::Real, cosmology::Cosmology = planck18)

    _R = rand(Float64)
    array_progenitors = zeros(0)

    if (P < _R || M1 < 2*Mres)
        if (M1 * (1-F) > Mres) 
            append!(array_progenitors, M1 * (1-F)) 
        end 
    else
        M2 = draw_mass_with_restrictions(M1, Mres, cosmology = cosmology)
        if (M1 * (1-F) - M2 > Mres)
            append!(array_progenitors, M1 * (1-F) - M2)
        end
        append!(array_progenitors, M2)
    end

    return array_progenitors

end
    
Δω_approx(Mhost::Real, Mres::Real; cosmology::Cosmology = planck18) = sqrt(1e-2 * Mres * abs(dσ²_dM(Mhost, SharpK, cosmology = cosmology)))

function subhalo_mass_function(Mhost_init::Real, Mres::Real; 
                                cosmology::Cosmology=planck18, 
                                growth_function::Function = growth_factor_Carroll, 
                                δ_c::Real = 1.686)

    # initialisation of the values
    the_end = false
    frac_init = 1.0
    Δω = 0.1 * Δω_approx(Mhost_init, Mres)
    z_array = redshift_array(0, Δω, growth_function = growth_function, δ_c = δ_c)
    

    z_accretion = zeros(0)
    subhalo_mass = zeros(0)

    i = 1

    while(the_end == false)

        the_end = true

        Mhost = Mhost_init * frac_init

        P = Δω * mean_number_progenitors(Mhost / 2.0, Mhost, Mres, cosmology = cosmology)
        F = Δω * mass_fraction_unresolved(Mhost, Mres, cosmology = cosmology)
        
        array_progenitors = one_step_merger_tree(P, F, Mhost, Mres, cosmology)

        if (size(array_progenitors)[1] == 1)
            frac_init = array_progenitors[1]/Mhost_init
            the_end = false
        end

        if (size(array_progenitors)[1]==0)
            frac_init = 0.0
        end

        if (size(array_progenitors)[1]==2)
            append!(z_accretion, z_array[i])
            if (array_progenitors[1] > array_progenitors[2])
                frac_init = array_progenitors[1] / Mhost_init
                append!(subhalo_mass, array_progenitors[2] / Mhost_init)
            else
                frac_init = array_progenitors[2] / Mhost_init
                append!(subhalo_mass, array_progenitors[1] / Mhost_init)
            end

            the_end = false
        end

        if i%1000 == 0
            println(z_array[i], " ", size(subhalo_mass)[1], " ", frac_init)
        end

        i = i+1

    end

    return subhalo_mass, z_accretion
end


function z_vs_Δω(z0::Real, Δω::Real; growth_function::Function = growth_factor_Carroll, δ_c::Real = 1.686)
    return find_zero(z -> δ_c * growth_function(0) * (1.0 / growth_function(z) - 1.0 /  growth_function(z0))  - Δω, (0, 1e+3), Bisection(), rtol=1e-6)
end


function redshift_array(z0::Real, Δω::Real; growth_function::Function = growth_factor_Carroll, δ_c::Real = 1.686)
    
    z_array = zeros(0)
    append!(z_array, z_vs_Δω(z0, Δω, growth_function = growth_function, δ_c = δ_c))

    i = 1

    while (i < 1e+8 && z_array[i] < 10)
        append!(z_array, z_vs_Δω(z_array[i], Δω, growth_function = growth_function, δ_c = δ_c))
        i = i+1
    end
    istop = i
    while (i < 1e+8 && z_array[i] < 50)
        append!(z_array, z_array[i] + Δω / growth_function(0) / δ_c * (1 + z_array[istop]) * growth_function(z_array[istop]) )
        i = i+1
    end 

    return z_array
end



end # end of module MassFunction