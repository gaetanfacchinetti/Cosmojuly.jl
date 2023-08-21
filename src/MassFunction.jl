module MassFunction

include("./PowerSpectrum.jl")

using Roots, Random, Interpolations
using JLD2
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

export mass_vs_S, mass_fraction_unresolved, mean_number_small_progenitors, cmf_inv_progenitors
export mean_number_progenitors, subhalo_mass_function, cmf_progenitors, Δω_approx, redshift_array
export subhalo_mass_function_array, one_step_merger_tree, pdf_progenitors, interpolate_functions_PF, z_vs_Δω
export interpolate_functions_z
export _save_cmf_inv_progenitors, _load_cmf_inv_progenitors
export subhalo_mass_function_binned

# In the excursion set framework S=σ² 
# The only valid window function is the SharpK function
# All masses are in solar masses

mass_vs_S(S::Real; cosmology::Cosmology = planck18) = 10^(find_zero(y -> σ²_vs_M(10^y, SharpK, cosmology = cosmology) - S, (-20, 20), Bisection(), xrtol=1e-4))
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

    return  M1 / sqrt(2.0 * π)  * quadgk(lnM2 -> _integrand(exp(lnM2), _S1, cosmology) * exp(lnM2), log(Mres), log(M), rtol=1e-8, order=10)[1]
end


function pdf_progenitors(M2::Real, M1::Real, Mres::Real; cosmology::Cosmology = planck18) 

    if M2 < Mres
        return 0.0
    end

    _ΔS = σ²_vs_M(M2, SharpK, cosmology = cosmology) -  σ²_vs_M(M1, SharpK, cosmology = cosmology)
    _dS2_dM = dσ²_dM(M2, SharpK, cosmology = cosmology)

    return  M1 / M2 / sqrt(2.0 * π) / _ΔS^(3/2) * abs(_dS2_dM) / mean_number_progenitors(M1/2.0, M1, Mres, cosmology = cosmology) 
end

cmf_progenitors(M2::Real, M1::Real, Mres::Real; cosmology::Cosmology = planck18) =  M2 > M1/2.0 ? 1.0 : mean_number_progenitors(M2, M1, Mres, cosmology = cosmology) / mean_number_progenitors(M1/2.0, M1, Mres, cosmology = cosmology) 
cmf_progenitors(M2::Real, M1::Real, Mres::Real, nProgTot::Real; cosmology::Cosmology = planck18) = M2 > M1/2.0 ? 1.0 : mean_number_progenitors(M2, M1, Mres, cosmology = cosmology) / nProgTot

function cmf_inv_progenitors(x::Real, M1::Real, Mres::Real; cosmology::Cosmology = planck18) 
    
    if M1 <= 2.001*Mres
        return 0.0
    end

    nProgTot = mean_number_progenitors(M1/2.0, M1, Mres, cosmology = cosmology) 

    return 10.0^find_zero(z -> x - cmf_progenitors(10.0^z, M1, Mres, nProgTot, cosmology=cosmology), (log10(Mres), log10(M1/2.0)), Bisection(), xrtol=1e-3)
end



function _save_cmf_inv_progenitors(mhost::Real, mres::Real; p_array_size::Integer = 100, mass_array_size::Integer = 100,  cosmology::Cosmology = planck18)
   
    q_array = range(-8.0, 0, length=p_array_size)
    m1_array = 10.0.^range(log10(2.0*mres), log10(mhost), length=mass_array_size)

    #############################################
    # Checking if the file does not already exist
    hash_value = hash((mhost, mres, cosmology.name))
    filenames = readdir("../cache/")

    file = "cmf_inv_progenitors_" * string(hash_value, base=16) * ".jld2" 
 
    if file in filenames
        existing_data = jldopen("../cache/cmf_inv_progenitors_" * string(hash_value, base=16) * ".jld2")

        if existing_data["q"] == q_array && existing_data["m1"] == m1_array
            println("This file is already cached")
            return nothing
        end
    end
    #############################################

    _cmf_inv_progenitors = cmf_inv_progenitors.((1.0 .- 10.0.^q_array)', m1_array, mres, cosmology=cosmology)

    jldsave("../cache/cmf_inv_progenitors_" * string(hash_value, base=16) * ".jld2"; 
        q = q_array, 
        m1 = m1_array, 
        cmf = _cmf_inv_progenitors)

end


function _load_cmf_inv_progenitors(mhost::Real, mres::Real; cosmology::Cosmology = planck18)

    hash_value = hash((mhost, mres, cosmology.name))
    filenames = readdir("../cache/")

    file = "cmf_inv_progenitors_" * string(hash_value, base=16) * ".jld2" 

    if file in filenames

        data = jldopen("../cache/" * file)
        q = data["q"]
        m1 = data["m1"]
        cmf = data["cmf"]

        log10_func = interpolate((q, log10.(m1),), log10.(cmf'), Gridded(Linear()))
        
        func(p::Real, m1::Real) = log10(1.0-p) > -8.0 ? 10.0.^log10_func(log10(1.0-p), log10(m1)) : cmf_inv_progenitors(p, m1, mres, cosmology=cosmology)

        return func

    end

    return nothing

end

# Random number given externally
function cmf_inv_progenitors(x::Real, M1::Real, Mres::Real, itp_functionP::Union{Function, Nothing}; cosmology::Cosmology = planck18) 
    if itp_functionP === nothing 
        return cmf_inv_progenitors(x, M1, Mres, cosmology=cosmology) 
    end
    _cmf_inv_progenitors(M2::Real, M1::Real, Mres::Real, itp_functionP::Function, cosmology::Cosmology = planck18) =  M2 > M1/2.0 ? 1.0 : mean_number_progenitors(M2, M1, Mres, cosmology = cosmology) / itp_functionP(M1) 
    return 10.0^find_zero(z -> x - _cmf_inv_progenitors(10.0^z, M1, Mres, itp_functionP, cosmology), (log10(Mres), log10(M1/2.0)), Bisection(), xrtol=1e-3)
end


function one_step_merger_tree(P::Real, F::Real, M1::Real, Mres::Real, cmf::Function, cosmology::Cosmology = planck18, itp_functionP::Union{Function, Nothing} = nothing)

    R = rand(Float64)
    array_progenitors = zeros(0)

    if (P < R || M1 < 2*Mres)
        if (M1 * (1-F) > Mres) 
            append!(array_progenitors, M1 * (1-F)) 
        end 
    else
        M2 = cmf(rand(Float64), M1)
        if (M1 * (1-F) - M2 > Mres)
            append!(array_progenitors, M1 * (1-F) - M2)
        end
        append!(array_progenitors, M2)
    end

    return array_progenitors
end

    
Δω_approx(Mhost::Real, Mres::Real; cosmology::Cosmology = planck18) = sqrt(1e-2 * Mres * abs(dσ²_dM(Mhost, SharpK, cosmology = cosmology)))


function interpolate_functions_PF(Mhost_init::Real, Mres::Real; cosmology::Cosmology=planck18)

    log10m_P = range(log10(2.0 * Mres),stop=log10(Mhost_init),length=2000)
    log10m_F = range(log10(Mres),stop=log10(Mhost_init),length=4000)
    function_log10P = interpolate((log10m_P,), log10.(mean_number_progenitors.(10.0 .^log10m_P  ./ 2.0, 10 .^log10m_P, Mres, cosmology=cosmology)),  Gridded(Linear()))
    function_log10F = interpolate((log10m_F,), log10.(mass_fraction_unresolved.(10 .^log10m_F, Mres, cosmology=cosmology)),  Gridded(Linear()))
  
    function_P(M1::Real, Mres::Real) = M1 < 2.0*Mres ? 0 : 10.0^function_log10P(log10(M1))
    function_F(M1::Real) = 10.0^function_log10F(log10(M1))
    
    return function_P, function_F
end



function subhalo_mass_function(Mhost_init::Real, Mres::Real; 
                                z_vs_Δω::Function = _z_vs_Δω,
                                cosmology::Cosmology=planck18,
                                load_cmf_inv_progenitors::Bool = true,
                                save_cmf_inv_progenitors::Bool = true)


    #############################
    # Loading / computing precomputed tables
    function_P, function_F = interpolate_functions_PF(Mhost_init, Mres, cosmology=cosmology)
    
    _cmf_inv_progenitors::Union{Nothing, Function} = nothing

    if save_cmf_inv_progenitors === true
        _save_cmf_inv_progenitors(Mhost_init, Mres, p_array_size = 100, mass_array_size = 100, cosmology = cosmology)
    end

    if load_cmf_inv_progenitors === true 
        _cmf_inv_progenitors = _load_cmf_inv_progenitors(Mhost_init, Mres, cosmology=cosmology)
    end

    if load_cmf_inv_progenitors === false || _cmf_inv_progenitors === nothing
        _cmf_inv_progenitors = (x, M1) -> cmf_inv_progenitors(x, M1, Mres, cosmology=cosmology)
    end
    #############################

    
    # initialisation of the values
    the_end = false
    frac_init = 1.0
    z = 0.0

    z_steps     = zeros(0)
    m_host      = zeros(0)
    z_accretion = zeros(0)
    subhalo_mass = zeros(0)

    i = 1

    P = 0.1
    δω = P / function_P(Mhost_init, Mres)
    Δz = z_vs_Δω(δω)
    Δω = 0.0

    Mhost = Mhost_init


    while (the_end == false && Mhost/2.0 > Mres)

        the_end = true

        δω = P / function_P(Mhost, Mres)
        F = δω * function_F(Mhost)

        Δω = Δω + δω
        if Δω == Inf
            println(Mhost, " ", Mhost/2.0, " ", Mres, " ",  Δω, " ", function_P(Mhost, Mres), " ", mean_number_progenitors(Mhost/2.0, Mhost, Mres))
        end

        z = z_vs_Δω(Δω)
        
        array_progenitors = one_step_merger_tree(P, F, Mhost, Mres, _cmf_inv_progenitors, cosmology,  Mhost->function_P(Mhost, Mres))

        if (size(array_progenitors)[1] == 1)
            frac_init = array_progenitors[1]/Mhost_init
            the_end = false
        end

        if (size(array_progenitors)[1]==0)
            frac_init = 0.0
        end

        if (size(array_progenitors)[1]==2)
            append!(z_accretion, z)
            if (array_progenitors[1] > array_progenitors[2])
                frac_init = array_progenitors[1] / Mhost_init
                append!(subhalo_mass, array_progenitors[2] / Mhost_init)
            else
                frac_init = array_progenitors[2] / Mhost_init
                append!(subhalo_mass, array_progenitors[1] / Mhost_init)
            end

            the_end = false
        end

        Mhost = Mhost_init * frac_init
        i = i+1

        if i%max(trunc(0.03/Δz), 1) == 0
            append!(z_steps, z)
            append!(m_host, Mhost)   
        end

      

    end

    return subhalo_mass, m_host, z_steps, z_accretion
end


function subhalo_mass_function_binned(mhost::Real, mres::Real; 
    z_vs_Δω::Function = _z_vs_Δω,
    cosmology::Cosmology=planck18,
    load_cmf_inv_progenitors::Bool = true,
    save_cmf_inv_progenitors::Bool = true)

    #############################
    # Loading / computing precomputed tables
    println("INITIALISATION")
    println(" - precomputing or loading the interpolation tables")

    function_P, function_F = interpolate_functions_PF(mhost, mres, cosmology=cosmology)
    
    _cmf_inv_progenitors::Union{Nothing, Function} = nothing

    if save_cmf_inv_progenitors === true
        _save_cmf_inv_progenitors(mhost, mres, p_array_size = 100, mass_array_size = 100, cosmology = cosmology)
    end

    if load_cmf_inv_progenitors === true 
        _cmf_inv_progenitors = _load_cmf_inv_progenitors(mhost, mres, cosmology=cosmology)
    end

    if load_cmf_inv_progenitors === false || _cmf_inv_progenitors === nothing
        _cmf_inv_progenitors = (x, m1) -> cmf_inv_progenitors(x, m1, mres, cosmology=cosmology)
    end
    println("INITIALISATION DONE: STARTING MERGER TREE")
    #############################

    # initialisation of the values
    the_end = false
    frac_init = 1.0
    z = 0.0

    i = 1

    P = 0.1
    δω = P / function_P(mhost, mres)
    Δω = 0.0

    mmain = mhost

    n_bins     = trunc(Int64, log10.(mhost/mres)) * 10
    mass_edges = 10.0.^range(log10(mres/mhost), 0, n_bins+1)
    z_bins     = zeros(UInt64, (n_bins, 60))
    z_edges    = 10.0.^range(-9, 5, 61)


    while (the_end == false && mmain/2.0 > mres)

        the_end = true

        δω = P / function_P(mmain, mres)
        F = δω * function_F(mmain)

        Δω = Δω + δω
   
        z = z_vs_Δω(Δω)
        
        array_progenitors = one_step_merger_tree(P, F, mmain, mres, _cmf_inv_progenitors, cosmology,  x->function_P(x, mres))

        if (size(array_progenitors)[1] == 1)
            frac_init = array_progenitors[1]/mhost
            the_end = false
        end

        if (size(array_progenitors)[1]==0)
            frac_init = 0.0
        end

        if (size(array_progenitors)[1]==2)

            if (array_progenitors[1] > array_progenitors[2])
                frac_init = array_progenitors[1] / mhost
                mass_progenitor = array_progenitors[2]
            else
                frac_init = array_progenitors[2] / mhost
                mass_progenitor = array_progenitors[1]
            end

            mass_index = searchsortedfirst(mass_edges, mass_progenitor / mhost) - 1
            z_index    = searchsortedfirst(z_edges, z) - 1

            z_bins[mass_index, z_index] = z_bins[mass_index, z_index] + 1

            the_end = false
        end

        mmain = mhost * frac_init
        i = i+1

    end

    return z_bins, mass_edges, z_edges

end

function interpolate_functions_z(z0::Real = 0; growth_function::Function = growth_factor_Carroll, δ_c::Real = 1.686)
   
    log10_Δω = range(-10,stop=2,length=5000)
    function_log10_z = interpolate((log10_Δω,), log10.(_z_vs_Δω.(10.0.^log10_Δω, z0, growth_function=growth_function, δ_c=δ_c)),  Gridded(Linear()))
    
    function_z(Δω::Real) = 10.0^function_log10_z(log10(Δω))
    
    return function_z
end


function Δz_vs_Δω(Δω::Real, z0::Real=0; growth_function::Function = growth_factor_Carroll, δ_c::Real = 1.686)
    if z0 < 10.0
        return 10.0^find_zero(log10z -> δ_c * growth_function(0) * (1.0 / growth_function(10^log10z) - 1.0 /  growth_function(z0)) - Δω, (-10, +3), Bisection(), xrtol=1e-10) - z0
    else
        return Δω / growth_function(0) / δ_c
    end
end

function _z_vs_Δω(Δω::Real, z0::Real = 0; growth_function::Function = growth_factor_Carroll, δ_c::Real = 1.686)
    return 10.0^find_zero(log10z -> δ_c * growth_function(0) * (1.0 / growth_function(10^log10z) - 1.0 /  growth_function(z0)) - Δω, (-10, +3), Bisection(), xrtol=1e-10)
end


function one_step_merger_tree(P::Vector{<:Real}, F::Vector{<:Real}, M1::Vector{<:Real}, Mres::Real, cosmology::Cosmology = planck18, itp_functionP::Union{Function, Nothing} = nothing)

    n = length(M1)
    _R = rand(Float64, n)

    M2 = zeros(n)
    array_progenitors = zeros(n, 2)

    id_tree_0 = ((P .< _R) .| (M1 .< 2 * Mres)  .&  (M1 .* (1.0 .-F) .> Mres)) 
    array_progenitors[id_tree_0, 1] =  M1[id_tree_0] .* (1.0 .- F[id_tree_0])

    id_tree_1 = ((P .> _R) .& (M1 .> 2 * Mres))

    _random = rand(Float64, sum(id_tree_1))

    #_cumulative_progenitors(M2::Real, M1::Real, Mres::Real, itp_functionP::Function, cosmology::Cosmology = planck18) =  M2 > M1/2.0 ? 1.0 : mean_number_progenitors(M2, M1, Mres, cosmology = cosmology) / itp_functionP(M1) 
    #M2[id_tree_1] = [10.0^find_zero(z -> _random[ind] - _cumulative_progenitors(10.0^z, M1[ind], Mres, itp_functionP, cosmology), (log10(Mres), log10.(M1[ind]/2.0)), Bisection(), rtol=1e-3) for ind in findall(id_tree_1)]

    M2[id_tree_1] = draw_mass_with_restrictions.(_random, M1[id_tree_1], Mres, itp_functionP, cosmology = cosmology)
    array_progenitors[id_tree_1, 2] =  M2[id_tree_1]
    id_tree_2 = (id_tree_1 .& (M1 .* (1.0 .- F) .- M2 .> Mres))
    array_progenitors[id_tree_2, 1] =  M1[id_tree_2] .* (1.0 .- F[id_tree_2]) .- M2[id_tree_2]

    return array_progenitors
end


function subhalo_mass_function_array(Mhost_init::Real, Mres::Real; 
                                cosmology::Cosmology=planck18, 
                                growth_function::Function = growth_factor_Carroll, 
                                δ_c::Real = 1.686)


    log10m_P = range(log10(log10(2.0 * Mres)),stop=log10(log10(Mhost_init)),length=2000)
    log10m_F = range(log10(log10(Mres)),stop=log10(log10(Mhost_init)),length=4000)
    function_log10P = interpolate((log10m_P,), log10.(mean_number_progenitors.(10.0 .^ (10.0 .^log10m_P ) ./ 2.0, 10.0 .^(10 .^log10m_P), Mres, cosmology=cosmology)),  Gridded(Linear()))
    function_log10F = interpolate((log10m_F,), log10.(mass_fraction_unresolved.(10.0 .^(10 .^log10m_F), Mres, cosmology=cosmology)),  Gridded(Linear()))
  
    function function_P(M1::Vector{<:Real}, Mres::Real) 
        n = length(M1)
        res = zeros(n)
        res[M1/2.0 .> Mres] = 10.0.^function_log10P.(log10.(log10.(M1[M1/2.0 .> Mres])))
        return res
    end

    function_F(M1::Vector{<:Real}) = 10.0.^function_log10F.(log10.(log10.(M1)))


    # initialisation of the values
    n=2
    active = fill(true, n)
    frac_init = ones(n)
    z = zeros(n)
    Mhost = zeros(n)
    Δω = zeros(n)
    P = zeros(n)
    F = zeros(n)

    subhalo_mass = [Vector{Float64}(undef, 0) for _ in 1:n]

    i = 1

    while any(active)

        Mhost[active] = Mhost_init .* frac_init[active]

        Δω[active] = 0.005 .* Δω_approx.(Mhost[active], Mres)
        z[active] = z_vs_Δω.(z[active], Δω[active], growth_function=growth_function, δ_c = δ_c)

        P[active] = Δω[active] .* function_P(Mhost[active], Mres)
        F[active] = Δω[active] .* function_F(Mhost[active])

        #array_progenitors = one_step_merger_tree(P, F, Mhost, Mres, cosmology,  x->function_P(x, Mres))
        array_progenitors=[[1e+10, 0.0];; [1e+10, 0.0]]
        
        trees_zero_progenitors = active .& ((array_progenitors[:, 1] .== 0.0) .& (array_progenitors[:, 2] .== 0.0))
        frac_init[trees_zero_progenitors] .= 0.0
        active[trees_zero_progenitors] .= false

        trees_one_progenitor = active .& (array_progenitors[:, 2] .== 0)
        frac_init[trees_one_progenitor] = array_progenitors[trees_one_progenitor, 1] ./ Mhost_init
    
        trees_two_progenitors = active .& (array_progenitors[:, 1] .> 0.0) .& (array_progenitors[:, 2] .> 0.0)

        for index in findall(trees_two_progenitors)
            push!(subhalo_mass[index], min(array_progenitors[index, 1], array_progenitors[index, 2]) / Mhost_init)
        end
        frac_init[trees_two_progenitors] = max.(array_progenitors[trees_two_progenitors, 1], array_progenitors[trees_two_progenitors, 2])./ Mhost_init
        
        if i%10000 == 0
            println(i, " ", active)
        end

        i = i+1

    end

    return subhalo_mass
end


end # end of module MassFunction