
module StellarEncounters

include("./Halos.jl")
include("./Hosts.jl")

import QuadGK: quadgk
using JLD2,  Interpolations, Roots, SpecialFunctions
import Unitful: km, s, Gyr, K, Myr, NoUnits
import UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

import Main.Cosmojuly.BackgroundCosmo: BkgCosmology, planck18_bkg, lookback_redshift, δt_s, z_to_a
import Main.Cosmojuly.PowerSpectrum: Cosmology, planck18
import Main.Cosmojuly.Halos: Halo, HaloProfile, nfwProfile, αβγProfile, halo_from_ρs_and_rs, m_halo, ρ_halo, μ_halo, coreProfile, plummerProfile
import Main.Cosmojuly.Hosts: ρ_stellar_disc, HostModel, σ_stellar_disc, circular_velocity, velocity_dispersion_spherical, MM17Gamma1, name_model

#######################
## STAR PROPERTIES

export stellar_mass_function_C03, moments_C03, b_max, number_stellar_encounter, pdf_relative_speed, pdf_η, pseudo_mass, _pseudo_mass
export w_parallel, w_perp, cdf_η, inverse_cdf_η, average_relative_speed, average_inverse_relative_speed

""" result in (Msol^{-1}) from Chabrier 2003 """
function stellar_mass_function_C03(m::Real)
    
    (m <= 1) && (return 0.158 * exp(-(log10(m) - log10(0.079))^2 / (2. * 0.69^2)) / m / 0.6046645064846679) 
    (0 < log10(m) && log10(m) <= 0.54) && (return 4.4e-2 *  m^(-5.37) / 0.6046645064846679)
    (0.54 < log10(m) && log10(m) <= 1.26) && (return 1.5e-2 * m^(-4.53) / 0.6046645064846679)
    (1.26 < log10(m) && log10(m) <= 1.80) && (return 2.5e-4 * m^(-3.11) / 0.6046645064846679)

    return 0
end

function moments_C03(n::Int)
    return quadgk(lnm -> exp(lnm)^(n+1) * stellar_mass_function_C03(exp(lnm)), log(1e-7), log(10.0^1.8), rtol=1e-10)[1] 
end 

function b_max(r::Real, ::Type{T}) where {T<:HostModel}
    return moments_C03(1)^(1/3)/σ_stellar_disc(r, T) * quadgk(lnz -> exp(lnz) * ρ_stellar_disc(r, exp(lnz))^(2/3), log(1e-10), log(1e+0), rtol=1e-10)[1] 
end

function number_stellar_encounter(r::Real, ::Type{T}) where {T<:HostModel}
    return floor(Int, σ_stellar_disc(r, T) / moments_C03(1) * π / 0.5 * b_max(r, T)^2)
end

pdf_relative_speed(v::Real, σ::Real, vstar::Real) = (vstar^2 + v^2)/(2.0*σ^2) > 1e+2 ? 0.0 : sqrt(2.0/π) * v / (σ * vstar) * sinh(v * vstar / σ^2) *exp(-(vstar^2 + v^2)/(2.0*σ^2))
pdf_relative_speed(v::Real, r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = pdf_relative_speed(v, velocity_dispersion_spherical(r, T), circular_velocity(r, T))

""" average relative speed in units of σ and vstar """
function average_relative_speed(σ::Real, vstar::Real)
    X = vstar / (sqrt(2.0) * σ)
    return σ * sqrt(2.0 / π) * (exp(-X^2) + sqrt(π)/2.0*(1+2*X^2)*erf(X)/X)
end

average_inverse_relative_speed(σ::Real, vstar::Real) = erf(vstar/(sqrt(2.0) * σ))/vstar

"""  η = m/<m> * <v>/v """
function pdf_η(η::Real, σ::Real, vstar::Real, mstar_avg::Real, v_avg::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} 
    v_m = v_avg / mstar_avg
    v_max = 10^(1.80) * v_m
    v_min = 1e-5 * v_m
    return  quadgk(lnv -> stellar_mass_function_C03((exp(lnv) / v_m * η)) * pdf_relative_speed(exp(lnv), σ, vstar) * exp(lnv)^2, log(v_min), log(v_max))[1] / v_m
end

#cdf_η(η::Real, σ::Real, vstar::Real, mstar_avg::Real, v_avg::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} = quadgk(lnu -> pdf_u(exp(lnu), r, σ, vstar, T) * exp(lnu), log(1e-12), log(u), rtol=1e-6)[1]

function cdf_η(η::Real, σ::Real, vstar::Real, mstar_avg::Real, v_avg::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel}
    x = vstar / (sqrt(2.0) * σ)
    
    function integrand(m::Real)
        y = v_avg / (sqrt(2.0) * σ) * m / mstar_avg / η
        res = (exp(-(x+y)^2)*(-1 + exp(4.0*x*y))/sqrt(π)/x + erfc(y-x) + erfc(x+y))/2.0
        (res === NaN) && return 0.0
        return res
    end

    return quadgk(lnm -> integrand(exp(lnm)) * stellar_mass_function_C03(exp(lnm)) * exp(lnm), log(1e-7), log(10.0^1.8), rtol=1e-10)[1]
end


function inverse_cdf_η(rnd::Real, σ::Real, vstar::Real, mstar_avg::Real, v_avg::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} 
    return exp(find_zero(lnu -> cdf_η(exp(lnu), σ, vstar, mstar_avg, v_avg, T) - rnd, (log(1e-8), log(1e+6)), Bisection(), rtol=1e-10, atol=1e-10)) 
end

export draw_velocity_kick

function draw_velocity_kick(rp::Real, subhalo::Halo, r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} 
    
    n = number_stellar_encounter(r, T)
    b_max = b_max(r, T)
    inv_η = _load_inverse_cdf_η(r, T)

    rt = jacobi_radius(r, subhalo, T)
    rs = subhalo.rs

    (rp > rt) && return false

    # Randomly sampling the distributions
    θb = 2.0 * π * rand(n)
    β = rand(n)

    b = b_max * β # assuming b_min = 0 here

    η          = inv_η.(rand(n))
    w_parallel = w_parallel.(rp, θb, b / rs, rt / rs)
    w_perp     = w_perp.(rp, θb, b / rs, rt / rs)

    v_parallel = w_parallel .* η ./ b * b_max 
    v_perp = w_perp .* η ./ b * b_max 

    return v_parallel, v_perp
end


export _save_inverse_cdf_η, _load_inverse_cdf_η

function _save_inverse_cdf_η(r::Real, ::Type{T}) where {T<:HostModel}
    
    σ         = velocity_dispersion_spherical(r, T)
    vstar     = circular_velocity(r, T)
    mstar_avg = moments_C03(1)
    v_avg     = 1.0/average_inverse_relative_speed(σ, vstar)

    rnd_array = 10.0.^range(-8, -0.000001, 500)

    # -------------------------------------------
    # Checking if the file does not already exist
    hash_value = hash((r, name_model(T)))
    file = "cdf_eta_" * string(hash_value, base=16) * ".jld2" 
 
    if file in readdir("../cache/hosts/")
        existing_data = jldopen("../cache/hosts/cdf_eta_" * string(hash_value, base=16) * ".jld2")
        (existing_data["r"] == r_array) && @info "| file to save is already cached" && return nothing
    end
    # -------------------------------------------

    inv_cdf = inverse_cdf_η.(rnd_array, σ, vstar, mstar_avg, v_avg, T)
    jldsave("../cache/hosts/cdf_eta_" * string(hash_value, base=16) * ".jld2"; rnd = rnd_array, inverse_cdf_eta = inv_cdf)

    return true
end



## Possibility to interpolate the model
function _load_inverse_cdf_η(r::Real, ::Type{T}) where {T<:HostModel}
    """ change that to a save function """

    hash_value = hash((r, name_model(T)))
    filenames = readdir("../cache/hosts/")

    file = "cdf_eta_" * string(hash_value, base=16) * ".jld2" 


    if file in filenames
        data = jldopen("../cache/hosts/" * file)
        rnd_array = data["rnd"]
        inv_cdf = data["inverse_cdf_eta"]
    end

    
    log10inv_cdf = interpolate((log10.(rnd_array),), log10.(inv_cdf),  Gridded(Linear()))
    inv_cdf_η(rnd::Real) = 10.0^log10inv_cdf(log10(rnd)) 

    return inv_cdf_η

end



function pseudo_mass(bs::Real, xt::Real, shp::HaloProfile = nfwProfile)

    (xt <= bs)  && return 1.0
    (bs < 1e-5) && return _pseudo_mass(bs, xt, shp)

    ((typeof(shp) <: αβγProfile) && (shp == plummerProfile)) && return (1.0 - (1.0 / (1.0 + bs^2)) * (1 - bs^2 / (xt^2))^(1.5))

    if ((typeof(shp) <: αβγProfile) && (shp == nfwProfile))
        (bs > 1)  && return 1.0 + (sqrt(xt * xt - bs * bs) / (1 + xt) - acosh(xt / bs) + (2. / sqrt(bs * bs - 1)) * atan(sqrt((bs - 1) / (bs + 1)) * tanh(0.5 * acosh(xt / bs)))) / μ_halo(xt, shp)
        (bs == 1) && return 1.0 - (-2 * sqrt((xt - 1) / (xt + 1)) + 2 * asinh(sqrt((xt - 1) / 2))) / μ_halo(xt, shp)
        (bs < 1)  && return 1.0 + (sqrt(xt * xt - bs * bs) / (1 + xt) - acosh(xt / bs) + (2. / sqrt(1 - bs * bs)) * atanh(sqrt((1 - bs) / (bs + 1)) * tanh(0.5 * acosh(xt / bs)))) / μ_halo(xt, shp)
    end 

    # For whatever different profile
    return _pseudo_mass(bs, xt, shp)

end

_pseudo_mass(bs::Real, xt::Real, shp::HaloProfile = nfwProfile) = 1.0 - quadgk(lnx-> sqrt(exp(lnx)^2 - bs^2)  * ρ_halo(exp(lnx), shp) * exp(lnx)^2 , log(bs), log(xt), rtol=1e-10)[1] / μ_halo(xt, shp)

pseudo_mass(b::Real, rt::Real, sh::Halo) = pseudo_mass(b/sh.rs, rt/sh.rs, sh.hp)


w_parallel(rps::Real, θb::Real, bs::Real, xt::Real, shp::HaloProfile = nfwProfile) = (pseudo_mass(bs, xt, shp) * sin(θb) - (rps * bs + bs^2 * sin(θb))/(rps^2 + bs^2 + 2*rps*bs*sin(θb)) )
w_perp(rps::Real, θb::Real, bs::Real, xt::Real, shp::HaloProfile = nfwProfile) = (pseudo_mass(bs, xt, shp) * cos(θb) - (bs^2 * cos(θb))/(rps^2 + bs^2 + 2*rps*bs*sin(θb)) )

#######################

end # module StellarEncounters
