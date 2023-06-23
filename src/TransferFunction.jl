module TransferFunction

include("../src/MyCosmology.jl")

using .MyCosmology

abstract type ParametersTF{T<:Real} end

struct ParametersEH98{T<:Real} <: ParametersTF{T}
    
    with_baryons::Bool
    cosmo::Cosmology{T}
    z_drag::T
    sound_horizon_Mpc::T
    α_c::T
    α_b::T
    β_c::T
    β_b::T
    k_Silk_Mpc::T
   
end

g_func(y::Real)::Real = -6.0*sqrt(1.0+y)+(2.0+3.0*y)*log((sqrt(1.0+y)+1)/(sqrt(1.0+y)-1.0)) * y

function ParametersEH98(with_baryons::Bool, cosmo::Cosmology{<:Real}, ::Type{T} = Float64) where {T<:Real}
    
    Ω_m0_h2 = cosmo.Ω_m0 * cosmo.h^2
    Ω_b0_h2 = cosmo.Ω_b0 * cosmo.h^2
    Ω_χ0_h2 = cosmo.Ω_χ0 * cosmo.h^2

    Θ27 = cosmo.T0_CMB_K / 2.7
    z_eq = z_eq_mr(cosmo)
    k_eq_Mpc = k_eq_mr_Mpc(cosmo)

    # z_drag
    b1::T = 0.313 * Ω_m0_h2^(-0.419) * (1. + 0.607 * Ω_m0_h2^0.674)
    b2::T = 0.238 * Ω_m0_h2^0.223
    z_drag::T = 1291. * Ω_m0_h2^0.251 / (1. + 0.659 * Ω_m0_h2^0.828)  * (1. + b1 * Ω_b0_h2^b2)
    
    # sound horizon
    R_drag::T = 31.5 * Ω_b0_h2 * Θ27^(-4) * 1e+3 / z_drag
    R_eq::T   = 31.5 * Ω_b0_h2 * Θ27^(-4) * 1e+3 / z_eq
    sound_horizon_Mpc::T = 2. / (3. * k_eq_Mpc) * sqrt(6. / R_eq) * log((sqrt(1. + R_drag) + sqrt(R_drag + R_eq))/(1+sqrt(R_eq)))

    # α_c
    a1::T  = (46.9 * Ω_m0_h2)^0.670 * (1.0 + (32.1 * Ω_m0_h2)^(-0.532) )
    a2::T  = (12.0 * Ω_m0_h2)^0.424 * (1.0 + (45.0 * Ω_m0_h2)^(-0.582) )
    α_c::T =  a1^(-Ω_b0_h2/Ω_m0_h2) * a2^(-(Ω_b0_h2/Ω_m0_h2)^3) 
    
    # β_c
    b1_2::T = 0.944 / (1+(458.0*Ω_m0_h2)^(-0.708))
    b2_2::T = (0.395 * Ω_m0_h2)^(-0.0266)
    β_c::T  = 1.0 / (1.0 + b1_2 * ((Ω_χ0_h2/Ω_m0_h2)^b2_2 -.01))

    # k_silk, α_b, and β_b
    k_Silk_Mpc::T = 1.6 * ( Ω_b0_h2^0.52 ) * (Ω_m0_h2^0.73 ) * ( 1.0 + (10.4 * Ω_m0_h2)^(-0.95) )
    α_b::T = 2.07 * k_eq_Mpc * sound_horizon_Mpc * (1.0 + R_drag)^(-0.75) * g_func( (1.0 + z_eq) / (1.0 + z_drag) )
    β_b::T = 0.5 + Ω_b0_h2 / Ω_m0_h2 + (3.0 - 2.0 * Ω_b0_h2/Ω_m0_h2) * sqrt( 1.0 + (17.2 * Ω_m0_h2)^2 )

    return ParametersEH98(with_baryons, convert_cosmo(T, cosmo), z_drag, sound_horizon_Mpc, α_c, α_b, β_c, β_b, k_Silk_Mpc)
end


const parametersEH98_planck18_wi_baryons = ParametersEH98(true, planck18)
const parametersEH98_planck18_wo_baryons = ParametersEH98(false, planck18)

function temperature_0_tilde(q::Real, α_c::Real, β_c::Real)::Real
    C = 14.2 / α_c + 386.0 / (1.0 + 69.9 * q^1.08 )
    return log(exp(1.0) + 1.8 * β_c * q ) / (log(exp(1.0) + 1.8 * β_c * q) + C * q^2)
end 

shape_parameter(k_Mpc::Real, p::ParametersEH98)::Real = k_Mpc * (p.cosmo.T0_CMB_K / 2.7)^2 / (p.cosmo.Ω_m0 * p.cosmo.h^2)

function temperature_cdm(k_Mpc::Real, p::ParametersEH98)::Real
    q = shape_parameter(k_Mpc, p)
    t0_1 = temperature_0_tilde(q, 1.0, p.β_c)
    t0_2 = temperature_0_tilde(q, p.α_c, p.β_c)
    f = 1.0 / (1.0  +  (k_Mpc * p.sound_horizon_Mpc / 5.4)^4)

    return f * t0_1 + (1-f) * t0_2
end

function s_tilde_Mpc(k_Mpc::Real, p::ParametersEH98)::Real
    β_node = 8.41 * (p.cosmo.Ω_m0 * p.cosmo.h^2)^0.435
    return p.sound_horizon_Mpc * (1.0 + (β_node/(k_Mpc * p.sound_horizon_Mpc))^3 )^1.0/3.0
end

function temperature_baryons(k_Mpc::Real, p::ParametersEH98)::Real
    q = shape_parameter(k_Mpc, p)
    j0 = sinc(k_Mpc * s_tilde_Mpc(k_Mpc, p))
    return j0 * temperature_0_tilde(q, 1.0, 1.0) / (1.0 + (k_Mpc * p.sound_horizon_Mpc/5.2)^2 ) + p.α_b/(1.0 + (p.β_b / (k_Mpc * p.sound_horizon_Mpc))^3 ) * exp(-(k_Mpc/p.k_Silk_Mpc)^1.4)
end

function transfer_function(k_Mpc::Real, p::ParametersEH98 = parametersEH98_planck18_wi_baryons)::Real
    tc = temperature_cdm(k_Mpc, p)
    tb = temperature_baryons(k_Mpc, p)
    return p.with_baryons ? abs(p.cosmo.Ω_b0 / p.cosmo.Ω_m0 * tb + p.cosmo.Ω_χ0/p.cosmo.Ω_m0 * tc) : abs(p.cosmo.Ω_χ0 / p.cosmo.Ω_m0 * tc)
end

function transfer_function(k_Mpc::Real, p::ParametersTF{<:Real})::Real end

end # module TransferFunction
