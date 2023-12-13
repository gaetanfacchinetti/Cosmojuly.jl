
module StellarEncounters

include("./Halos.jl")
include("./Hosts.jl")


import QuadGK: quadgk
using JLD2,  Interpolations, Roots, SpecialFunctions, Base.Threads, Statistics
import Unitful: km, s, Gyr, K, Myr, NoUnits
import UnitfulAstro: Mpc, Gpc, Msun
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

import Main.Cosmojuly.BackgroundCosmo: BkgCosmology, planck18_bkg, lookback_redshift, δt_s, z_to_a
import Main.Cosmojuly.PowerSpectrum: Cosmology, planck18
import Main.Cosmojuly.Halos: Halo, HaloProfile, nfwProfile, αβγProfile, halo_from_ρs_and_rs, m_halo, ρ_halo, μ_halo, coreProfile, plummerProfile, velocity_dispersion
import Main.Cosmojuly.Hosts: ρ_stellar_disc, HostModel, σ_stellar_disc, circular_velocity, velocity_dispersion_spherical, MM17Gamma1, name_model, ρ_host_spherical, m_host_spherical

#######################
## STAR PROPERTIES

export stellar_mass_function_C03, moments_C03, b_max, number_stellar_encounter, pdf_relative_speed, pdf_η, pseudo_mass, _pseudo_mass
export w_parallel, w_perp, cdf_η, inverse_cdf_η, average_relative_speed, average_inverse_relative_speed
export jacobi_radius, jacobi_scale

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
    return floor(Int, σ_stellar_disc(r, T) / moments_C03(1) * π / 1 * b_max(r, T)^2)
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

""" bs = b / rs """
w_parallel(xp::Real, θb::Real, bs::Real, xt::Real, shp::HaloProfile = nfwProfile) = (pseudo_mass(bs, xt, shp) * sin(θb) - (xp * bs + bs^2 * sin(θb))/(xp^2 + bs^2 + 2*xp*bs*sin(θb)) )
w_perp(xp::Real, θb::Real, bs::Real, xt::Real, shp::HaloProfile = nfwProfile) = (pseudo_mass(bs, xt, shp) * cos(θb) - (bs^2 * cos(θb))/(xp^2 + bs^2 + 2*xp*bs*sin(θb)) )

function w(xp::Real, θb::Real, bs::Real, xt::Real, shp::HaloProfile = nfwProfile)
    
    sθb   = sin(θb)
    cθb   = cos(θb)
    denom = (xp^2 + bs^2 + 2*xp*bs*sθb)
    pm    = pseudo_mass(bs, xt, shp)
  
    (pm * sθb - (xp * bs + bs^2 * sθb)/denom), (pm * cθb - (bs^2 * cθb)/denom) 
end


function draw_velocity_kick(rp::Real, subhalo::Halo, r::Real, ::Type{T} = MM17Gamma1) where {T<:HostModel} 
    
    # initialisation for a given value of r
    n     = number_stellar_encounter(r, T)
    b_m   = b_max(r, T)
    inv_η = _load_inverse_cdf_η(r, T)

    rt = jacobi_radius(r, subhalo, T)
    rs = subhalo.rs

    (rp > rt) && return false

    # Randomly sampling the distributions
    θb = 2.0 * π * rand(n)
    β  = sqrt.(rand(n))
    η  = inv_η.(rand(n))

    v_parallel = w_parallel.(rp / rs, θb, b_m * β / rs, rt / rs, subhalo.hp) .* η ./ β # assuming b_min = 0 here
    v_perp     = w_perp.(rp / rs, θb, b_m * β / rs, rt / rs, subhalo.hp) .* η ./ β     # assuming b_min = 0 here
    
    return v_parallel, v_perp
end


export draw_parameter_B

draw_parameter_B(r::Real, ::Type{T} = MM17Gamma1; nstars::Int = number_stellar_encounter(r, T)) where {T<:HostModel}  = draw_parameter_B(r, velocity_dispersion_spherical(r, T), circular_velocity(r, T), T)

function draw_parameter_B(r::Real, σ::Real, vstar::Real, ::Type{T} = MM17Gamma1; nstars::Int = number_stellar_encounter(r, T)) where {T<:HostModel} 
    
    # initialisation for a given value of r
    b_m   = b_max(r, T)
    inv_η = _load_inverse_cdf_η(r, T)
    mstar_avg = moments_C03(1)
    v_avg     = 1.0/average_inverse_relative_speed(σ, vstar)

    println("average velocity = ", v_avg, " km/s | mstat_avg = ", mstar_avg)

    # Randomly sampling the distributions
    pref = 2*G_NEWTON * (mstar_avg * Msun) / (v_avg * (km / s)) / (b_m^2 * Mpc^2) * s |> NoUnits 

    return pref * (inv_η.(rand(nstars)) ./ rand(nstars))
end





""" nmem maximal number of iteration for the memory"""
function draw_velocity_kick(xp::Vector{<:Real}, subhalo::Halo, r::Real, ::Type{T} = MM17Gamma1; nrep::Int = 1, nmem::Int = 10000) where {T<:HostModel} 

    # initialisation for a given value of r
    rs = subhalo.rs
    nstars = number_stellar_encounter(r, T)
    b_ms    = b_max(r, T) / rs
    inv_η  = _load_inverse_cdf_η(r, T)
    xt     = jacobi_radius(r, subhalo, T) / rs

    all(xp .> xt) && return false

    nxp = length(xp) # size of the point vectors we want to look at
    nmem  = (nmem ÷ nstars) * nstars # we want the memory maximal number to be a multiple of nstar
    nturn = 1      # number of iteration we must do to not overload the memory
    ndraw = nstars # number of draw at every iteraction according to the memory requirement
    

    if nstars*nrep > nmem
        nturn = (nstars*nrep)÷nmem + 1
        ndraw = nmem
    end

    v_parallel = Matrix{Float64}(undef, nxp, nrep)
    v_perp     = Matrix{Float64}(undef, nxp, nrep)

    irep = 1

    @info "nturn" nturn

    for it in 1:nturn

        # at the last step we only take the reminder number of draws necessary
        (it == nturn) && (ndraw = (nstars*nrep) % nmem)

        nchunks = (ndraw÷nstars)

        # randomly sampling the distributions
        θb = 2.0 * π * rand(ndraw)'
        β  = sqrt.(rand(ndraw))'
        η  = inv_η.(rand(ndraw))'

        # assuming b_min = 0 here

        sθb   = sin.(θb)
        y     = xp ./ (b_ms .* β) 
        denom = @. y^2 + 1 + 2 * y * sθb
        pm    = pseudo_mass.(b_ms * β, xt, subhalo.hp)

        _v_parallel = @. (pm * sθb - (y + sθb)/denom) * η / β
        _v_perp     = @. (pm  -  1/denom) * cos(θb) * η / β


        # summing all the contributions
        for j in 1:nchunks
            v_parallel[:, irep] = sum(_v_parallel[:, (j-1)*nstars+1:j*nstars], dims = 2)'
            v_perp[:, irep]     = sum(_v_perp[:, (j-1)*nstars+1:j*nstars], dims = 2)'      
            irep = irep + 1  
        end
    
    end

    return v_parallel, v_perp
end


export draw_velocity_kick_complex, VelocityKickDraw, ccdf_ΔE, ccdf_ΔE_CL


""" nmem maximal number of iteration for the memory"""
function draw_velocity_kick_complex(
    z::Vector{Complex{S}}, 
    subhalo::Halo, r_host::Real, 
    xt::Union{Real, Nothing} = nothing, 
    ::Type{T} = MM17Gamma1; 
    nrep::Int = 1, nmem::Int = 10000,
    ηb::Real = 0.0) where {T<:HostModel, S <:Real} 

    # initialisation for a given value of r
    rs = subhalo.rs
    nstars = number_stellar_encounter(r_host, T)
    β_max    = b_max(r_host, T) / rs 
    β_min    = β_max * ηb
    inv_η  = _load_inverse_cdf_η(r_host, T)

    (xt === nothing) && (xt = jacobi_radius(r_host, subhalo, T) / rs)

    all(abs.(z) .> xt) && return false
    (nmem < nstars) && return false

    nz = length(z) # size of the point vectors we want to look at
    nmem  = (nmem ÷ nstars) * nstars # we want the memory maximal number to be a multiple of nstar
    nturn = 1      # number of iteration we must do to not overload the memory
    ndraw = nstars # number of draw at every iteraction according to the memory requirement
    
    if nstars*nrep > nmem
        nturn = (nstars*nrep)÷nmem + 1
        ndraw = nmem
    end
    

    @info "nturn" nturn

    ######## initialisation
    # the idea is to make all quantities thread safe 
    # initialise the cut into chunks
    # different behaviour for the last part
    nchunks = [nmem÷nstars for i in 1:nturn]
    nchunks[nturn] = ((nstars*nrep) % nmem) ÷ nstars
    ndraws = [ndraw for i in 1:nturn]
    ndraws[nturn] = (nstars*nrep) % nmem
    
    chunks = Matrix{Union{UnitRange{Int64}, Missing}}(missing, nturn, nmem÷nstars)
    irep   = Matrix{Union{Int64, Missing}}(missing, nturn, nmem÷nstars) # index of the repetition

    for it in 1:nturn
        chunks[it, 1:nchunks[it]] .= [(j-1)*nstars+1:j*nstars for j in 1:nchunks[it]]
        irep[it, 1:nchunks[it]] .= sum(nchunks[1:it-1]) .+ [j for j in 1:nchunks[it]]
    end

    # initialisation of the result matrix
    Δw = Matrix{Complex{S}}(undef, nz, nrep)
    ###### end of initialisation

    ###### entring the main loop
    Threads.@threads for it in 1:nturn

        # randomly sampling the distributions
        β_norm = (sqrt.(rand(ndraws[it]) .* (β_max^2 - β_min^2) .+ β_min))
        β  = β_norm .* exp.(- 2.0im  * π * rand(ndraws[it]))  # b/b_max assuming b_min = 0
        η  = inv_η.(rand(ndraws[it]))
        pm = pseudo_mass.(β_norm, xt, subhalo.hp) ./ β

        δw = η .* (pm .- 1.0 ./ (z' .+ β))

        # summing all the contributions
        for j in 1:nchunks[it]
            Δw[:, irep[it, j]] .= sum(δw'[:, chunks[it, j]], dims = 2)
        end
    end
    ##### end of the main loop

    return Δw
end


function draw_velocity_kick_complex_approx(   
    x::Vector{S}, 
    subhalo::Halo, 
    r_host::Real, 
    xt::Union{Real, Nothing} = nothing, 
    ::Type{T} = MM17Gamma1; 
    nrep::Int = 1, nmem::Int = 10000,
    ηb::Real = 0.0) where {T<:HostModel, S <:Real}

     # initialisation for a given value of r_host
    rs = subhalo.rs
    nstars = number_stellar_encounter(r_host, T)
    β_max    = b_max(r_host, T) / rs 
    β_min    = β_max * ηb
    inv_η  = _load_inverse_cdf_η(r_host, T)

    (xt === nothing) && (xt = jacobi_radius(r_host, subhalo, T) / rs)

    all(abs.(x) .> xt) && return false
    (nmem < nstars) && return false

    nx = length(x) # size of the point vectors we want to look at
    nmem  = (nmem ÷ nstars) * nstars # we want the memory maximal number to be a multiple of nstar
    nturn = 1      # number of iteration we must do to not overload the memory
    ndraw = nstars # number of draw at every iteraction according to the memory requirement
    
    if nstars*nrep > nmem
        nturn = (nstars*nrep)÷nmem + 1
        ndraw = nmem
    end
    
    @info "nturn" nturn

    ######## initialisation
    # the idea is to make all quantities thread safe 
    # initialise the cut into chunks
    # different behaviour for the last part
    nchunks = [nmem÷nstars for i in 1:nturn]
    nchunks[nturn] = ((nstars*nrep) % nmem) ÷ nstars
    ndraws = [ndraw for i in 1:nturn]
    ndraws[nturn] = (nstars*nrep) % nmem
    
    chunks = Matrix{Union{UnitRange{Int64}, Missing}}(missing, nturn, nmem÷nstars)
    irep   = Matrix{Union{Int64, Missing}}(missing, nturn, nmem÷nstars) # index of the repetition

    for it in 1:nturn
        chunks[it, 1:nchunks[it]] .= [(j-1)*nstars+1:j*nstars for j in 1:nchunks[it]]
        irep[it, 1:nchunks[it]] .= sum(nchunks[1:it-1]) .+ [j for j in 1:nchunks[it]]
    end

    # initialisation of the result matrix
    Δw = Matrix{Complex{S}}(undef, nx, nrep)
    ###### end of initialisation

    ###### entring the main loop
    Threads.@threads for it in 1:nturn

        # randomly sampling the distributions
        β = (sqrt.(rand(ndraws[it]) .* (β_max^2 - β_min^2) .+ β_min)) # b/b_max assuming b_min = 0
        η  = inv_η.(rand(ndraws[it]))
        pm = pseudo_mass.(β, xt, subhalo.hp)

        δw = η ./ β .* sqrt.(pm.^2 .+ 3.0 .* (1.0 .- 2.0.*pm) ./ (3.0 .+ 2.0 .* (x'./ β).^2)) .* exp.(- 2.0im  * π * rand(ndraws[it]))

        # summing all the contributions
        for j in 1:nchunks[it]
            Δw[:, irep[it, j]] .= sum(δw'[:, chunks[it, j]], dims = 2)
        end
    end
    ##### end of the main loop

    return Δw

end

mutable struct VelocityKickDraw{S<:Real}
    subhalo::Halo
    rt::Real
    r_host::Real
    T_host_model::DataType
    x::Union{Vector{S}, StepRangeLen{S}}
    ψ::Union{Vector{S}, StepRangeLen{S}, Nothing}
    φ::Union{Vector{S}, StepRangeLen{S}, Nothing}
    Δw::Array{Complex{S}}
    δv0::Real
    ηb::Real
end


function draw_velocity_kick_complex(
    subhalo::Halo, 
    x::Union{Vector{S}, StepRangeLen{S}}, 
    ψ::Union{Vector{S}, StepRangeLen{S}, Nothing} = nothing, 
    φ::Union{Vector{S}, StepRangeLen{S}, Nothing} = nothing,
    rt::Union{Real, Nothing} = nothing,
    ::Type{T} = MM17Gamma1;
    r_host::Real = 8e-3, nrep::Int = 1, nmem::Int = 10000, 
    dflt_ψ::S = π/2.0, dflt_φ::S = 0.0, approx::Bool = false, 
    ηb::Real = 0.0) where {T<:HostModel, S <: Real} 
    

    (rt === nothing) && (rt = jacobi_radius(r_host, subhalo, T))

    if !approx 

        (ψ === nothing) && (ψ = [dflt_ψ]) 
        (φ === nothing) && (φ = [dflt_φ])

        nx = length(x) 
        nψ = length(ψ)
        nφ = length(φ)

        # converting x, ψ and φ into a complex number
        function _z_vs_coord(x::S, ψ::S, φ::S)
            (φ == 2*π) && return (1.0 + 0.0im) * x * sin(ψ)
            return x * sin(ψ) * exp(-im*φ)
        end
        
        z_array = [_z_vs_coord(_x, _ψ, _φ) for _x in x, _ψ in ψ, _φ in φ]
        linear  =  LinearIndices(z_array)
        z       = z_array[linear[:]]

        #z = [_z_vs_coord(x[i], ψ[j], φ[k]) for (i, j, k) in one_to_three_dim(nx, nψ, nφ)]

        Δw = draw_velocity_kick_complex(z, subhalo, r_host, rt/subhalo.rs , T; nrep = nrep, nmem = nmem, ηb = ηb)

        res_array = Array{Complex{S}, 4}(undef, nx, nψ, nφ, nrep)
        
        for k=1:nφ, j=1:nψ
            Threads.@threads for i=1:nx
                #res_array[i, j, k, :] = Δw[i + nx*(j-1) + nx*nψ*(k-1), :]
                res_array[i, j, k, :] = Δw[linear[i, j, k], :]
            end
        end

        (nψ == 1 && nφ != 1) && (res_array = res_array[:, 1, :, :])
        (nψ != 1 && nφ == 1) && (res_array = res_array[:, :, 1, :])
        (nψ == 1 && nφ == 1) && (res_array = res_array[:, 1, 1, :])
    else
        res_array = draw_velocity_kick_complex_approx(x, subhalo, r_host, rt/subhalo.rs , T; nrep = nrep, nmem = nmem,  ηb = ηb)
    end

    # computing the normalisation
    σ = velocity_dispersion_spherical(r_host, T)
    vstar = circular_velocity(r_host, T)
    mstar_avg = moments_C03(1)
    v_avg = 1.0/average_inverse_relative_speed(σ, vstar)

    δv0 = 2*G_NEWTON * mstar_avg * Msun / (v_avg * km /s)  / (subhalo.rs * Mpc) / (km/s) |> NoUnits

    return VelocityKickDraw(subhalo, rt, r_host, T, x, ψ, φ, res_array, δv0, ηb)
    
end


"""
function one_to_three_dim(ni::Int64, nj::Int64, nk::Int64)

    res = [(1, 1, 1)]

    for index = 2:(ni * nj * nk)
        push!(res, one_to_three_dim(index, ni, nj, nk))
    end

    return res

end

function one_to_three_dim(index::Int64, ni::Int64, nj::Int64, nk::Int64)

    ntot = ni * nj * nk 
    (index > ntot) && return false

    (index%ni != 0) && (i = index%ni)
    (index%ni == 0) && (i = ni)

    j = convert(Int64, (index - i)/ni % nj + 1)
    k = convert(Int64, (index - i - ni*(j-1))/(ni*nj) % nk + 1)
    
    return i, j, k
end
"""

# ongoing work
function pdf_dE(dE::Real, r::Real, dv::Vector{<:Real}, subhalo::Halo, ::Type{T}) where {T<:HostModel}
    rt = jacobi_radius(r * subhalo.rs, subhalo, T)
    sigma = velocity_dispersion(r, rt, subhalo)
    pref = 1.0/sqrt( 2.0 * π * sigma)
    res = sum(@. 1/dv * exp((dE - dv^2/2.0)^2/2.0/sigma^2/dv^2))
end

"""
function mean_ΔE(draw::VelocityKickDraw{<:Real}, on_angle_ψ::Bool = false)
    
    dim   = length(size(draw.Δw))
    sigma = velocity_dispersion.(draw.x * draw.subhalo.rs, draw.rt, draw.subhalo)
    Δv2   = draw.δv0^2 .* abs2.(draw.Δw)
    ΔE    = Δv2 ./ 2.0

    if on_angle_ψ
        (length(size(ΔE)) < 3) && throw(ArgumentError())

        a = selectdim((ΔE .* (sin.(draw.ψ))'), 2, 1:size-1)
        b = selectdim((ΔE .* (sin.(draw.ψ))'), 2, 2:size)

        return sum((a.+b) ./ 2.0 .* diff(draw.ψ)' , dims = 2)
        
    return ΔE
end
"""

function ccdf_ΔE(ΔE::Real, draw::VelocityKickDraw{<:Real})
    
    sigma = velocity_dispersion.(draw.x * draw.subhalo.rs, draw.rt, draw.subhalo)
    dim   = length(size(draw.Δw))
    Δv2   = (draw.δv0)^2 .* abs2.(draw.Δw) 

    return 0.5 .* (1 .+ mean(erf.((Δv2 ./ 2.0 .- ΔE) ./ (sqrt.(2.0 * Δv2) .* sigma )), dims = dim))
end

function ccdf_ΔE_CL(ΔE::Real, draw::VelocityKickDraw{<:Real})
    
    dim        = length(size(draw.Δw))
    sigma      = velocity_dispersion.(draw.x * draw.subhalo.rs, draw.rt, draw.subhalo)
    Δv2        = draw.δv0^2 .* abs2.(draw.Δw)
    ΔE_average = selectdim(mean(Δv2 ./ 2.0 , dims = dim), dim, 1)
    s          = sqrt.(ΔE_average)./(2.0.*sigma)
    ξ          = sqrt.(1.0 .+ s.^2)./s

    (ΔE  > 0) && return @. (1.0+ξ)/(2.0*ξ)*exp(-ΔE/(2.0*sigma^2)*(ξ-1.0))
    (ΔE <=0) && return @. 1.0-(ξ-1.0)(2.0*ξ) *exp(ΔE/(2.0*sigma^2)*(1.0+ξ))
end


## Need to change the conditions on n with selectdim() .= something
function _ccdf_ΔE_array(ΔE::Union{Vector{<:Real}, StepRangeLen{<:Real}}, draw::VelocityKickDraw{<:Real}, ccdf_ΔE_single::Function)

    n = length(size(draw.Δw))
    res_ΔE = Array{Float64, n}(undef, size(draw.Δw)[1:end-1]..., length(ΔE))

    for index in eachindex(ΔE)
        (n == 2) && (res_ΔE[:, index] .= ccdf_ΔE_single(ΔE[index], draw))
        (n == 3) && (res_ΔE[:, :, index] .= ccdf_ΔE_single(ΔE[index], draw))
        (n == 4) && (res_ΔE[:, :, :, index] .= ccdf_ΔE_single(ΔE[index], draw))
    end

    return res_ΔE
end

ccdf_ΔE(ΔE::Union{Vector{<:Real}, StepRangeLen{<:Real}}, draw::VelocityKickDraw{<:Real}) = _ccdf_ΔE_array(ΔE, draw, ccdf_ΔE)
ccdf_ΔE_CL(ΔE::Union{Vector{<:Real}, StepRangeLen{<:Real}}, draw::VelocityKickDraw{<:Real}) = _ccdf_ΔE_array(ΔE, draw, ccdf_ΔE_CL)




export _save_inverse_cdf_η, _load_inverse_cdf_η

function _save_inverse_cdf_η(r::Real, ::Type{T}) where {T<:HostModel}
    
    σ         = velocity_dispersion_spherical(r, T)
    vstar     = circular_velocity(r, T)
    mstar_avg = moments_C03(1)
    v_avg     = 1.0/average_inverse_relative_speed(σ, vstar)

    rnd_array = 10.0.^range(-8, -1e-12, 1000)

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
    !(file in filenames) && _save_inverse_cdf_η(r, T)
    
    data = jldopen("../cache/hosts/" * file)
    rnd_array = data["rnd"]
    inv_cdf = data["inverse_cdf_eta"]
        
    log10inv_cdf = interpolate((log10.(rnd_array),), log10.(inv_cdf),  Gridded(Linear()))
   
    function inv_cdf_η(rnd::Real) 
        try
            return 10.0^log10inv_cdf(log10(rnd)) 
        catch e
            println(rnd)
            return false
        end
    end

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




@doc raw"""
    jacobi_scale(r, ρs, hp, ρ_host, m_host)

Jacobi scale radius for a subhalo of scale density `ρs` (in Msol/Mpc^3) with a HaloProfile `hp`
at a distance r from the host centre  such that the sphericised mass density of the host at r is `ρ_host` 
and the sphericised enclosed mass inside the sphere of radius r is `m_host`.

More precisely, returns `xt` solution to
``\frac{xt^3}{\mu(xt)} - \frac{\rho_{\rm s}}{\rho_{\rm host}(r)} \frac{\hat \rho}{1-\hat \rho}``
with `` reduced sperical host density being
`` = \frac{4\pi}{3} r^3 \frac{\rho_{\rm host}(r)}{m_{\rm host}(r)}``
"""
function jacobi_scale(r::Real, ρs::Real, hp::HaloProfile, ρ_host::Real, m_host::Real)
    reduced_ρ =  4 * π * r^3 *  ρ_host / 3.0 / m_host
    to_zero(xt::Real) = xt^3/μ_halo(xt, hp) - ρs/ρ_host * reduced_ρ / (1.0 - reduced_ρ)
    return exp(Roots.find_zero(lnxt -> to_zero(exp(lnxt)), (-5, +5), Bisection())) 
end


jacobi_scale(r::Real, ρs::Real, hp::HaloProfile, ρ_host::Function, m_host::Function) = jacobi_scale(r, ρs, hp, ρ_host(r), m_host(r))
jacobi_scale(r::Real, ρs::Real, hp::HaloProfile, ::Type{T} = MM17Gamma1) where {T<:HostModel} = jacobi_scale(r, ρs, hp, r->ρ_host_spherical(r, T), r->m_host_spherical(r, T))
jacobi_scale(r::Real, subhalo::Halo{<:Real}, ::Type{T} = MM17Gamma1) where {T<:HostModel} = jacobi_scale(r, subhalo.ρs, subhalo.hp, T)
jacobi_scale(r::Real, subhalo::Halo{<:Real}, ρ_host::Real, m_host::Real) = jacobi_scale(r, subhalo.ρs, subhalo.hp,  ρ_host, m_host)
jacobi_scale(r::Real, subhalo::Halo{<:Real}, ρ_host::Function, m_host::Function) = jacobi_scale(r, subhalo.ρs, subhalo.hp,  ρ_host(r), m_host(r))

jacobi_radius(r::Real, subhalo::Halo{<:Real}, ::Type{T} = MM17Gamma1) where {T<:HostModel} = subhalo.rs * jacobi_scale(r, subhalo.ρs, subhalo.hp, T)


#######################

end # module StellarEncounters
