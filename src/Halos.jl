module Halos

include("../src/MyCosmology.jl")

using QuadGK, Roots, Unitful, HypergeometricFunctions
import Unitful: km, s, Gyr, K, Temperature, DimensionlessQuantity, Density, Length, Mass
using UnitfulAstro: Mpc, Gpc, Msun, kpc
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

import .MyCosmology: FLRWPLANCK18

abstract type HaloProfile end

struct αβγProfile{T<:Real} <: HaloProfile
    α::T
    β::T
    γ::T
end


αβγProfile(α::Real, β::Real, γ::Real) = αβγProfile(promote(α, β, γ)...)
#Halo_from_ρs_and_rs(p::HaloProfile, ρs::Density{<:Real}, rs::Length{<:Real}) = Halo(p, promote(ρs, rs)...)
#Halo_from_mΔ_and_cΔ(p::HaloProfile, mΔ::Real, rs::Real) = Halo(p, promote(ρs, rs)...)

## Definition of densities and mass profiles
NFWProfile = αβγProfile(1, 3, 1)
ρ(x::Real, p::αβγProfile = NFWProfile) = x^(-p.γ) * (1+x^p.α)^(-(p.β - p.γ)/p.α)
μ(x::Real, p::αβγProfile = NFWProfile) = HypergeometricFunctions._₂F₁((3 - p.γ)/p.α, (p.β - p.γ)/p.α, (3 + p.α - p.γ)/p.α, -x^p.α) * x^(3-p.γ) / (3-p.γ)

## Definition of relationships between concentration, mass, scale density and scale radius
function cΔ_from_ρs(ρs::Density{<:Real}, p::HaloProfile = NFWProfile, Δ::Real = 200, ρ_ref::Density{<:Real} = FLRWPLANCK18.ρ_c0 )
    g(c) = c^3 / μ(c, p) - 3 * ρs / Δ / ρ_ref
    Roots.find_zero(g, (1e-10, 1e+10), Bisection()) 
end

mΔ_from_ρs_and_rs(ρs::Density{<:Real}, rs::Length{<:Real},  p::HaloProfile = NFWProfile, Δ::Real = 200, ρ_ref::Density{<:Real} = FLRWPLANCK18.ρ_c0) = 4 * pi * ρs * rs^3 * μ(c_Δ_from_ρs(ρs, p, Δ, ρ_ref), p) |> Msun
ρs_from_cΔ(cΔ::Real, p::HaloProfile = NFWProfile , Δ::Real = 200, ρ_ref::Density{<:Real} = FLRWPLANCK18.ρ_c0, ) = Δ * ρ_ref  / 3 * cΔ^3 / μ(cΔ, p) |> Msun / kpc^3
rs_from_cΔ_and_mΔ(cΔ::Real, mΔ::Mass{<:Real}, Δ::Real = 200, ρ_ref::Density{<:Real} = FLRWPLANCK18.ρ_c0) =  (3 * mΔ / (4 * pi * Δ * ρ_ref))^(1 // 3) / cΔ |> kpc



end # module Halos
