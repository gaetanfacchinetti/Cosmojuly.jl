##################################################################################
# This file is part of Cosmojuly.
#
# Copyright (c) 2023, Gaétan Facchinetti
#
# Cosmojuly is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or any 
# later version. 21cmCAST is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU 
# General Public License along with 21cmCAST. 
# If not, see <https://www.gnu.org/licenses/>.
#
##################################################################################

module Halos

include("../src/BackgroundCosmo.jl")
include("../src/PowerSpectrum.jl")

using QuadGK, Roots, Unitful, HypergeometricFunctions
import Unitful: km, s, Gyr, K 
using UnitfulAstro: Mpc, Gpc, Msun, kpc
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

import Main.Cosmojuly.PowerSpectrum: Cosmology, planck18
import Main.Cosmojuly.BackgroundCosmo: planck18_bkg

export Halo, NFWProfile, αβγProfile, HaloProfile
export halo_from_ρs_and_rs, halo_from_mΔ_and_cΔ
export mΔ_from_ρs_and_rs, mΔ, rΔ_from_ρs_and_rs, rΔ, ρ_halo, μ_halo, m_halo

abstract type HaloProfile{T<:Real} end

################################################
# Dimensionless Halo

struct αβγProfile{T<:Real} <: HaloProfile{T}
    α::T
    β::T
    γ::T
end

αβγProfile(α::Real, β::Real, γ::Real) = αβγProfile(promote(α, β, γ)...)

## Definition of densities and mass profiles
NFWProfile = αβγProfile(1, 3, 1)
ρ_halo(x::Real, p::αβγProfile = NFWProfile) = x^(-p.γ) * (1+x^p.α)^(-(p.β - p.γ)/p.α)
ρ_halo(x::Vector{<:Real}, p::αβγProfile = NFWProfile) = x.^(-p.γ) .* (1 .+ x.^p.α).^(-(p.β - p.γ)/p.α)
μ_halo(x::Real, p::αβγProfile = NFWProfile) = HypergeometricFunctions._₂F₁((3 - p.γ)/p.α, (p.β - p.γ)/p.α, (3 + p.α - p.γ)/p.α, -x^p.α) * x^(3-p.γ) / (3-p.γ)
μ_halo(x::Vector{<:Real}, p::αβγProfile = NFWProfile) = HypergeometricFunctions._₂F₁.((3 - p.γ)/p.α, (p.β - p.γ)/p.α, (3 + p.α - p.γ)/p.α, -x.^p.α) .* x.^(3-p.γ) ./ (3-p.γ)


################################################
#
# Dimensionfull Halo

# Whenever possible do all the computation using the dimensionless profile
# Otherwise unit conversion slightly slow down the code

## Definition of relationships between concentration, mass, scale density and scale radius
function cΔ_from_ρs(ρs::Real, hp::HaloProfile = NFWProfile, Δ::Real = 200, ρ_ref::Real = planck18_bkg.ρ_c0)
    g(c) = c^3 / μ(c, hp) - 3 * ρs / Δ / ρ_ref
    Roots.find_zero(g, (1e-10, 1e+10), Bisection()) 
end

mΔ_from_ρs_and_rs(ρs::Real, rs::Real, hp::HaloProfile = NFWProfile, Δ::Real = 200, ρ_ref::Real = planck18_bkg.ρ_c0) = 4 * pi * ρs * rs^3 * μ_halo(cΔ_from_ρs(ρs, hp, Δ, ρ_ref), hp)
ρs_from_cΔ(cΔ::Real, hp::HaloProfile = NFWProfile , Δ::Real = 200, ρ_ref::Real = planck18_bkg.ρ_c0) = Δ * ρ_ref  / 3 * cΔ^3 / μ_halo(cΔ, hp) 
rs_from_cΔ_and_mΔ(cΔ::Real, mΔ::Real, Δ::Real = 200, ρ_ref::Real = planck18_bkg.ρ_c0) =  (3 * mΔ / (4 * pi * Δ * ρ_ref))^(1 // 3) / cΔ 
rΔ_from_ρs_and_rs(ρs::Real, rs::Real, hp::HaloProfile = NFWProfile, Δ::Real = 200, ρ_ref::Real = planck18_bkg.ρ_c0) = (3 * mΔ_from_ρs_and_rs(ρs, rs, hp, Δ, ρ_ref) / (4*π*Δ*ρ_ref))^(1//3)

struct Halo{T<:Real}
    hp::HaloProfile
    ρs::T
    rs::T
end

halo_from_ρs_and_rs(hp::HaloProfile, ρs::Real, rs::Real) = Halo(hp, promote(ρs, rs)...)

function halo_from_mΔ_and_cΔ(hp::HaloProfile, mΔ::Real, cΔ::Real;  Δ::Real = 200, ρ_ref::Real = planck18_bkg.ρ_c0)
    ρs = ρs_from_cΔ(cΔ, hp, Δ, ρ_ref)
    rs = rs_from_cΔ_and_mΔ(cΔ, mΔ, Δ, ρ_ref)
    return Halo(hp, promote(ρs, rs)...)
end

ρ_halo(r::Real, h::Halo{<:Real}) = h.ρs * ρ_halo(r/h.rs, h.hp)
ρ_halo(r::Vector{<:Real}, h::Halo{<:Real}) = h.ρs * ρ_halo(r./h.rs, h.hp)
m_halo(r::Real, h::Halo{<:Real}) = 4.0 * π * h.ρs * h.rs^3 * μ_halo(r/h.rs, h.hp)
m_halo(r::Vector{<:Real}, h::Halo{<:Real}) = 4.0 * π * h.ρs * h.rs^3 * μ_halo(r./h.rs, h.hp)
mΔ(h::Halo{<:Real}, Δ::Real = 200, ρ_ref::Real = planck18_bkg.ρ_c0) = mΔ_from_ρs_and_rs(h.ρs, h.rs, h.hp, Δ, ρ_ref)
rΔ(h::Halo{<:Real}, Δ::Real = 200, ρ_ref::Real = planck18_bkg.ρ_c0) = rΔ_from_ρs_and_rs(h.ρs, h.rs, h.hp, Δ, ρ_ref)

ρs(h::Halo) = h.ρs * ρ_0
rs(h::Halo) = h.rs * r_0




end # module Halos
