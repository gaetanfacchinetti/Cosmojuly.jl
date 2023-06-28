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

include("../src/MyCosmology.jl")

using QuadGK, Roots, Unitful, HypergeometricFunctions
import Unitful: km, s, Gyr, K 
using UnitfulAstro: Mpc, Gpc, Msun, kpc
import PhysicalConstants.CODATA2018: c_0, G as G_NEWTON

import .MyCosmology: FLRWPLANCK18

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
ρ(x::Real, p::αβγProfile = NFWProfile) = x^(-p.γ) * (1+x^p.α)^(-(p.β - p.γ)/p.α)
μ(x::Real, p::αβγProfile = NFWProfile) = HypergeometricFunctions._₂F₁((3 - p.γ)/p.α, (p.β - p.γ)/p.α, (3 + p.α - p.γ)/p.α, -x^p.α) * x^(3-p.γ) / (3-p.γ)


################################################
#
# Dimensionfull Halo

# Whenever possible do all the computation using the dimensionless profile
# Otherwise unit conversion slightly slow down the code

## Definition of relationships between concentration, mass, scale density and scale radius
function cΔ_from_ρs(ρs::Density{<:Real}, p::HaloProfile = NFWProfile, Δ::Real = 200, ρ_ref::Density{<:Real} = FLRWPLANCK18.ρ_c0)
    g(c) = c^3 / μ(c, p) - 3 * ρs / Δ / ρ_ref
    Roots.find_zero(g, (1e-10, 1e+10), Bisection()) 
end

mΔ_from_ρs_and_rs(ρs::Real, rs::Real,  p::HaloProfile = NFWProfile, Δ::Real = 200, ρ_ref::Real = FLRWPLANCK18.ρ_c0) = 4 * pi * ρs * rs^3 * μ(c_Δ_from_ρs(ρs, p, Δ, ρ_ref), p)
ρs_from_cΔ(cΔ::Real, p::HaloProfile = NFWProfile , Δ::Real = 200, ρ_ref::Real = FLRWPLANCK18.ρ_c0) = Δ * ρ_ref  / 3 * cΔ^3 / μ(cΔ, p) 
rs_from_cΔ_and_mΔ(cΔ::Real, mΔ::Real, Δ::Real = 200, ρ_ref::Real = FLRWPLANCK18.ρ_c0) =  (3 * mΔ / (4 * pi * Δ * ρ_ref))^(1 // 3) / cΔ 

 
struct Halo{T<:Real}
    p::HaloProfile
    ρs::T
    rs::T
end

ρ_0 = Msun / kpc^3
r_0 = kpc
m_0 = Msun

Halo_from_ρs_and_rs(p::HaloProfile, ρs::Density{<:Real}, rs::Length{<:Real}) = Halo(p, promote(ρs/ρ_0, rs/r_0)...)

function Halo_from_mΔ_and_cΔ(p::HaloProfile, mΔ::Mass{<:Real}, cΔ::Real;  Δ::Real = 200, ρ_ref::Density{<:Real} = FLRWPLANCK18.ρ_c0)
    ρs = ρs_from_cΔ(cΔ, p, Δ, ρ_ref)/ρ_0
    rs = rs_from_cΔ_and_mΔ(cΔ, mΔ, Δ, ρ_ref)/r_0
    return Halo(p, promote(ρs, rs)...)
end

ρ(r::Real, h::Halo{<:Real}) = h.ρs * ρ(r/h.rs, p)
m(r::Real, h::Halo{<:Real}) = 4. * pi * h.ρs * h.rs^3 * μ(r/h.rs / r_0, h.p) * m_0

ρs(h::Halo) = h.ρs * ρ_0
rs(h::Halo) = h.rs * r_0


end # module Halos
