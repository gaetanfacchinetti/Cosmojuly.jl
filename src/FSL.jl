
module FSLModel

#include("./FSL.jl")
#import Main.MyCosmology: Cosmology, z_eq_mr, k_eq_mr_Mpc, planck18

function mass_function_MergerTree(mΔ_sub::Real, mΔ_host::Real) 
    
    γ1 = 0.019
    α1 = -0.94
    γ2 = 0.464
    α2 = -0.58
    β = 24.0
    ζ = 3.4

    x = mΔ_sub / mΔ_host

    return (γ1*x^α1 + γ2*x^α2) * exp(-β * x^ζ )/mΔ_sub
end

function mass_function_PowerLaw(mΔ_sub::Real, α::Real)
    return mΔ_sub^(-α)
end

function pdf_concentration(cΔ::Real, mΔ::Real) ::Real
   
    # Definition of properties from Sanchez-Conde & Prada relation
    σ_c = 0.14 * log(10.0);
    median_c = c_bar(mΔ);

    Kc = 0.5 * erfc(-log(median_c) / (sqrt(2.0) * σ_c));

    return 1.0 / Kc / c200 / sqrt(2.0 * π) / σ_c * exp(-(log(cΔ) - log(median_c)) / sqrt(2.0) / σ_c^2)

end

function median_concentration_SCP12(mΔ::Real, h::Real)::Real
#m200 in Msol
    cn::Vector = [37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7]

    mΔ_min = mΔ;

    if (mΔ <= 7.24e-10)
        mΔ_min = 7.24e-10;

    return sum(cn .* log(mΔ_min * h).^ (0:5))
end


function pdf_virial_mass(mΔ::Real)

end


end # module FSL
