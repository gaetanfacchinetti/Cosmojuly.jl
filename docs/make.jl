include("../src/Cosmojuly.jl")

using Documenter, .Cosmojuly

import .Cosmojuly: BkgCosmology, FLRW, Species, Neutrinos, Photons, Radiation, Matter, ColdDarkMatter
import .Cosmojuly: subhalo_mass_function_template, mass_function_merger_tree

makedocs(sitename="Cosmojuly")#, pages = [ "Page title" => "FSLModel.md", ])
deploydocs(repo = "github.com/gaetanfacchinetti/Cosmojuly.jl.git")