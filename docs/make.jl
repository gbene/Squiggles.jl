using Documenter, Fractalizer, CairoMakie

pages = ["Home" => "index.md"]

makedocs(sitename="Squiggles.jl", pages=pages, clean=true)


deploydocs(repo="github.com:gbene/Fractalizer.jl.git",devbranch="dev")
