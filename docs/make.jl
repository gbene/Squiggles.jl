using Documenter, Squiggles, CairoMakie

pages = ["Home" => "index.md",
         "Code design" => "code.md",
         "Correlograms, coefficients and lags" => "correlate.md",
         "Available kernels" => "kernels.md",
         "Utilities" => "utils.md",
         "Plotting" => "plotters.md"
        ]

makedocs(sitename="Squiggles.jl", pages=pages, clean=true)


deploydocs(repo="github.com:gbene/Fractalizer.jl.git",devbranch="dev")
