using Documenter, Squiggles, CairoMakie

pages = ["Home" => "index.md",
         "Examples" => "examples.md",
         "Code design" => "code.md",
         "Correlograms, coefficients and lags" => "correlate.md",
         "Available kernels" => "kernels.md",
         "Utilities" => "utils.md",
         "API index" => "api.md"
        ]

makedocs(sitename="Squiggles.jl", pages=pages, clean=true)


deploydocs(repo="github.com:gbene/Squiggles.jl.git")
