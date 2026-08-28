using Documenter
using FluxKAN

DocMeta.setdocmeta!(FluxKAN, :DocTestSetup, :(using FluxKAN); recursive=true)

const DOCS_ROOT = @__DIR__
const DOCS_BUILD = joinpath(DOCS_ROOT, "..", "docs-build")

makedocs(
    root=DOCS_ROOT,
    sitename="FluxKAN.jl",
    modules=[FluxKAN],
    source=".",
    build=DOCS_BUILD,
    clean=true,
    doctest=true,
    checkdocs=:exports,
    format=Documenter.HTML(
        canonical="https://cometscome.github.io/FluxKAN.jl",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "API contract" => "api.md",
        "Migration from 0.1" => "migration.md",
        "Research comparison" => "research-comparison.md",
        "Performance" => "performance.md",
        "Release checklist" => "release-checklist.md",
        "1.0 scope" => "release-scope.md",
    ],
)

deploydocs(
    root=DOCS_ROOT,
    target=DOCS_BUILD,
    repo="github.com/cometscome/FluxKAN.jl.git",
    push_preview=false,
)
