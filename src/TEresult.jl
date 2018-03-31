using Parameters

@with_kw struct TEresult
    embedding::Array{Float64, 2} = Array{Float64}()
    lag::Int
    triangulation::Triangulation
    markovmatrix::Array{Float64, 2} = Array{Float64, 2}()
    invmeasure::InvariantDistribution.InvDist = InvariantDistribution.InvDist()
    binsizes::Vector{Float64} = Vector{Float64}()
    TE::Array{Float64, 2} =  Array{Float64, 2}()
end

"""
Convert a TEresult to a dictionary. Used when saving to .jls files, which requires
a dictionary.
"""
todict(result::TEresult) = Dict([fn => getfield(result, fn) for fn = fieldnames(result)])

export todict
