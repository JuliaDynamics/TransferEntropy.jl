import DelayEmbeddings: Dataset
import Distances: Metric, evaluate, Euclidean, pairwise
import StaticArrays: MVector, SVector 
import DelayEmbeddings: KDTree

# Define pairwise distances methods for vectors of static vectors, vectors of vectors,
# and Datasets and Customreconstructions.
const VSV = Union{Dataset, Vector{<:SVector}, Vector{<:MVector}, Vector{Vector}}

#KDTree(pts::Vector{<:SVector}, metric::Metric = Euclidean()) = KDTree(pts, metric)

function pairwise(x::VSV, y::VSV, metric::Metric)
    length(x) == length(y) || error("Lengths of input data does not match")

    dists = zeros(eltype(eltype(x)), length(x), length(y))
    @inbounds for j in 1:length(y)
        for i in 1:length(x)
            dists[i, j] = evaluate(metric, x[i], y[j])
        end
    end
    return dists
end

function pairwise(x::VSV, metric::Metric)
    dists = zeros(eltype(eltype(x)), length(x), length(x))
    @inbounds for j in 1:length(x)
        for i in 1:length(x)
            dists[i, j] = evaluate(metric, x[i], x[j])
        end
    end
    return dists
end

function find_dists(x::VSV, y::VSV, metric::Metric)
    length(x) == length(y) || error("Lengths of input data does not match")
    dists = zeros(eltype(eltype(x)), length(x))
    
    @inbounds for i = 1:length(x)
        dists[i] = evaluate(metric, x[i], y[i])
    end
    
    return dists
end