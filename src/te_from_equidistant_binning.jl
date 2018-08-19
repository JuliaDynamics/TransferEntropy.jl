"""
    TransferEntropyVariables(XY, YZ, Y, W)

"""
struct TransferEntropyVariables
    XY::AbstractArray{Int, 1} #
    YZ::AbstractArray{Int, 1} #
    Y::AbstractArray{Int, 1}  #
    W::AbstractArray{Int, 1}  # extra variables
end

"""
Returns a column vector with the same number of elements as there are unique
rows in V. The value of the ith element is the row indices of rows in V
matching the ith unique row.
"""
marginal_indices(V) = GroupSlices.groupinds(GroupSlices.groupslices(V, 1))

"""
How many times does each unique row in V appear? Returns a column vector with the same
number of elements as there are unique rows in V. The value of the ith element of the
return vector is the number of times the ith unique row of V appears in V.
"""
marginal_multiplicity(V) = [length(x) for x in marginal_indices(V)]

"""
Compute entropy of a probability distribution.
"""
function nat_entropy(prob)
    te = 0.0

    @inbounds for i = 1:size(prob, 1)
        te -= prob[i] * log(prob[i])
    end

    return te
end

"""
Compute the marginal for a binning with an associated transfer operator. The
    marginal is computed for the columns `cols`.
"""
function marginal(cols::Vector{Int},
                    eqb::StateSpaceReconstruction.RectangularBinning,
                    to::PerronFrobenius.RectangularBinningTransferOperator,
                    iv::PerronFrobenius.InvariantDistribution)

    # Loop over the positively measured bins.
    marginal_inds = marginal_indices(eqb.positive_measure_bins[iv.nonzero_inds, cols])
    marginal = zeros(Float64, size(marginal_inds, 1))

    for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(iv.dist[iv.nonzero_inds][marginal_inds[i]])
    end

    return marginal
end

function marginal(cols::Vector{Int},
                    positive_measure_bins::Array{Int, 2},
                    iv::PerronFrobenius.InvariantDistribution)

    # Loop over the positively measured bins.
    marginal_inds = marginal_indices(positive_measure_bins[:, cols])
    marginal = zeros(Float64, size(marginal_inds, 1))

    for i = 1:size(marginal_inds, 1)
        marginal[i] = sum(iv.dist[iv.nonzero_inds][marginal_inds[i]])
    end

    return marginal
end

"""
Compute transfer entropy. This is the workhorse function.
"""
function transferentropy(eqb::StateSpaceReconstruction.RectangularBinning,
            iv::PerronFrobenius.InvariantDistribution,
            vars::TransferEntropyVariables)

    positive_measure_bins = eqb.unique_nonempty_bins[iv.nonzero_inds, :]

    p_Y  = marginal([vars.Y;  vars.W], positive_measure_bins, iv)
    p_XY = marginal([vars.XY; vars.W], positive_measure_bins, iv)
    p_YZ = marginal([vars.YZ; vars.W], positive_measure_bins, iv)

    ((nat_entropy(p_YZ) +
        nat_entropy(p_XY) -
        nat_entropy(p_Y)) -
        nat_entropy(iv.dist[iv.nonzero_inds])) / log(2)
end

"""
Compute transfer entropy.
"""
function transferentropy(unique_nonempty_bins::Array{Int, 2},
            iv::PerronFrobenius.InvariantDistribution,
            vars::TransferEntropyVariables)

    positive_measure_bins = unique_nonempty_bins[iv.nonzero_inds, :]
    p_Y  = marginal([vars.Y;  vars.W], positive_measure_bins, iv)
    p_XY = marginal([vars.XY; vars.W], positive_measure_bins, iv)
    p_YZ = marginal([vars.YZ; vars.W], positive_measure_bins, iv)

    ((nat_entropy(p_YZ) +
        nat_entropy(p_XY) -
        nat_entropy(p_Y)) -
        nat_entropy(iv.dist[iv.nonzero_inds])) / log(2)
end

"""
Estimate transfer entropy from an embedding.
"""
function transferentropy(E::StateSpaceReconstruction.Embedding,
    binsizes::Vector{Int},
    vars::TransferEntropyVariables)

    n = length(binsizes)
    binnings = Vector{StateSpaceReconstruction.RectangularBinning}(n)
    transferoperators = Vector{PerronFrobenius.TransferOperator}(n)
    invariantdists = Vector{PerronFrobenius.InvariantDistribution}(n)
    transferentropies = Vector{Float64}(n)

    for i = 1:n
        binnings[i] = bin_rectangular(E, binsizes[i])
        transferoperators[i] = transferoperator(binnings[i])
        invariantdists[i] = PerronFrobenius.left_eigenvector(transferoperators[i])
    end

    for i = 1:n
        transferentropies[i] = transferentropy(binnings[i], invariantdists[i], vars)
    end

    return transferentropies
end

"""
Estimate transfer entropy from a set of precomputed binnings.
"""
function transferentropy(binnings::Vector{StateSpaceReconstruction.RectangularBinning},
                        vars::TransferEntropyVariables)

    n = length(binnings)
    transferoperators = Vector{PerronFrobenius.TransferOperator}(n)
    invariantdists = Vector{PerronFrobenius.InvariantDistribution}(n)
    transferentropies = Vector{Float64}(n)

    for i = 1:n
        transferoperators[i] = transferoperator(binnings[i])
        invariantdists[i] = PerronFrobenius.left_eigenvector(transferoperators[i])
    end

    for i = 1:n
        transferentropies[i] = transferentropy(binnings[i], invariantdists[i], vars)
    end

    return transferentropies
end

"""
Compute transfer entropy from an embedding.
"""
function transferentropy(E::StateSpaceReconstruction.Embedding,
                        binsize::Int,
                        vars::TransferEntropyVariables)
    b = bin_rectangular(E, binsize)
    to = PerronFrobenius.transferoperator(b)
    invariantdistribution = PerronFrobenius.left_eigenvector(to)
    transferentropy(b, invariantdistribution, vars)
end

"""
Compute transfer entropy from a binning.
"""
function transferentropy(b::StateSpaceReconstruction.RectangularBinning,
                        vars::TransferEntropyVariables)
    to = PerronFrobenius.transferoperator(b)
    invdist = PerronFrobenius.left_eigenvector(to)
    transferentropy(b, invdist, vars)
end

"""
Compute the transfer entropy resulting only from the geometry of the reconstructed
attractor. How? Assign uniformly distributed states on the volumes of the
reconstructed state space with nonzero measure.
"""
function shape_transferentropy(bins::Array{Int, 2}, vars::TransferEntropyVariables)
   dim, n_nonempty_bins = size(bins, 2), size(bins, 1)

   n_XY = marginal_multiplicity(bins[:, [vars.XY; vars.W]])
   n_Y  = marginal_multiplicity(bins[:, [vars.Y;  vars.W]])
   n_YZ = marginal_multiplicity(bins[:, [vars.YZ; vars.W]])

   # Transfer entropy as the sum of the marginal entropies
   ((nat_entropy(n_YZ) + nat_entropy(n_XY) - nat_entropy(n_Y)) / n_nonempty_bins) / log(2)
end

function shape_transferentropy(eqb::StateSpaceReconstruction.RectangularBinning,
                               vars::TransferEntropyVariables)

    shape_transferentropy(eqb.unique_nonempty_bins, vars)
end
