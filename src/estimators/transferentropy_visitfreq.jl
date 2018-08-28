"""
    transferentropy_visitfreq(
        E::AbstractEmbedding,
        ϵ::Union{Int, Float64, Vector{Float64}},
        v::TransferEntropy.TEVars) -> Float64

Using the traditional method of estimation probability
distribution by visitation frequencies [1], calculate
transfer entropy from the embedding `E`, given a
discretization scheme controlled by `\epsilon` and
information `v::TEVars`about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.
"""

function transferentropy_visitfreq(
                    E::AbstractEmbedding,
                    ϵ::Union{Int, Float64, Vector{Float64}, Vector{Int}},
                    v::TransferEntropy.TEVars)

    all_inds = unique(vcat(v.target_future,
                    v.target_presentpast,
                    v.source_presentpast,
                    v.conditioned_presentpast))

    visited_bin_inds = assign_bin_labels(E, ϵ)
    slices = groupslices(visited_bin_inds, 2)
    n_visited_bins = length(groupinds(slices))

    # The total number of variables in the embedding
    nvars = size(visited_bin_inds, 1)
    C = v.conditioned_presentpast
    XY = [v.target_future;      v.target_presentpast; C]
    YZ = [v.target_presentpast; v.source_presentpast; C]
    Y =  [v.target_presentpast;                       C]


    # How many points are there in the embedding?
    # Used to normalise probabilities when computing
    # marginals.
    npts = size(E.points, 1)

    p_Y  = marginal_visitation_freq([Y;  C], visited_bin_inds, n_visited_bins)
    p_XY = marginal_visitation_freq([XY; C], visited_bin_inds, n_visited_bins)
    p_YZ = marginal_visitation_freq([YZ; C], visited_bin_inds, n_visited_bins)
    P_joint = marginal_visitation_freq(all_inds, visited_bin_inds, n_visited_bins)
    te_estimate = ((entropy(p_YZ) +
                    entropy(p_XY) -
                    entropy(p_Y)) -
                    entropy(P_joint)) / log(2)
    te_estimate
end
