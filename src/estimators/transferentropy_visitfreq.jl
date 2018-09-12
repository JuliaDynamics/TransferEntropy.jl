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

	# Use base 2 for the entropy, so that we get transfer entropy in bits
	te = ((entropy(p_YZ, base = 2) +
                    entropy(p_XY, base = 2) -
                    entropy(p_Y), base = 2) -
                    entropy(P_joint, base = 2))
end


""" Compute transfer entropy over a range of bin sizes. """
function transferentropy_visitfreq(E::AbstractEmbedding,
        ϵ::Vector{Union{Int, Float64, Vector{Float64}, Vector{Int}}},
        v::TEVars)
    map(ϵᵢ -> transferentropy_visitfreq(E, ϵᵢ, v), ϵ)
end

tefreq = transferentropy_visitfreq
