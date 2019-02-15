import StateSpaceReconstruction:
    cembed,
    assign_bin_labels,
    groupslices, groupinds,
    marginal_visitation_freq

import PerronFrobenius:
        InvariantDistribution

"""
    transferentropy_visitfreq(
        E::AbstractEmbedding,
        ϵ::Union{Int, Float64, Vector{Float64}},
        v::TransferEntropy.TEVars,
        normalise_to_tPP) -> Float64

Using the traditional method of estimation probability
distribution by visitation frequencies [1], calculate
transfer entropy from the embedding `E`, given a
discretization scheme controlled by `\epsilon` and
information `v::TEVars`about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.


If `normalise_to_tPP = true`, then the TE estimate is normalised to
the entropy rate of the target variable, `H(target_future | target_presentpast)`.
"""

function transferentropy_visitfreq(
                    E::Embeddings.AbstractEmbedding,
                    ϵ::Union{Int, Float64, Vector{Float64}, Vector{Int}},
                    v::TEVars,
                    normalise_to_tPP = false)

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

    p_Y  = marginal_visitation_freq(Y, visited_bin_inds, n_visited_bins)
    p_XY = marginal_visitation_freq(XY, visited_bin_inds, n_visited_bins)
    p_YZ = marginal_visitation_freq(YZ, visited_bin_inds, n_visited_bins)
    P_joint = marginal_visitation_freq(all_inds, visited_bin_inds, n_visited_bins)

	# Use base 2 for the entropy, so that we get transfer entropy in bits
    if normalise_to_tPP
        te = entropy(p_YZ, b = 2) +
            entropy(p_XY, b = 2) -
            entropy(p_Y, b = 2) -
            entropy(P_joint, b = 2)
		return te
        # Normalise to the history of the target to isolate the effect of
        # the source. To do this, we need to compute
        # H(target_future | target_presentpast) =
        #   H(target_future, target_presentpast) - H(target_presentpast) =

        #tF_tPP = [v.target_future; v.target_presentpast; C]
        #tPP = [v.target_presentpast; C]

        #P_tF_tPP = marginal_visitation_freq(tF_tPP, visited_bin_inds, n_visited_bins)
        #P_tPP = marginal_visitation_freq(tPP, visited_bin_inds, n_visited_bins)

        #te = te / (entropy(P_tF_tPP, b = 2) - entropy(P_tPP, b = 2))
    else
        te = entropy(p_YZ, b = 2) +
            entropy(p_XY, b = 2) -
            entropy(p_Y, b = 2) -
            entropy(P_joint, b = 2)
    end

    return te
end


"""
Compute transfer entropy over a range of bin sizes.

Using the traditional method of estimation probability
distribution by visitation frequencies [1], calculate
transfer entropy from the embedding `E`, given a
discretization scheme controlled by the `\epsilon`s and
information `v::TEVars`about which columns of the embedding to
consider for each of the marginal distributions. From these
marginal distributions, we calculate marginal entropies and
insert these into the transfer entropy expression.

If `normalise_to_tPP = true`, then the TE estimate is normalised to
the entropy rate of the target variable, `H(target_future | target_presentpast)`.
"""
function transferentropy_visitfreq(E::Embeddings.AbstractEmbedding,
        ϵ::Vector{Union{Int, Float64, Vector{Float64}, Vector{Int}}},
        v::TEVars,
        normalise_to_tPP = false)
    map(ϵᵢ -> transferentropy_visitfreq(E, ϵᵢ, v, normalise_to_tPP), ϵ)
end

"""
    transferentropy_visitfreq(pts, ϵ, vars::TEVars; b = 2)

Compute transfer entropy for a set of ordered points representing
an appropriate embedding of some time series. See documentation for 
`TEVars` for info on how to specify the marginals (i.e. which variables 
of the embedding are treated as what). 

`b` sets the base of the logarithm (e.g `b = 2` gives the transfer 
entropy in bits). 
"""
function transferentropy_visitfreq(pts, ϵ, vars::TEVars; b = 2)
    
    # Collect variables for the marginals 
    C = vars.conditioned_presentpast
    XY = [vars.target_future;      vars.target_presentpast; C]
    YZ = [vars.target_presentpast; vars.source_presentpast; C]
    Y =  [vars.target_presentpast;                          C]
    
    # Find the bins visited by the joint system (and then get 
    # the marginal visits from that, so we don't have to encode 
    # bins multiple times). 
    joint_bin_visits = joint_visits(pts, ϵ)

    # Compute visitation frequencies for nonempty bi
    p_Y = non0hist(marginal_visits(joint_bin_visits, Y))
    p_XY = non0hist(marginal_visits(joint_bin_visits, XY))
    p_YZ = non0hist(marginal_visits(joint_bin_visits, YZ))
    p_joint = non0hist(joint_bin_visits)
    
    te = StatsBase.entropy(p_YZ, b) +
            StatsBase.entropy(p_XY, b) -
            StatsBase.entropy(p_Y, b) -
            StatsBase.entropy(p_joint, b)
end

tefreq = transferentropy_visitfreq


export
transferentropy_visitfreq,
tefreq
