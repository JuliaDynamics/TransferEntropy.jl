
"""
transferentropy(pts, ϵ, vars::TEVars, estimator::VisitationFrequency; b = 2)

Compute transfer entropy for a set of ordered points representing
an appropriate embedding of some time series. See documentation for 
`TEVars` for info on how to specify the marginals (i.e. which variables 
of the embedding are treated as what). 

`b` sets the base of the logarithm (e.g `b = 2` gives the transfer 
entropy in bits). 
"""
function transferentropy(pts, ϵ, vars::TEVars, estimator::VisitationFrequency; b = 2)

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