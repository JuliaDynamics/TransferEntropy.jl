function _transferentropy(pts, vars::TEVars, binning::RectangularBinning, estimator::VisitationFrequency)

    # Collect variables for the marginals 
    C = vars.C
    XY = [vars.𝒯; vars.T; C]
    YZ = [vars.T; vars.S; C]
    Y =  [vars.T; C]
    
    # Find the bins visited by the joint system (and then get 
    # the marginal visits from that, so we don't have to encode 
    # bins multiple times). 
    joint_bin_visits = joint_visits(pts, binning)

    # Compute visitation frequencies for the nonempty bins.
    p_Y = non0hist(marginal_visits(joint_bin_visits, Y))
    p_XY = non0hist(marginal_visits(joint_bin_visits, XY))
    p_YZ = non0hist(marginal_visits(joint_bin_visits, YZ))
    p_joint = non0hist(joint_bin_visits)
    
    # Base of the logarithm
    b = estimator.b

    te = StatsBase.entropy(p_YZ, b) +
            StatsBase.entropy(p_XY, b) -
            StatsBase.entropy(p_Y, b) -
            StatsBase.entropy(p_joint, b)
end
