using DelayEmbeddings, Statistics

"""
    construct_candidate_variables(
        source::Vector{AbstractVector}, 
        target::Vector{AbstractVector}, 
        [cond::Vector{AbstractVector}];
        k::Int = 1, include_instantaneous = true,
        τexclude::Union{Int, Nothing} = nothing,
        maxlag::Union{Int, Float64} = 0.05
    ) → ([τs_source, τs_target, τs_cond, ks_targetfuture], [js_source, js_target, js_cond, js_targetfuture])

Construct candidate variables from input time series. `source` is a vector of equal-length time series
assumed to represent the putative source process. `target` and `cond` are the same, but contains time series
of the target process and of the conditional processes, respectively. `k` is the desired prediction lag. 

If `include_instantaneous == true`, then the analysis will also consider instantaneous interactions between
the variables.

If `maxlag` is an integer, `maxlag` is taken as the maximum allowed embedding lag. If `maxlag` is a float, 
then the maximum embedding lag is taken as `maximum([length.(source); length.(target); length.(cond)])*maxlag`.

If `τexclude` is an integer, all variables whose embedding lag has absolute value equal to `exclude` will be 
excluded.
"""
function construct_candidate_variables(source, target, cond;
        k::Int = 1,
        τexclude::Union{Int, Nothing} = nothing,
        include_instantaneous = true,
        method_delay = "ac_min",
        maxlag::Union{Int, Float64} = 0.05)
    
    # Ensure all time series are of the same length.
    Ls = [length.(source); length.(target); length.(cond)]
    @assert all(Ls .== maximum(Ls))
    
    if maxlag isa Int
        τs = 1:maxlag
    else
        τs = 1:ceil(Int, maximum(Ls)*maxlag)
    end
    
    # Find the maximum allowed embedding lag for each of the candidates.
    τsmax_source = [estimate_delay(s, method_delay, τs) for s in source]
    τsmax_target = [estimate_delay(t, method_delay, τs) for t in target]
    τsmax_cond = [estimate_delay(c, method_delay, τs) for c in cond]

    # Generate candidate set
    startlag = include_instantaneous ? 0 : -1

    τs_source = [[startlag:-1:-τ...,] for τ in τsmax_source]
    τs_target = [[startlag:-1:-τ...,] for τ in τsmax_target]
    τs_cond = [[startlag:-1:-τ...,] for τ in τsmax_cond]
    
    ks_targetfuture = [k for i in 1:length(target)]
    js_targetfuture = [i for i in length(τs_source)+1:length(τs_source)+length(τs_target)]
    τs = [τs_source..., τs_target..., τs_cond...]
    js = [[i for x in 1:length(τs[i])] for i = 1:length(τs)]

    # Variable filtering, if desired
    if τexclude isa Int
        τs = [filtered_τs(τsᵢ, jsᵢ, τexclude) for (τsᵢ, jsᵢ) in zip(τs, js)]
        js = [filtered_js(τsᵢ, jsᵢ, τexclude) for (τsᵢ, jsᵢ) in zip(τs, js)]
    end
    return [τs..., ks_targetfuture], [js..., js_targetfuture]
end

# Usaully, we use all lags from startlag:-\tau_max to construct variables. In some situations,
# we may want to exclude som of those variables. 
function filtered_τs(τs::AbstractVector{Int}, js::AbstractVector{Int}, τexclude::Int)
    [τ for τ in τs if abs(τ) != abs.(τexclude)]
end

function filtered_js(τs::AbstractVector{Int}, js::AbstractVector{Int}, τexclude::Int)
    [j for (τ, j) in zip(τs, js) if abs(τ) != abs.(τexclude)]
end

# source & target variant 
function construct_candidate_variables(source, target; 
        k::Int = 1, 
        τexclude::Union{Int, Nothing} = nothing,
        include_instantaneous = true,
        method_delay = "ac_min",
        maxlag::Union{Int, Float64} = 0.05)
    
    # Ensure all time series are of the same length.
    Ls = [length.(source); length.(target)]
    @assert all(Ls .== maximum(Ls))
    
    if maxlag isa Int
        τs = 1:maxlag
    else
        τs = 1:ceil(Int, maximum(Ls)*maxlag)
    end
    
    # Find the maximum allowed embedding lag for each of the candidates.
    τsmax_source = [estimate_delay(s, method_delay, τs) for s in source]
    τsmax_target = [estimate_delay(t, method_delay, τs) for t in target]

    # Generate candidate set
    startlag = include_instantaneous ? 0 : -1
    τs_source = [[startlag:-1:-τ...,] for τ in τsmax_source]
    τs_target = [[startlag:-1:-τ...,] for τ in τsmax_target]
    
    ks_targetfuture = [k for i in 1:length(target)]
    js_targetfuture = [i for i in length(τs_source)+1:length(τs_source)+length(τs_target)]
    τs = [τs_source..., τs_target...,]
    js = [[i for x in 1:length(τs[i])] for i = 1:length(τs)]

    # Variable filtering, if desired
    if τexclude isa Int
        τs = [filtered_τs(τsᵢ, jsᵢ, τexclude) for (τsᵢ, jsᵢ) in zip(τs, js)]
        js = [filtered_js(τsᵢ, jsᵢ, τexclude) for (τsᵢ, jsᵢ) in zip(τs, js)]
    end
        
    return [τs..., ks_targetfuture], [js..., js_targetfuture]
end


# source, target & cond variant
function embed_candidate_variables(source, target, cond;
        η::Int = 1, 
        τexclude::Union{Int, Nothing} = nothing,
        include_instantaneous = true,
        method_delay = "mi_min",
        maxlag::Union{Int, Float64} = 0.05)
    
    τs, js = construct_candidate_variables(source, target, cond, k = η, τexclude = τexclude)

    # TODO: This is more efficient if not using datasets. Re-do manually.
    data = Dataset([source..., target..., cond...,]...,)
    ℰ = genembed(data, ((τs...)...,), ((js...)...,))
    
    # Get all variables except the target future (which are the last columns of ℰ)
    n_timeseries = size(ℰ, 2)
    n_timeseries_target = length(target)
    Ω = [ℰ[:, i] for i = 1:n_timeseries - n_timeseries_target]
    Y⁺ = ℰ[:, n_timeseries - n_timeseries_target+1:end]

    # We need to keep track of which variables are from the source, because 
    # when computing the final TE, we need a marginal which is 𝒮 \ 𝒮_source.
    # Hence, we need to know which indices in `js` correspond to the source.
    idxs_source = 1:length(source)
    idxs_target = length(source)+1:length(source)+length(target)
    idxs_cond = length(source)+length(target)+1:length(source)+length(target)+length(cond)

    return Ω, Y⁺, τs, js, idxs_source, idxs_target, idxs_cond
end

# source & target variant
function embed_candidate_variables(source, target; 
        η::Int = 1, 
        τexclude::Union{Int, Nothing} = nothing,
        include_instantaneous = true,
        method_delay = "mi_min",
        maxlag::Union{Int, Float64} = 0.05)
    
    τs, js = construct_candidate_variables(source, target, k = η, τexclude = τexclude)
    
    # TODO: This is more efficient if not using datasets. Re-do manually.
    data = Dataset([source..., target...,]...,)
    ℰ = genembed(data, ((τs...)...,), ((js...)...,))
    
    # Get all variables except the target future (which are the last columns of ℰ)
    n_timeseries = size(ℰ, 2)
    n_timeseries_target = length(target)
    Ω = [ℰ[:, i] for i = 1:n_timeseries - n_timeseries_target]
    
    Y⁺ = ℰ[:, n_timeseries - n_timeseries_target+1:end]
    idxs_source = 1:length(source)
    idxs_target = length(source)+1:length(source)+length(target)
    idxs_cond = Int[]

    return Ω, Y⁺, τs, js, idxs_source, idxs_target, idxs_cond
end

function optim_te(Ω, Y⁺, τs, js, idxs_source, idxs_target, idxs_cond, est; 
        uq = 0.95, nsurr = 100, q = 1, base = 2)
    
    τs_comb = [(τs...)...,]
    js_comb = [(js...)...,]
    
    npts = length(Y⁺)
    n_candidate_variables = length(Ω)
    
    𝒮 = Vector{Vector{Float64}}(undef, 0)
    𝒮_τs = Vector{Int}(undef, 0)
    𝒮_js = Vector{Int}(undef, 0)
    
    k = 1
    while k <= n_candidate_variables
        n_remaining_candidates = length(Ω)
        CMIs_between_Y⁺_and_candidates = zeros(n_remaining_candidates)
        
        # At first iteration, only loop through source variable. If no source variable is found that 
        # yields significant TE, terminate.
        for i = 1:n_remaining_candidates
            if k == 1 || length(𝒮) == 0
                Cᵢ = Ω[i]
                CMI_Y⁺_Cᵢ = 
                    genentropy(Dataset(Y⁺, Dataset(Cᵢ)), est, q = q, base = base) - 
                    genentropy(Dataset(Cᵢ), est, q = q, base = base)
            else
                Cᵢ = [Ω[i], 𝒮...]
                CMI_Y⁺_Cᵢ = 
                    genentropy(Dataset(Y⁺, Dataset(Cᵢ...,)), est, q = q, base = base) - 
                    genentropy(Dataset(Cᵢ...,), est, q = q, base = base)
            end
            CMIs_between_Y⁺_and_candidates[i] = CMI_Y⁺_Cᵢ
        end
        
        idx = findfirst(x -> x == minimum(CMIs_between_Y⁺_and_candidates), CMIs_between_Y⁺_and_candidates)
        Wₖ = Ω[idx]
                
        # Test significance of this candidate by using a random permutation test
        CMI_permutations = zeros(nsurr)
        
        # A circular shift surrogate generator, to exclude effects of autocorrelation
        s = surrogenerator(Wₖ, CircShift(collect(1:npts - 1)))
        #s = surrogenerator(Wₖ, RandomShuffle())
        
        if k == 1
            cmiₖ = CMIs_between_Y⁺_and_candidates[idx]

            for i = 1:nsurr
                surr_wₖ = s() # Surrogate version of Wₖ
                CMI_permutations[i] = mutualinfo(Y⁺, surr_wₖ, est)
            end
        else
            # Precompute terms that do not change during surrogate loop
            H_Y⁺_𝒮 = genentropy(Dataset(Y⁺, Dataset(𝒮...,)), est, q = q, base = base)
            
            # ORIGIANL TE
            H_𝒮 = genentropy(Dataset(𝒮...), est, q = q, base = base)
            cmiₖ = H_Y⁺_𝒮 + 
                    genentropy(Dataset([Wₖ, 𝒮...,]...,), est, q = q, base = base) - 
                    genentropy(Dataset(Y⁺, Dataset([Wₖ, 𝒮...,]...,)), est, q = q, base = base) - 
                    H_𝒮

            for i = 1:nsurr
                surr_wₖ = s() # Surrogate version of Wₖ
                CMI_permutations[i] = H_Y⁺_𝒮 + 
                    genentropy(Dataset([surr_wₖ, 𝒮...]...,), est, q = q, base = base) - 
                    genentropy(Dataset(Y⁺, Dataset([surr_wₖ, 𝒮...]...,)), est, q = q, base = base) - 
                    H_𝒮
            end
            
        end
       # If the candidate passes the significance test
        if cmiₖ > quantile(CMI_permutations, uq)
            # Add the candidate to list of selected candidates
            push!(𝒮, Wₖ)
            push!(𝒮_τs, τs_comb[idx])
            push!(𝒮_js, js_comb[idx])
            
            # Delete the candidate from the list of remaining candidates
            deleteat!(Ω, idx)
            deleteat!(τs_comb, idx)
            deleteat!(js_comb, idx)

            k = k + 1
        else 
            k = n_candidate_variables + 1
        end
    end
    
    
    # No variables were selected
    if length(𝒮) == 0
        return 0.0, Int[], Int[], idxs_source, idxs_target, idxs_cond
    end
    
    # No variables were selected from the source process
    n_source_vars_picked = count(x -> x ∈ idxs_source, 𝒮_js)
    if n_source_vars_picked == 0
        return 0.0, Int[], Int[], idxs_source, idxs_target, idxs_cond
    end
    
    # No variables were selected from the target or conditional processes.
    𝒮_nonX = [ts for (ts, j) in zip(𝒮, 𝒮_js) if j ∉ idxs_source]
    if length(𝒮_nonX) == 0
        return 0.0, Int[], Int[], idxs_source, idxs_target, idxs_cond
    end
        
    CE2 = genentropy(Dataset(Y⁺, Dataset(𝒮...,)), est, base = base, q = q) - 
        genentropy(Dataset(𝒮...,), est, base = base, q = q)
    
    CE1 = genentropy(Dataset(Y⁺, Dataset(𝒮_nonX...,)), est, base = base, q = q) - 
        genentropy(Dataset(𝒮_nonX...,), est, base = base, q = q)
    
    CMI = CE1 - CE2
    return CMI, 𝒮_js, 𝒮_τs, idxs_source, idxs_target, idxs_cond
    
end

process_input(ts::Vector{T}) where T <: Real = [ts]
process_input(ts::AbstractVector{Vector{T}}) where T <: Real = ts
