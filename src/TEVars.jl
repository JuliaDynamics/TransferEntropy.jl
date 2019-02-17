
"""
    TEVars(target_future::Vector{Int}
        target_presentpast::Vector{Int}
        source_presentpast::Vector{Int}
        conditioned_presentpast::Vector{Int})

Which axes of the state space correspond to the future of the target,
the present/past of the target, the present/past of the source, and
the present/past of any conditioned variables? Indices correspond to variables
of the embedding or colums of a `Dataset`.

This information is used by the transfer entropy estimators to ensure
the marginal distributions are computed correctly.
"""
struct TEVars
    target_future::Vector{Int}
    target_presentpast::Vector{Int}
    source_presentpast::Vector{Int}
    conditioned_presentpast::Vector{Int}
end

"""
    TEVars(target_future::Vector{Int},
            target_presentpast::Vector{Int},
            source_presentpast::Vector{Int})

Which axes of the state space correspond to the future of the target,
the present/past of the target, and the present/past of the source?
Indices correspond to variables of the embedding or colums of a `Dataset`.

This information is used by the transfer entropy estimators to ensure
the marginal distributions are computed correctly.

This three-argument constructor assumes there will be no conditional variables.
"""
TEVars(target_future::Vector{Int},
    target_presentpast::Vector{Int},
    source_presentpast::Vector{Int}) =
TEVars(target_future, target_presentpast, source_presentpast, Int[])

"""
    TEVars(;target_future = Int[],
            target_presentpast = Int[],
            source_presentpast = Int[],
            conditioned_presentpast = Int[])

Which axes of the state space correspond to the future of the target,
the present/past of the target, the present/past of the source, and
the present/past of any conditioned variables? Indices correspond to variables
of the embedding or colums of a `Dataset`.

This information is used by the transfer entropy estimators to ensure
the marginal distributions are computed correctly.
"""
TEVars(;target_future::Vector{Int} = Int[],
	    target_presentpast::Vector{Int} = Int[],
	    source_presentpast::Vector{Int} = Int[],
		conditioned_presentpast::Vector{Int} = Int[]) =
	TEVars(target_future, target_presentpast, source_presentpast,
        conditioned_presentpast)

export TEVars
