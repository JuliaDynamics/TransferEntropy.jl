
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
TEVars(Tf::Vector{Int}, Tpp::Vector{Int}, Spp::Vector{Int}) = TEVars(Tf, Tpp, Spp, Int[])

"""
    TEVars(;Tf = Int[], Tpp = Int[], Spp = Int[], Cpp = Int[])

Which axes of the state space correspond to the future of the target (`Tf`),
the present/past of the target (`Tpp`), the present/past of the source (`Spp`), and
the present/past of any conditioned variables (`Cpp`)? Indices correspond to variables
of the embedding or colums of a `Dataset`.

This information is used by the transfer entropy estimators to ensure
the marginal distributions are computed correctly.
"""
TEVars(;Tf::Vector{Int} = Int[],
	    Tpp::Vector{Int} = Int[],
	    Spp::Vector{Int} = Int[],
		Cpp::Vector{Int} = Int[]) =
	TEVars(Tf, Tpp, Spp, Cpp)
    
export TEVars
