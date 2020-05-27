
"""
    TEVars(𝒯::Vector{Int}, T::Vector{Int}, S::Vector{Int})
    TEVars(𝒯::Vector{Int}, T::Vector{Int}, S::Vector{Int}, C::Vector{Int})
    TEVars(;𝒯 = Int[], T = Int[], S = Int[], C = Int[]) -> TEVars

Which axes of the state space correspond to the future of the target (`𝒯`),
the present/past of the target (`T`), the present/past of the source (`S`), and
the present/past of any conditioned variables (`C`)?  This information is used by 
the transfer entropy estimators to ensure that marginal distributions are computed correctly.

Indices correspond to variables of the embedding, or, equivalently, colums of a `Dataset`.

- The three-argument constructor assumes there will be no conditional variables.
- The four-argument constructor assumes there will be conditional variables.

"""
struct TEVars
    𝒯::Vector{Int}
    T::Vector{Int}
    S::Vector{Int}
    C::Vector{Int}
end

"""
    TEVars(𝒯::Vector{Int},
            T::Vector{Int},
            S::Vector{Int})

Which axes of the state space correspond to the future of the target,
the present/past of the target, and the present/past of the source?
Indices correspond to variables of the embedding or colums of a `Dataset`.

This information is used by the transfer entropy estimators to ensure
the marginal distributions are computed correctly.

This three-argument constructor assumes there will be no conditional variables.
"""
TEVars(𝒯::Vector{Int}, T::Vector{Int}, S::Vector{Int}) = TEVars(𝒯, T, S, Int[])

TEVars(;𝒯::Vector{Int} = Int[],
	    T::Vector{Int} = Int[],
	    S::Vector{Int} = Int[],
		C::Vector{Int} = Int[]) =
	TEVars(𝒯, T, S, C)
    
function Base.show(io::IO, tv::TEVars) 
    s = "$(typeof(tv))(𝒯 = $(tv.𝒯), T = $(tv.T), S = $(tv.S), C = $(tv.C))"
    print(io, s)
end

export TEVars
