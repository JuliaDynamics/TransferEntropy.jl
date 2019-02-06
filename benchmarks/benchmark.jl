struct BenchmarkSet
    dynamicalsystem::Function
    n_realizations::Int
    te_func::Function
    tuning_param::Symbol
    tuning_vals::Vector
    which_ts::Vector{Int}
    which_pos::Vector{Int}
    which_lags::Vector{Int}
    vars::TEVars
    ts_lengths::Vector{Int}
    binsizes::Union{Vector{Int}, Vector{Float64}}
end

function summary(b::BenchmarkSet)

end
