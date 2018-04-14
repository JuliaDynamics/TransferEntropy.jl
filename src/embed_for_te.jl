"""
	`embed_for_te(source::Array{Float64, 1}, target::Array{Float64, 1}, te_lag::Int)``

Embed the `source` and `target` time series for transfer entropy computation.
The lag in the transfer entropy computation, `te_lag`, is negative for the
causal direction (`source` influences future values of `target`) and positive
for the noncausal direction (`source` influences past values of `target`).
"""
function embed_for_te(source::Array{Float64, 1},
						target::Array{Float64, 1},
						te_lag::Int)

	if te_lag < 0
		x = target[(1 + abs(te_lag)):end]
		y = target[1:(end - abs(te_lag))]
		z = source[1:(end - abs(te_lag))]
	elseif te_lag > 0
		x = target[1:(end - te_lag)]
		y = target[(1 + te_lag):end]
		z = source[(1 + te_lag):end]
	elseif te_lag == 0
		error("te_lag cannot be $te_lag")
	end

	hcat(x, y, z)
end
