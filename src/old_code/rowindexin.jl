function indexin_rows(A1::Array{T, 2}, A2::Array{T, 2}) where {T<:Number}
    inds = []
    for j = 1:size(A1, 1)
        for i = 1:size(A2, 1)
            if all(A1[j, :] .== A2[i, :])
                push!(inds, i)
            end
        end
    end
    return inds
end
