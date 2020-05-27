
"""
    TransferEntropyEstimator

An abstract type for transfer entropy estimators. 
"""
abstract type TransferEntropyEstimator end 

function Base.show(io::IO, estimator::TransferEntropyEstimator)
    s = "$(typeof(estimator))($(estimator.b))"
    print(io, s)
end
