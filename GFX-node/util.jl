"Utility functions"

import ForneyLab: cholinv

function wMatrix(γ, order)
    mW = 1e4*Matrix{Float64}(I, order, order)
    mW[end, end] = γ
    return mW
end

function cholinv(M::AbstractMatrix)
    return LinearAlgebra.inv(M)
end