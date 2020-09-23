"Utility functions"

function wMatrix(γ, order)
    mW = 1e8*Matrix{Float64}(I, order, order)
    mW[end, end] = γ
    return mW
end
