"Utility functions"

function wMatrix(γ, order)
    mW = huge*Matrix{Float64}(I, order, order)
    mW[end, end] = γ
    return mW
end
