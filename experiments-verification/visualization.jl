using Plots
pyplot()


function plot_forecast(output, predictions, ix_trn, ix_val; tt=1, posterior=false, plotargs=[])

    # Limits
    ylims = [minimum(output), maximum(output)]

    # Plot training output
    plot(ix_trn[1:tt:end], output[ix_trn[1:tt:end]], color="blue", ylims=ylims, label="training data", plotargs...)

    # Plot validation output
    scatter!(ix_val[1:tt:end], output[ix_val[1:tt:end]], color="black", ylims=ylims, label="validation data", plotargs...)

    # Plot fit and forecast
    if posterior
        sdev = sqrt.(predictions[2][ix_trn[1:tt:end]])
        plot!(ix_trn[1:tt:end], predictions[1][ix_trn[1:tt:end]], ribbon=[sdev, sdev], color="purple", label="predictions", plotargs...)
    else
        plot!(ix_trn[1:tt:end], predictions[ix_trn[1:tt:end]], color="purple", label="predictions", plotargs...)
    end
end


function plot_errors(output, predictions, ix_val; plotargs=[])

    # Compute prediction errors
    pred_errors = (predictions[ix_val] - output[ix_val]).^2

    # Compute root mean square
    RMS = sqrt(mean(pred_errors))

    # Plot errors of time
    scatter(pred_errors, label="RMS = "*string(RMS), ylabel="squared error", yscale=:log10, plotargs...)
end