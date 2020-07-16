@naiveVariationalRule(:node_type     => NAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNARXOutNPPPPP)

@naiveVariationalRule(:node_type     => NAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNARXIn1PNPPPP)

@naiveVariationalRule(:node_type     => NAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNARXIn2PPNPPP)

@naiveVariationalRule(:node_type     => NAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNARXIn3PPPNPP)

@naiveVariationalRule(:node_type     => NAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalNARXIn4PPPPNP)

@naiveVariationalRule(:node_type     => NAutoregressiveX,
                      :outbound_type => Message{Gamma},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalNARXIn5PPPPPN)

# # Structured updates
#
# @structuredVariationalRule(:node_type     => Autoregressive,
#                            :outbound_type => Message{GaussianMeanVariance},
#                            :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution),
#                            :name          => SVariationalAROutNPPP)
#
# @structuredVariationalRule(:node_type     => Autoregressive,
#                            :outbound_type => Message{GaussianMeanVariance},
#                            :inbound_types => (Message{Gaussian}, Nothing, ProbabilityDistribution, ProbabilityDistribution),
#                            :name          => SVariationalARIn1PNPP)
#
# @structuredVariationalRule(:node_type     => Autoregressive,
#                            :outbound_type => Message{GaussianMeanVariance},
#                            :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution),
#                            :name          => SVariationalARIn2PPNP)
#
# @structuredVariationalRule(:node_type     => Autoregressive,
#                            :outbound_type => Message{Gamma},
#                            :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing),
#                            :name          => SVariationalARIn3PPPN)
#
# @marginalRule(:node_type => Autoregressive,
#               :inbound_types => (Message{Gaussian}, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution),
#               :name => MGaussianMeanVarianceGGGD)
