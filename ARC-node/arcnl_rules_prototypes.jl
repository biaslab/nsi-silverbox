@naiveVariationalRule(:node_type     => AutoregressiveControlNL,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARCNLOutNPPPPP)

@naiveVariationalRule(:node_type     => AutoregressiveControlNL,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARCNLIn1PNPPPP)

@naiveVariationalRule(:node_type     => AutoregressiveControlNL,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARCNLIn2PPNPPP)

@naiveVariationalRule(:node_type     => AutoregressiveControlNL,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARCNLIn3PPPNPP)

@naiveVariationalRule(:node_type     => AutoregressiveControlNL,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalARCNLIn4PPPPNP)

@naiveVariationalRule(:node_type     => AutoregressiveControlNL,
                      :outbound_type => Message{Gamma},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalARCNLIn5PPPPPN)

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
