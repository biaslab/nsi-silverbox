@naiveVariationalRule(:node_type     => NLatentAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNLARXOutNPPPPP)

@naiveVariationalRule(:node_type     => NLatentAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNLARXIn1PNPPPP)

@naiveVariationalRule(:node_type     => NLatentAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNLARXIn2PPNPPP)

@naiveVariationalRule(:node_type     => NLatentAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNLARXIn3PPPNPP)

@naiveVariationalRule(:node_type     => NLatentAutoregressiveX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalNLARXIn4PPPPNP)

@naiveVariationalRule(:node_type     => NLatentAutoregressiveX,
                      :outbound_type => Message{Gamma},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalNLARXIn5PPPPPN)
