# Sum-product update rules
@sumProductRule(:node_type     => GeneralisedFilterX,
                :outbound_type => Message{GaussianWeightedMeanPrecision},
                :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                :name          => SPGeneralisedFilterXOutNPPP)

@sumProductRule(:node_type     => GeneralisedFilterX,
                :outbound_type => Message{GaussianWeightedMeanPrecision},
                :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                :name          => SPGeneralisedFilterXIn1PNPP)

@sumProductRule(:node_type     => GeneralisedFilterX,
                :outbound_type => Message{GaussianWeightedMeanPrecision},
                :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                :name          => SPGeneralisedFilterXIn2PPNP)

@sumProductRule(:node_type     => GeneralisedFilterX,
                :outbound_type => Message{GaussianWeightedMeanPrecision},
                :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                :name          => SPGeneralisedFilterXIn3PPPN)


# Mean-field variational update rules
@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARCNLOutNPPPPP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARCNLIn1PNPPPP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARCNLIn2PPNPPP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARCNLIn3PPPNPP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalARCNLIn4PPPPNP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{Gamma},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalARCNLIn5PPPPPN)

# Structured updates
# todo
