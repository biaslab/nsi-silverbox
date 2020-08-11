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
                      :name          => VariationalGFXOutNPPPPP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalGFXIn1PNPPPP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalGFXIn2PPNPPP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalGFXIn3PPPNPP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalGFXIn4PPPPNP)

@naiveVariationalRule(:node_type     => GeneralisedFilterX,
                      :outbound_type => Message{Gamma},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalGFXIn5PPPPPN)

# Structured updates
#TODO
