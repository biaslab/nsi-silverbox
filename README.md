## Online Nonlinear System Identification

The goal is infer parameters in a model that can predict future output, given new input. We are testing the following models:

- Nonlinear AutoRegressive model with eXogenous input (NARX; see [ForneyLab node code](https://github.com/biaslab/NARX))
- Nonlinear AutoRegressive Moving Average model with eXogenous input (NARMAX; see [ForneyLab node code](https://github.com/biaslab/NARMAX))

These models are standard in the control systems community. Typically, (recursive) least-squares or some other form of frequentist estimation is used. In this repo, we employ [Free Energy Minimisation](https://en.wikipedia.org/wiki/Free_energy_principle).

We run a series of verification experiments and a validation experiment on the Silverbox data set from the [Nonlinear Benchmark](https://sites.google.com/view/nonlinear-benchmark/).

#### Comments
Questions, comments and general feedback can be directed to the [issues tracker](https://github.com/biaslab/CDC2021-onsysidFEM/issues).
