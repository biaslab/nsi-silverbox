## System identification of Silverbox

Silverbox is a system identification benchmark problem, found at [http://nonlinearbenchmark.org/](http://nonlinearbenchmark.org/). It is a damped harmonic oscillator with a nonlinear spring coefficient, driven by a control signal.

The goal is infer parameters in a model that can predict future output, given new input. We are testing the following models:

- Nonlinear AutoRegressive model with eXogenous input (NARX)
- Nonlinear AutoRegressive Moving Average model with eXogenous input (NARMAX)

These models are standard in the control systems community. Typically, recursive least-squares minimization or some other form of frequentist estimation is used for inference. In this repo, we employ variational Bayesian inference (Free Energy Minimisation).

#### Comments
Questions, comments and general feedback can be directed to the [issues tracker](https://github.com/wmkouw/nsi-silverbox/issues).
