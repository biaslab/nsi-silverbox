## System identification of Silverbox

Silverbox is a system identification benchmark problem, found at [http://nonlinearbenchmark.org/](http://nonlinearbenchmark.org/). It is a damped harmonic oscillator with a nonlinear spring coefficient, driven by a control signal.

The goal is infer parameters in a model that can predict future output, given new input. We are testing the following models:

- Nonlinear Autoregressive model with eXogenous input (NARX)
- Nonlinear Latent Autoregressive model with eXogenous input (NLARX)
- Generalised Filter with eXogenous input (GFX)

These models are standard in the control systems community. Typically, least-squares minimization or some other form of frequentist estimation is used for inference. In this repo, we employ variational Bayesian inference (Free Energy Minimisation). It has some advantages, such as computational efficiency, robustness to overfitting and access to posterior probablity estimates. But perhaps it won't perform as well as Gaussian process regression or deep neural networks. We investigate that question here.

#### Comments
Questions, comments and general feedback can be directed to the [issues tracker](https://github.com/wmkouw/nsi-silverbox/issues).
