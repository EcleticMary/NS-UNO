# NS-UNO

### Neutron Star EoS Inference from an Unconstrained Number of Observations

<p align="center">
  <img src="scheme_UNO_NS.png" width="850">
</p>

<p align="center">
  <i>Every neutron star plays its part.</i>
</p>

---

## Overview

**NS-UNO** is a simulation-based inference framework for reconstructing the neutron star equation of state (EoS) from an **unconstrained number of observations**.

The framework combines a **hierarchical DeepSets architecture** with a **conditional normalizing flow**, allowing a single trained model to perform Neural Posterior Estimation (NPE) using observation sets of different sizes. Each neutron star observation is represented by a set of posterior samples, allowing the model to retain information about the observational uncertainties.

NS-UNO is designed for the increasingly diverse multimessenger observations expected from current and next-generation neutron star observations.

---

## Paper

The method and results are described in:

> **V. Carvalho, M. Ferreira, M. Bejger, and C. Providência**,
> *NS-UNO: Neutron Star EoS Inference from an Unconstrained Number of Observations*,
> arXiv:2608.30573 (2026).

📄 **Paper:** [arXiv:2608.30573](https://arxiv.org/abs/2608.30573)
📥 **PDF:** [Download the paper](https://arxiv.org/pdf/2608.30573)

### Abstract

Future multimessenger observations of neutron stars are expected to substantially increase both the number and precision of astrophysical constraints on the equation of state of dense matter. This motivates inference frameworks capable of accommodating a variable, non-fixed number of observations while preserving the posterior information associated with each measurement.

NS-UNO combines a hierarchical DeepSets model with a conditional normalising flow, enabling a single trained model to perform inference from mass-radius observation sets of varying size. The framework is trained jointly on piecewise-polytropic and non-parametric Gaussian-process EoS ensembles and is designed to accommodate observations with different numbers and precisions.

---

## Method

The NS-UNO framework consists of two main components:

1. **Hierarchical DeepSets**
   Processes an unconstrained number of neutron star observations while preserving permutation invariance.

2. **Conditional Normalizing Flow**
   Maps the encoded observational information to the posterior distribution of the neutron star EoS.

This allows the same model to be applied to different numbers of observed neutron stars without retraining the model for each observation count.

<p align="center">
  <img src="scheme_UNO_NS.png" width="850">
</p>

---

## EoS Models

The framework is trained using multiple classes of neutron star EoS models, including:

* **Piecewise-polytropic (PT) EoSs**
* **Non-parametric Gaussian-process (GP) EoSs**

Training on both parameterised and non-parametric EoS families allows the model to learn a broader representation of the possible neutron star EoS space.

---

## Observations

NS-UNO is designed to work with sets of neutron star observations, including:

* Mass measurements
* Radius measurements
* Mass-radius posterior samples
* Different numbers of observed neutron stars
* Observations with different levels of uncertainty

The number of observations is not fixed during inference, allowing the framework to naturally incorporate additional neutron stars as new observations become available.

---

## Repository Structure

```text
NS-UNO/
│
├── *.py                 # Source code
├── scheme_UNO_NS.png    # NS-UNO framework schematic
├── ...
```

Large datasets, generated results, plots, model checkpoints, and other generated files are excluded from version control through `.gitignore`.

---

## Requirements

The code is based primarily on Python and PyTorch.

Main dependencies include:

* Python
* PyTorch
* NumPy
* SciPy
* scikit-learn
* nflows
* Matplotlib

A complete list of dependencies can be provided in `requirements.txt`.

---

## Citation

If you use NS-UNO in your research, please cite:

```bibtex
@article{Carvalho2026NSUNO,
  title         = {NS-UNO: Neutron Star EoS Inference from an Unconstrained Number of Observations},
  author        = {Carvalho, Val{\'e}ria and Ferreira, M{\'a}rcio and
                   Bejger, Micha{\l} and Provid{\^e}ncia, Constan{\c{c}}a},
  journal       = {arXiv preprint arXiv:2608.30573},
  year          = {2026},
  eprint        = {2608.30573},
  archivePrefix = {arXiv},
  primaryClass  = {nucl-th}
}
```


