# riemann:  A research framework for MCMC

The Centre for Translational Data Science (CTDS) is a multidisciplinary centre
for data science research at the University of Sydney, developing new methods
in Bayesian statistics and machine learning to solve challenging problems 
across natural, biomedical, and social sciences with broad social impact.
Sampling distributions by Markov chain Monte Carlo (MCMC) is a key technique
consistently employed by CTDS across all its activities.
Many problems can be adequately sampled using off-the-shelf technology, but
since CTDS's research program involves the development of new data science 
methods as well as outcomes in different domain areas, there is no guarantee
that off-the-shelf packages will provide good results.

This code base contains a library of advanced MCMC methods for sampling 
complex, high-dimensional posterior distributions.  It focuses at the moment
on Metropolis-Hastings proposals, since this encompasses a very broad class
of widely used proposal types.  The project was started in particular with
geometric methods in mind -- those that take advantage of the local geometry
of the posterior distribution, such as Riemannian manifold Monte Carlo --
which is why it carries the working name `riemann`.
It is being designed with potentially quite complex models in mind, to be
compatible with auto-differentiation, GPU-based code, adaptive proposals,
and samplers to run on distributed architectures.
  
The code architecture
---------------------

At present the code contains three base class hierarchies:
* `Sampler`, a basic MCMC sampler
* `Proposal`, an abstract base class for Metropolis-Hastings proposals
* `Model`, an abstract base class for statistical models
Each `Sampler` will eventually understand how to display itself along with
key performance plots, such as trace plots or posterior slices.
`Proposal` instances include support for asymmetric proposals and for
gradient-based proposals by forwarding a request for gradient information
to each `Model`.

About contributing
------------------

Watch this space for code license, contribution guidelines, and more.
