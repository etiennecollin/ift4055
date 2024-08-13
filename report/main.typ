#import "/github/typst-templates/basic.typ": *
#import "@preview/unify:0.5.0": unit

#let abstract = [
  In various scientific fields, the exponential growth in data volume and
  resolution, driven by technological advancements, presents a formidable
  challenge for probabilistic data analysis. Traditional computational methods
  become impractical as some simulations, for example in cosmology, require an
  exorbitant 100 million CPU hours to evaluate a single likelihood or posterior.
  Because of this, scientists turn to modeling accurate approximations of their
  problems. But this creates a "chicken or egg" dilemma: training models to
  approximate these computationally intensive oracles necessitates prior
  evaluations of the oracle to generate training samples. There is a need for
  efficient and intelligent sampling.

  Addressing this problem is hard; we propose an innovative approach utilizing a
  Gaussian Process (GP) as a surrogate model, which provides an uncertainty on
  approximations. To efficiently train the GP, we employ a Generative Flow Network
  (GFN) for acquiring training samples. The inputs to the GFN are pre-processed
  using a Set Transformer, enhancing the model's capacity to identify and leverage
  patterns within the data. By training the GFN on inexpensive-to-sample problems
  of the same class, it learns underlying patterns, structures and symmetries in
  the data, thereby optimizing the sampling process for the GP. This methodology
  significantly reduces the computational burden, enabling more feasible and
  efficient analysis of high-resolution and high-dimensional data across various
  scientific disciplines.
]

#show: doc => article(
  title: "Summer Research Internship", subtitle: "Report", abstract: abstract, authors: (
    (
      name: "Etienne Collin", id: 20237904, affiliations: ("Université de Montréal", "CIELA", "MILA",), email: "collin.etienne.contact@gmail.com",
    ),
  ), supervisors: ("Yashar Hezaveh", "Laurence Perreault Levasseur",), class: (
    name: "Projet Informatique Honor", number: "IFT4055", section: "A", semester: "Summer 2024", instructor: "Professor Louis Salvail",
  ), institution: "Université de Montréal", duedate: "September 1, 2024", logo: "/github/typst-templates/logo_udem.png", bib: "/github/ift4055/report/refs.bib", bibstyle: "ieee", titlepage: true, toc: true, showdate: false, indent-first-line: true, cols: 1, lang: "en", paper: "us-letter", fonts: ("New Computer Modern", "Maple Mono NF", "JetBrainsMono Nerd Font"), fontsize: 12pt, doc,
)

= Terminology

/ Oracle: The oracle is the object (function or distribution) that we are trying to
  emulate/approximate. In our case, it is a distribution that is difficult to
  sample from due to its computational complexity.
/ Surrogate network: The surrogate network is a model that approximates the oracle. In our case, it
  is immensly easier to sample from than the oracle. Through training, the
  approximation should become more and more accurate.
/ Acquisition model: The acquisition model is a model that decides where to sample next. It uses the
  surrogate network to estimate the uncertainty in the model and decides where to
  sample the oracle next to reduce this uncertainty. It is akin to the concept of
  acquisition functions in Bayesian Optimization.

#pagebreak()
= Surrogate Network

When trying to approximate a complex problem, the first step should be to
determine which surrogate network to use. In our case, a Gaussian Process (GP)
was chosen as a surrogate network. The GP is a powerful tool for approximating
complex functions and distributions. It provides an uncertainty estimate on its
predictions, which is crucial for applications where accuracy is important.

In general, GPs are used in the context of Bayesian Optimization (BO) to model
an unknown function. The goal of BO itself is to optimize that unknown function.
In other words, BO is regularly used to optimize parameters of a process in
order to maximize its ouptut. In BO, along with the GP, is an acquisition
function used to decide where the unknown function should be sampled next to
provide new training data for the GP. The acquisition function uses the
uncertainty estimate provided by the GP to decide where to sample next. The GP
is then fitted with the new point and the loop repeats until the problem is
optimized (hoping for a global solution).

Hence, GPs take a dataset as an input and output the distribution over the
function that generated the dataset. This distribution is represented by a mean
and a variance, which can be used to make predictions and provide uncertainty.

#figure(
  image("assets/gp_simple.png", width: 100%), caption: [A simple example of a Gaussian Process trained on $y = sin(-2 pi (x-0.15)) + sin(-4 pi (x-0.15))$.
    The shaded area representing the $2 sigma$ uncertainty of the model and the blue
    line the mean of the model. Although the mean of the model often corresponds to
    the true function, as expected, its uncertainty grows when training samples are
    further away from eachother.],
) <fig-gp_simple>

This is somewhat similar to our problem, where we are trying to approximate a
complex distribution. However, our work aims at training the GP to model the
entire distribution and not only finding its minima/maxima.

Moreover, this taining process is incredibly important, as in order to sample
efficiently, we need to first see how the surrogate network is doing before
proposing new samples that will actually improve its performance. This is like a
student taking a test, seeing the results, and then studying the material they
did poorly on. A GFlowNet will fulfill this role.

Many other candidates were considered and researched to fill the surrogate
network role, such as Probabilistic Neural Networks, Variational Inference
networks and GFlowNets. However, GPs were chosen for their simplicity,
relatively inexpensive-to-sample and train nature and their ability to provide
uncertainty estimates. This last property proves to be crucial for our problem.

In particular, the GP used was based on the "High-Dimensional Bayesian
Optimization with Sparse Axis-Aligned Subspaces" (SAASBO) paper
@erikssonHighDimensionalBayesianOptimization2021 and implemented using
@balandatBoTorchFrameworkEfficient2020. This paper introduces a novel method to
train GPs that is more efficient on high-dimensional problems where training
samples are limited and expensive to obtain. This makes a lot of sense for our
problem, as we aim to minimize the number of samples required to train the GP
because of the cost of evaluating the oracle. As for BOTorch, it is an efficient
and modular framework built on top of PyTorch for Bayesian Optimization.

#pagebreak()
= Acquisition Model

Similar to Bayesian Optimization (BO), we require an acquisition model to
determine the next location for sampling the oracle, which will then provide the
training point for the Gaussian Process (GP). Typical BO uses pre-determined
acquisition functions such as Expected Improvement (EI), Probability of
Improvement (PI) or Upper Confidence Bound (UCB). However, these functions are
not well-suited for our problem, as they are designed to optimize a function and
not to sample from a distribution. Moreover, such acquisition functions would
not allow leveraging the structure and patterns of the data to optimize the
sampling process.

Let's imagine a simple example: we have class of problems in two dimensions
where each instance of the problem consists of a gaussian mixture of two
gaussians that are always symmetrically placed around the origin. The goal is to
sample from the oracle to train the GP. The acquisition model should be able to
leverage the fact that the problem is symmetric and that the oracle is a
gaussian mixture to propose samples that are likely to be informative. In other
words, when the acquisition model finds the first gaussian, it should not get
stuck in that mode, be able to infer the presence of the second gaussian and
propose samples that are likely to be close to it. All of this because the model
has learned the properties of that class of problems

Just like the surrogate network, many model types were considered for the
acquisition model. Reinforcement Learning (RL) was chosen as the best approach
because RL allows the model to learn the best strategy for the sampling process
by interacting with the environment (the oracle) over the course of many
iterations. This enables our training strategy presented further down.

RL learns thanks to a reward function that tells the model how well it is doing.
Contrary to other kinds of networks which use a loss function, rewards do not
need to be differentiable which allows us to use statistical analysis tools such
as PQMass @lemosPQMassProbabilisticAssessment2024 in their design.

RL also allows for fast inference as, given a state, the network can propose a
sample without having to compute a reward. This is interesting as when training
with inexpensive-to-sample problems, we can design a powerful reward that would
be too expensive to compute in inference on expensive problems.

#pagebreak()
== GFlowNet

RL is a vast domain which covers multiple learning strategies. Once again, many
were considered such as Deep Q-Learning, PPO, actor-critic, etc. One of the
challenges was making work RL in a continuous action space, where actions are
not constrained to a limited set of possibilities (such as the set of valid
moves in a game). In our situation, the points sampled from the oracle can be
anywhere in the space. Although classical RL has ways to make this work, we
settled on a different solution.

GFlowNets were chosen for their ability to sample proportionally to the reward.
In other words, GFNs do not simply maximize the reward; they are samplers that
will sample better points (according to the reward) more often, and points that
are equally good have the same probability of being sampled
@edwardhuAreGFlowNetsFuture2024. This is a great property for our problem as it
will make sure that the model samples the most informative points first.
Moreover, here are a few advantages of GFlowNets and their sampling nature
@bengioGFlowNetTutorial2022:

- By sampling proportionally to the reward, GFlowNets explore a wider variety of
  high-reward solutions rather than prematurely converging on suboptimal solutions
  and getting stuck in a local maxima. This helps in understanding the entire
  landscape of potential solutions, not just the highest peak.
- Solutions found by considering a broader range of high-reward paths are often
  more robust and generalize better to new, unseen data or scenarios.
- By sampling according to the reward distribution, the GFlowNet can focus
  computational resources on more promising areas of the solution space, leading
  to more efficient learning.
- In large or continuous action spaces, directly finding the maximum reward can be
  computationally infeasible. Proportional sampling provides a tractable way to
  explore these spaces effectively.
- In applications where outcomes are stochastic, the optimal strategy often
  involves probabilistically favoring high-reward actions rather than
  deterministically choosing a single action.

GFlowNets are based on flow networks and are comprised of a forward model and a
backward model. The forward model, or forward policy, is a policy that decides
the probability of transitioning from one state to another. It is denoted by $P_F (s' | s)$ where $s$ is
the current state and $s'$ is the next state after taking an action. Conversly,
the backward model, or backward policy, is a policy that determines the
probability of reversing the transition, i.e., moving from state $s'$ back to
state $s$. The backward policy is denoted by $P_B (s | s')$. To learn, these
models require a training objective.

#pagebreak()
=== Training Objectives

In recent years, several training objectives were devised to train GFlowNets.
These are, in a way, a set of rules or constraints that the model follows such
that sampling of the model is proportional to the reward. The most popular ones
are @bengioGFlowNetTutorial2022:

- Flow-matching objective
- Detailed balance objective
- Trajectory balance objective
- Subtrajectory balance objective

The *flow-matching objective* is the simplest, as it ensures that the total flow
into each non-terminal state equals the total flow out of it. In equation form,
this gives

$ sum_s' F(s' -> s) = sum_s'' F(s -> s'') $

which can be converted into a loss function to train the model. This enforces
consistency in the way states are visited during the generation process.

The *detailed balance objective* indirectly uses flow-matching by making the
output of the model a softmax of the possible actions in each state. This is
enforced by the following constraint

$ F(s)P_F (s'|s) = P_B (s|s') F(s') $

which can be useful in cases where the number parents for each state is large as
there is no sum to compute for each state compared to flow-matching.

The *trajectory balance objective* is a more complex objective that ensures that
the model samples trajectories that are consistent with the reward. A trajectory
is a sequence of states that the model visits that starts at $s_0$ and ends at a
terminal state $s_n$. This is enforced by the following equation

$
  F(s_0) product_(t=1)^n P_F (s_t|s_(t-1)) &= R(s_n) product_(t=1)^n P_B (s_(t-1)|s_t) \
  R(s_n)                                   &= (F(s_0) product_(t=1)^n P_F (s_t|s_(t-1))) / (product_(t=1)^n P_B (s_(t-1)|s_t))
$

According to @malkinTrajectoryBalanceImproved2023, this training objective can
lead to faster convergence of the model.

Finally, the *subtrajectory balance objective* is variant of trajectory balance
that, as the name implies, balance the flow over subtrajectories instead of
complete trajectories. Accorvding to @madanLearningGFlowNetsPartial2023, this
can accelerate convergence and provide better performance in sparse reward
spaces.

In practice, our GFlowNet implementation is inspired by the examples presented
in the torchgfn library @lahlouTorchgfnPyTorchGFlowNet2023 and is based on the
trajectory balance training objective.

#pagebreak()

#figure(
  image("assets/gfn_simple.png", width: 100%), caption: [A simple example of a GFlowNet trained on a gaussian mixture reward using a
    trajectory balance training objective @lahlouTorchgfnPyTorchGFlowNet2023. Each
    action taken by the model represents a relative move on the $x$ axis (position)
    and $s_0$ is the initial $x$ position. The upper section of the figure shows in
    red what the network learned to sample, and in black the gaussian mixture
    reward. The lower section of the figure shows the exploration/exploitation
    process of the network at each step. In this case, the GFlowNet isn't able to
    learn the leftmost mode of the gaussian mixture.],
) <fig-gfn_simple>

== Designing The Reward Function

The reward function is one of the most important part of the acquisition model.
In essence, it is dictating what the network learns when training. In our case,
there are a few things that must be conveyed to the model:

- We want to minimize the number of steps taken for training
  - This is important because the oracle is expensive to sample from. At each
    training step, we sample the oracle to get new training points for the surrogate
    network. The fewer steps taken, the less expensive the training process. This
    will also teach the network that it is important to sample the most impormative
    points first.
- We want to make sure the surrogate network improves thanks to the new training
  point we feed it each step.
  - This is important because the goal of the acquisition model is to provide the
    surrogate network with the most informative points to improve its approximation
    of the oracle. If the points provided do not improve the "score" of the
    surrogate, then the acquisition model is not doing its job.
- We want to make sure that the surrogate network learns to approximate the oracle
  accurately.
  - This is important because the goal of the surrogate network is to approximate
    the oracle. If the surrogate network is not accurate, then the acquisition model
    is not proposing points that are significant to the oracle; it is not helping
    the surrogate learn the "signature" of the oracle.

Conveying these three points to the model mathematically is not an easy task.
Furthermore, each term in the reward function has a different scale, we therfore
need normalize them. Here is what we came up with:

First, to minimize the number of steps taken for training, we use include the
following term in the reward function: $ - ("n_samples"/ "max_samples") $
Where "n_samples" is the total number of samples taken during training. This
will make sure that each step taken by the model is penalized. The model will
want to take the fewest samples possible to maximize its reward. We use a
hyperparameter representing the maximal number of samples allowed in training to
normalize n_samples. We believe this is reasonable as it allows the user to
control the maximal cost of their training.

To make sure the surrogate network improves thanks to the new training point we
feed it each step, we first need a way to compute how well the model is doing.
An intuitive way to do that is by computing the entropy of the surrogate network
before and after the new training point is fed to it. If the entropy decreases,
then the model has improved. If the entropy increases, then the model has not
improved. In practice, we use the Kullback-Leibler divergence (KL divergence) in
our reward term to compute the entropy of the surrogate network. The KL
divergence is a measure of how one probability distribution is different from a
second, reference probability distribution. In our case, the first probability
distribution is a surrogate network, and the second is the dataset after the new
training point is added to it. This means we are computing how different the
predictions of the surrogate network are from what it is trying to learn.

Hence, we include the following term
$C$ in the reward function: $
  A &= 1 - exp(-D_(K L)("gaussian_process"_(t-1), "new_dataset"))\
  B &= 1 - exp(-D_(K L)("gaussian_process"_t, "new_dataset"))\
  C &= -(B-A)
$
Where
$t$ represents a specific training step. We normalize each KL divergence using
the calculation $1 - e^(-D_(K L))$ @emreAnswerCanNormalize2011. The idea is that
if the GP is improving, then $B<A$ because at step $t-1$ (in $A$), the GP had
not been trained on the new datapoint and the KL divergence is bigger. The
negative in front of $B-A$ makes sure that the reward is positive when the
network improves.

Finally, to make sure that the surrogate network learns to approximate the
oracle accurately, we use PQMass, which is non-differentiable, to quantify the
probability that the predictions of the GP and the test dataset come from the
same distribution. We use the same tool to quantify how well the train and test
dataset distributions match.

During training, inexpensively sampled problems of the same class are used.
Hence, although the train dataset needs to be kept minimal such that the network
learns to sample efficiently, it is possible to use a large test dataset. This
means that, given a tightly uniformly sampled test dataset, we can accurately
compute how the predictions of the GP differ from the oracle with PQMass
@lemosPQMassProbabilisticAssessment2024.

To normalize the PQMass values, we use the following formula: $ ("pqm" -
min("pqm")) / (max("pqm") - min("pqm")) $
where pqm is the array returned by PQMass representing the values of a
$chi^2$ distribution.

We therefore include the following terms in the reward function: $ -
("pqm_gp_test" - min("pqm_gp_test")) / (max("pqm_gp_test") - min("pqm_gp_test"))
- ("pqm_train_test" - min("pqm_train_test")) / (max("pqm_train_test") -
min("pqm_train_test")) $
With all these terms combined, we have a reward function that should allow the
GFlowNet to provide meaningful points to the surrogate network.

#pagebreak()
== Designing The State and Action Space

Having a good reward function is not enough. The model must also be able to
understand the state of the problem it is in. It can be challenging to determine
what the state should be and how it should be encoded into the network.

Ideally, to help with training, we want the state to be as informative as
possible and to contain all the information the model needs to make a decision.
In our case, we determined that the state should contain the following
information:

- The list of already sampled points used to train the GP
- The output distribution of the GP which represents a distribution over the
  oracle approximations
  - This $m times n$ output is obtained by sampling the GP on a large range of
    points $[x_0, dots, x_n]$. This is extremely fast to obtain and the output is
    represented by a two dimensional tensor, viewed as a matrix, with each column ${C_i | i in [0, n]}$ containing $m$ realizations
    of the GP on input $x_i$. This means we have discretized the GP's output
    distribution into $n$ points which have their mean and variance computed from
    the $m$ realizations.
- The new point that will be evaluated by the oracle and added to the training
  dataset
- $t$ a time step
  - One of the important restrictions of a GFlowNet is that there must not be cycles
    in the state space. Adding an entry $t$ to the state prevents loops because the
    network cannot go back in time @lahlouTorchgfnPyTorchGFlowNet2023.

The action space, being continuous, will need to be represented by a probability
distribution obtained from the forward model of our GFlowNet. At each step, the
action can then be sampled from that distribution and applied to update the
state. In our case, the action is a new point to sample from the oracle.

One problem remains: how can we encode the list of points and the output of the
GP into the state? The solution is to use a Set Transformer.

#pagebreak()
= Set Transformers

Because we cannot directly feed the list of points and the output of the GP into
the state of the GFlowNet, we need to use a Set Transformer to process the input
and use its output embeddings as the state. Set Transformers are a type of
transformer that is particularly useful to process inputs when the order of the
data does not matter, as is the case with our list of points and the output of
the GP. Using a set transformer, it is possible to make sure that every
permutation of a list of numbers is treated the same way.

We pre-train the set transformer on a dataset of inexpensive-to-sample problems
and use it to process the input to the GFlowNet. This allows the GFN model to
learn the underlying patterns, structures and symmetries in the data without
being hindered by the order of the points.

#todo[
  - Add example of a set transformer IO
]

#pagebreak()
= Training Architecture and Process

#todo[
  - describe the final architecture of the project
  - describe the training process
  - describe the evaluation process
  - Diagram of the architecture
]

#pagebreak()
= Results

Now that we have presented the architecture of the project, we can discuss the
implementation and results. The project is still in its early stages, and many
challenges remain to be addressed. However, the initial results are promising.

#todo[
  - what are the results so far?
  - does it work?
]

#pagebreak()
= Discussion
#todo[
  - what are the limitations?
  - what are the advantages?
  - what are the potential applications?
]

#pagebreak()
= Conclusion
#todo[
- what are the key takeaways?
- what are the next steps?
  - try switching from TB to subtrajectory balance
  - Adding to output
  - `[[[x0, x1, ..., xn], y], ...]` a list of points to directly use in sampling
  because the acq model is sure they are good
- Good but intense introduction to AI and research
- Taught me a lot about what I like and don't like
- Struggles with productivity and progress
- Opening: I believe this project has a lot of potential and could impact a lot
of fields where sampling is a bottleneck. I am excited to see where this
research leads and how it can be applied in the real world. With more time and
more knowledge, I am confident that this project could lead to significant
advancements in the field of AI.
]

