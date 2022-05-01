













-   [Introduction](#introduction)
    -   [Transforming objectives to
        scores](#transforming-objectives-to-scores)
-   [Examples](#examples)
    -   [Optimizing volume and price of
        packages](#optimizing-volume-and-price-of-packages)
    -   [Czech problem: Finding variable
        importance](#czech-problem-finding-variable-importance)
-   [Research](#research)
-   [Examples](#examples-1)
    -   [Problem 1](#problem-1)
        -   [Pareto front](#pareto-front)
        -   [Pre-conditioning the
            problem](#pre-conditioning-the-problem)
        -   [Testing the pre-conditioned
            problem](#testing-the-pre-conditioned-problem)
        -   [Convergence towards desired
            trade-off](#convergence-towards-desired-trade-off)
        -   [Bonus: Estimate nadir vector](#bonus-estimate-nadir-vector)
        -   [Absolute best solution](#absolute-best-solution)
-   [References](#references)

# Introduction

Multi-objective optimization problems (MOOPs) attempt to find solutions
for a set of *non-aggregable* objectives. MOOPs are commonly formulated
as in :

Non-aggregable refers to conflicting objectives with differing range. A
solution **s**<sub>1</sub> dominates another solution **s**<sub>2</sub>
iff ∥ **s**<sub>1</sub> ∥ \< ∥ **s**<sub>2</sub> ∥ and
∀ *s* ∈ (**s**<sub>1</sub>−**s**<sub>2</sub>) *s* ≤ 0. Furthermore,
non-aggregable means that we cannot make any assumptions about the
*trade-off* between found solutions for single objectives, except for
the stated dominance assumption. The scalarizer in a MOOP is only means
to an end, one has to inspect the resulting solution. The scalarizer in
MOOPs does not technically transform a MOOP into a single-objective
optimization problem (SOOP), because its value when a solution is found
is meaningless.

However, we suggest using a form of **pre-conditioning** for MOOPs that
allows the scalarizer to quantify the trade-off between objectives and
therefore relaxes the dominance assumption. With the relaxation in
place, the scalarizer’s value becomes meaningful in that we can use it
to compare solutions. If we were to do minimization, then a lower
scalarized value would mean that a more optimal solution was found.

## Transforming objectives to scores

Commonly, objectives in MOOPs each have their own range (*magnitude*).
To address this problem, objectives may be scaled into a dimensionless
unit scale. This is done by defining
*f*′<sub>*i*</sub> = *f*<sub>*i*</sub> × (*z*<sub>*i*</sub><sup>nad</sup>−*z*<sub>*i*</sub><sup> \* \*</sup>)<sup>−1</sup>,
where the *nadir*- and *utopian* (\*\*) vectors are, simply speaking,
the worst and ideal objective vectors (refer to (Miettinen 2008) for a
more precise definition). However, while this scaling results in
objectives with a range of $\\interval{0}{1}$ (which we call
**scores**), these scores do **not** have linear behavior. For example,
a score of 0.9 might not be better by 0.1 than a score of 0.8, but
considerably better if high scores are hard to obtain for a specific
objective.

Pre-conditioning of scores refers to transforming such raw scores into
ones with linear behavior. It can be achieved by approximating the
empirical cumulative distribution (ECDF ) of an objective and then using
this function to scale and *rectify* an objective’s values. Furthermore,
this approach does not require to scale by the range as established by
nadir- and utopian vectors. While the nadir-vector is usually difficult
to obtain, the utopian vector is the result of minimizing each objective
separately in order to obtain the lowest possible loss.

In order to approximate the ECDF<sub>*i*</sub>, we can take *k*
uniformly chosen samples **h** from the hyperparameter space **H**
(where *k* ⋘ *m*) and compute losses for each *i*-th objective . The
observed losses are then used to approximate the marginal distribution
of this objective and to create a uniform score with linear behavior .

If all of the objectives in a scalarizer were such pre-conditioned
scores, then its resulting values can be used to compare solutions,
which was not previously possible.

# Examples

Here, we show some examples where this may be useful.

## Optimizing volume and price of packages

In this problem, we want to reduce the volume of packages, while
simultaneously increasing the price. In other words, we want to produce
smaller items and sell them at a higher price. Note that the
corresponding pre-conditioned scores are defined as
𝒮<sup>vol</sup> = 1 − ECDF<sup>vol</sup> and
𝒮<sup>price</sup> = ECDF<sup>price</sup>, respectively, such that a
relatively low observed volume has a high score, and a relatively high
observed price has a high score. This problem has no ground truth. It is
a classical MOOP where the decision maker may express some
**preference** for the trade-off of a desirable solution.

**Goal**: Learn hyperparameters that maximize the scalarizer given the
trade-off preference.

**Formulation**: The scalarizer is perhaps simply defined as in and the
optimization lies in finding the hyperparameters that maximize it .

Perturbations to the preference (weights) can lead to different Pareto
optimal solutions. However, since the scalarizer’s value gives an
indication as to the *absolute* goodness of the solution (since the
trade-off is now perfectly quantifiable), we can select the best
solution among a set of solutions that were produced with the same
preference. The conjecture is, however, that this scalarizer produces
unique Pareto optimal solutions. This means that in this case, there is
a unique Pareto optimal solution for every possible realization of the
preference.

## Czech problem: Finding variable importance

In this problem, we have a ground truth and hundreds of objectives. Each
objective measures the deviation from some ideal continuous process
model, in a specific segment. We pre-condition these objectives by
simulating a large number of random processes as they may occur in
reality. We do this to obtain losses as they typically occur given the
*constant* process model (i.e., what is a typical deviation from the
process model in a given segment). In the next step, we take the 15
processes that were actually observed (the student projects) and compute
the losses for each project and each objective. This results in a loss
matrix **L**<sup>*m* × *n*</sup>, where *m* is the projects, and *n* the
objective. Next, this loss matrix is transformed into a score matrix
**S**<sup>*m* × *n*</sup> , by plugging each project’s *n*-th loss into
the corresponding ECCDF  (i.e., 1 − ECDF ) as approximated before (a low
observed loss corresponds to a high score).

**Goal**: Find variable importances such that we can learn which loss in
which segment is important in order to predict the ground truth.

**Formulation**: We use the words *variable importance* instead of
weights, as the scalarizer is a weighted mean . In it, all coefficients
must be larger than or equal to 0, but not all of them can be 0. Instead
of a constraint, we use a piece-wise definition where all weights are
set to 1 when they are all zero, as they all are equally important (we
do this for numeric stability so we are not dividing by 0).

Obviously, here we want to minimize the deviation between the ground
truth and the predicted ground truth . The variable importances can
directly be used in a regression model (using the scalarizer from the
objective).

# Research

Here we list a few open problems that could be subject to potential
research.

-   Pre-conditioned scores lead to *proper* Pareto optimal solutions,
    where *properly* refers to unbounded trade-offs between the
    objectives **not** being allowed. In fact, not only are the
    trade-offs bounded, but further *comparable*. We need to investigate
    what the impact of this is.
-   Specifying preference with ordinary or simply scaled objectives in a
    MOOP may not lead to the desired result. We would have to
    investigate if using pre-conditioned scores alleviates the problem.
    The role of weights (relative importance) has been shown to be often
    misleading (e.g., Roy and Mousseau (1996)). We have to investigate
    if this role is more clear now using pre-conditioned scores.
-   Conjecture: A linear scalarizer using pre-conditioned scores
    produces a) provably Pareto optimal solutions, and b) unique
    solutions (same preference leads to the same solution).
-   An evenly distributed set of weights does not necessarily produce an
    evenly distributed representation of the Pareto optimal set, even if
    the problem is convex. We would need to investigate if this still
    applies when using pre-conditioned scores instead of raw or scaled
    objectives.

# Examples

In this section, we will make some tests, that is, compute some actual
problems and examine what the impact of pre-conditioning is.

## Problem 1

Here, we pick a problem for optimization from[1]. More concretely, we
will optimize the Fonseca–Fleming function (Fonseca and Fleming 1995).
We know about this problem that it has a 50/50 trade-off optimum at
 ≈ \[0.63212,0.63212\]. Setting all weights to zero will lead to it.
However, it is hard to achieve the actually desired trade-off in this
problem. While we can compute the Pareto front on which all solutions
are considered to be equally good, we can show using pre-conditioning
that this is not actually true, as we now have the means to **order**
solutions. Therefore, the solutions often jump to either  ≈ \[0,1\] or
 ≈ \[1,0\].

The problem is defined as in .

We will test this problem with **`10`** variables (*i* = 10).

``` r
# Set up number of variables and lower/upper bounds for this problem:
i <- 10
lb <- -4
ub <- 4
```

The **goal** of this problem is to apply pre-conditioning so that we are
able to find solutions that match our preference better.

Here is what we will do:

1.  Compute the Pareto front for the problem:
    -   I suggest using a so-called augmented Chebyshev scalarizer
        (Miettinen 2008), as it has a couple of advantages: It avoids
        weakly Pareto optimal solutions, it uses the utopian vector as
        reference point for optimization (which is just a vector of
        ones/highest scores in our case), it can find any Pareto optimal
        solution
    -   We would express the preference a priori using the weights in
        the scalarizer, and then repeat the optimization a few times
        with different random starting points.
    -   This way we’d get an idea of how the weights actually lead to a
        desired trade-off
2.  Transform all problems to P\* using pre-conditioning
3.  Repeat what we have done in 2 with all P\*. Now we can investigate
    the following:
    -   Does the solution lie on the Pareto front? If not, how far off
        is it? If off, is it within margin of error or is there
        significance?
    -   Since we have calibrated scores and a preference per solution,
        we can now exactly measure by how far the trade-off between
        objectives deviates from the trade-off as found by the solution.
        For example, we perhaps wanted a 50:50 trade-off, so the
        solution must be, e.g., $\\approx\\interval{0.3}{0.31}$,
        $\\approx\\interval{1}{1}$, $\\approx\\interval{0.75}{0.73}$,
        etc.; i.e., the scores must be well-balanced in this case.

In this kind of problem, the preference is expressed by the decision
maker a priori or interactively, and then one would find the
hyperparameters that produce a Pareto optimal solution according to this
preference. However, and I think here is another edge: If the scalarizer
were a weighted mean (as in the CZ problem), and we would learn the
weights as well, then the optimization would perhaps find the single
(global) absolute best solution, that produces the maximum possible
score of the entire trade-off. We will investigate this here as well
(see section ).

First we will use a simple weighting method, in which we express the
preference a priori and then minimize the objective .

### Pareto front

We will compute the Pareto front using some randomly chosen weights.
First we define the first problem’s single objectives in `R`:

``` r
# For this one, values are astronomically small, so we cannot pre-condition the problem!
# Another problem is, that the objectives become non-conflicting at some point. For
# example, both have their global minimum at the zero vector. This means, we cannot use
# weights to achieve a trade-off any longer. It actually appears that both objectives
# are kind of mirroring each other.
ex1_f1 <- function(x) {
  l <- length(x)
  1 - exp(-1 * sum((x - 1 / sqrt(l))^2))
}
ex1_f2 <- function(x) {
  l <- length(x)
  1 - exp(-1 * sum((x + 1 / sqrt(l))^2))
}
```

Then the weighted objective:

``` r
ex1_scalarizer <- function(x, w) {
  w[1] * ex1_f1(x) + w[2] * ex1_f2(x)
}
```

Let’s define a wrapper for our optimization routines:

``` r
do_opt <- function(fn, x0 = stats::runif(n = i, min = lb, max = ub), grad = function(x_) pracma::grad(f = fn, x0 = x_), lower = rep(lb, i), upper = rep(ub, i)) {
  nloptr::nloptr(
    x0 = x0,
    opts = list(
      maxeval = 1e3,
      algorithm = "NLOPT_LD_TNEWTON_RESTART"),
    eval_f = fn,
    eval_grad_f = grad,
    lb = lower,
    ub = upper)
}
```

Let’s do some quick tests using various weights. We would want that our
preference is to be found in the solution. So for example, equal weights
should lead to a loss of  ≈ \[0.63,0.63\] for objectives
*f*<sub>1</sub>, *f*<sub>2</sub>.

``` r
set.seed(1337)
temp <- matrix(data = c(1,1, 1,2, 1,3, 1,5, 1,10, 2,1, 3,1, 5,1, 10,1, runif(n = 10)), ncol = 2, byrow = TRUE)

temp1 <- matrix(data = sapply(X = 1:nrow(temp), FUN = function(idx) {
  res <- do_opt(fn = function(x) ex1_scalarizer(x = x, w = temp[idx, ]))
  c(ex1_f1(res$solution), ex1_f2(res$solution))
}), ncol = 2, byrow = TRUE)

temp2 <- matrix(data = sapply(X = 1:nrow(temp), FUN = function(idx) {
  # Return wanted vs. gotten trade-off
  foo <- c(temp[idx, 1] / temp[idx, 2], temp1[idx, 2] / temp1[idx, 1])
  c(foo, abs(foo[1] - foo[2]) / (temp[idx, 1] / temp[idx, 2]))
}), ncol = 3, byrow = TRUE)

temp3 <- `colnames<-`(
  x = cbind(round(temp, 3), round(temp1, 5), round(temp2, 5)),
  value = c("w1", "w2", "loss_f1", "loss_f2", "TO_wanted", "TO_gotten", "TO_diff_perc"))
```

|     w1 |     w2 | loss_f1 | loss_f2 | TO_wanted | TO_gotten | TO_diff_perc |
|-------:|-------:|--------:|--------:|----------:|----------:|-------------:|
|  1.000 |  1.000 |       1 |       1 |   1.00000 |         1 |      0.00000 |
|  1.000 |  2.000 |       1 |       1 |   0.50000 |         1 |      1.00000 |
|  1.000 |  3.000 |       1 |       1 |   0.33333 |         1 |      2.00000 |
|  1.000 |  5.000 |       1 |       1 |   0.20000 |         1 |      4.00000 |
|  1.000 | 10.000 |       1 |       1 |   0.10000 |         1 |      9.00000 |
|  2.000 |  1.000 |       1 |       1 |   2.00000 |         1 |      0.50000 |
|  3.000 |  1.000 |       1 |       1 |   3.00000 |         1 |      0.66667 |
|  5.000 |  1.000 |       1 |       1 |   5.00000 |         1 |      0.80000 |
| 10.000 |  1.000 |       1 |       1 |  10.00000 |         1 |      0.90000 |
|  0.576 |  0.565 |       1 |       1 |   1.02050 |         1 |      0.02009 |
|  0.074 |  0.454 |       1 |       1 |   0.16302 |         1 |      5.13413 |
|  0.373 |  0.331 |       1 |       1 |   1.12665 |         1 |      0.11241 |
|  0.948 |  0.281 |       1 |       1 |   3.37094 |         1 |      0.70335 |
|  0.245 |  0.146 |       1 |       1 |   1.68035 |         1 |      0.40489 |

Desired and actual trade-offs (TO) for the first example problem in its
original form.

So it appears that we cannot actually optimize the problem in its
original form, as the loss of both objectives is always  ≈ 1 (table ).
This is perhaps due to its numerical instability, as perturbations to
the weights result only in abysmal changes. If the changes are too
small, then so is the gradient. With the maximum possible loss (given
random starting values), the trade-off of 1:1 is always the same and the
results are useless. Therefore, we need to re-define the problem in a
more numerically stable way.

We will use the following surrogate functions to map the objectives of
the optimization problem into a more numerically stable range
(log -transform). In pre-conditioning, we want to learn a **bijective**
function that maps a loss to a score, so this is a valid thing to do, as
long as we apply the *same* transformation to *all* objectives.

``` r
ex1_f1_prime <- function(x) {
  l <- length(x)
  1 - log(exp(-1 * sum((x - 1 / sqrt(l))^2)))
}
ex1_f2_prime <- function(x) {
  l <- length(x)
  1 - log(exp(-1 * sum((x + 1 / sqrt(l))^2)))
}
```

``` r
ex1_scalarizer_prime <- function(x, w) {
  w[1] * ex1_f1_prime(x) + w[2] * ex1_f2_prime(x)
}
```

We can now attempt to re-run the previous test using these versions.

``` r
set.seed(1337)
temp <- matrix(data = c(1,1, 1,2, 1,3, 1,5, 1,10, 2,1, 3,1, 5,1, 10,1, runif(n = 10)), ncol = 2, byrow = TRUE)

temp1 <- matrix(data = sapply(X = 1:nrow(temp), FUN = function(idx) {
  res <- do_opt(fn = function(x) ex1_scalarizer_prime(x = x, w = temp[idx, ]))
  c(ex1_f1(res$solution), ex1_f2(res$solution))
}), ncol = 2, byrow = TRUE)

temp2 <- matrix(data = sapply(X = 1:nrow(temp), FUN = function(idx) {
  # Return wanted vs. gotten trade-off
  foo <- c(temp[idx, 1] / temp[idx, 2], temp1[idx, 2] / temp1[idx, 1])
  c(foo, abs(foo[1] - foo[2]) / (temp[idx, 1] / temp[idx, 2]))
}), ncol = 3, byrow = TRUE)

temp3 <- `colnames<-`(
  x = cbind(round(temp, 3), round(temp1, 5), round(temp2, 5)),
  value = c("w1", "w2", "loss_f1", "loss_f2", "TO_wanted", "TO_gotten", "TO_diff_perc"))
```

|     w1 |     w2 | loss_f1 | loss_f2 | TO_wanted | TO_gotten | TO_diff_perc |
|-------:|-------:|--------:|--------:|----------:|----------:|-------------:|
|  1.000 |  1.000 | 0.63212 | 0.63212 |   1.00000 |   1.00000 |      0.00000 |
|  1.000 |  2.000 | 0.83099 | 0.35882 |   0.50000 |   0.43180 |      0.13640 |
|  1.000 |  3.000 | 0.89460 | 0.22120 |   0.33333 |   0.24726 |      0.25822 |
|  1.000 |  5.000 | 0.93782 | 0.10516 |   0.20000 |   0.11213 |      0.43934 |
|  1.000 | 10.000 | 0.96333 | 0.03252 |   0.10000 |   0.03376 |      0.66245 |
|  2.000 |  1.000 | 0.35882 | 0.83099 |   2.00000 |   2.31589 |      0.15794 |
|  3.000 |  1.000 | 0.22120 | 0.89460 |   3.00000 |   4.04432 |      0.34811 |
|  5.000 |  1.000 | 0.10516 | 0.93782 |   5.00000 |   8.91800 |      0.78360 |
| 10.000 |  1.000 | 0.03252 | 0.96333 |  10.00000 |  29.62504 |      1.96250 |
|  0.576 |  0.565 | 0.62462 | 0.63955 |   1.02050 |   1.02391 |      0.00333 |
|  0.074 |  0.454 | 0.94804 | 0.07558 |   0.16302 |   0.07973 |      0.51095 |
|  0.373 |  0.331 | 0.58705 | 0.67459 |   1.12665 |   1.14910 |      0.01993 |
|  0.948 |  0.281 | 0.18890 | 0.90737 |   3.37094 |   4.80334 |      0.42492 |
|  0.245 |  0.146 | 0.42694 | 0.79239 |   1.68035 |   1.85595 |      0.10450 |

Desired and actual trade-offs (TO) for the first example problem, as
transformed to a more numerically stable version.

Now we get some more usable results. Table shows that with a slightly
transformed version of the first example problem, we can achieve
trade-offs, although they are often not close to what we wanted.

The average difference in percent between trade-offs is ≈ 41.52% (with a
maximum deviation of 196.25%), which is perhaps unusable in practice.

In order to approximate the Pareto front, we will simulate a large
number of possible weight constellations. We will compute the problem
for 5, 000 different weight constellations (table ):

``` r
set.seed(1001)
constellations <- 5e3

weight_grid <- data.frame(
  w1 = runif(n = constellations),
  w2 = runif(n = constellations))
```

|        w1 |        w2 |
|----------:|----------:|
| 0.9856888 | 0.0712001 |
| 0.4126285 | 0.9097685 |
| 0.4295392 | 0.3224388 |
| 0.4191722 | 0.5624066 |
| 0.4265066 | 0.5234740 |
| 0.8877976 | 0.5406125 |
| 0.0060960 | 0.8343477 |
| 0.0812158 | 0.6399067 |
| 0.2886574 | 0.4205101 |
| 0.7653421 | 0.3135948 |

Some of the weight constellations used for the first example problem to
approximate the Pareto front (10 out of 5000 are shown).

``` r
res <- loadResultsOrCompute(file = "../results/ex1_pareto_front.rds", computeExpr = {
  res <- as.data.frame(doWithParallelCluster(numCores = 15, expr = {
    library(foreach)
    
    foreach::foreach(
      rn = rownames(weight_grid),
      .combine = rbind,
      .inorder = FALSE
    ) %dopar% {
      w <- as.numeric(weight_grid[rn,])
      set.seed(seed = as.numeric(rn))
      
      res <- nloptr::nloptr(
        x0 = runif(n = i, min = lb, max = ub),
        opts = list(
          maxeval = 2e2,
          algorithm = "NLOPT_LD_TNEWTON_RESTART"),
        eval_f = function(x) ex1_scalarizer_prime(x = x, w = w),
        eval_grad_f = function(x) pracma::grad(f = function(x_) ex1_scalarizer_prime(x = x_, w = w), x0 = x),
        lb = rep(lb, i),
        ub = rep(ub, i))
      
      `colnames<-`(
        x = matrix(data = c(ex1_f1(res$solution), ex1_f2(res$solution), w, res$objective, res$solution), nrow = 1),
        value = c("f1", "f2", "w1", "w2", "value", paste0("par", 1:i)))
    }
  }))
  
  # Next, we have to identify the Pareto front, by selecting non-dominated solutions.
  res$optimal <- FALSE
  for (rn in rownames(res)) {
    this_sol <- res[rn, c("f1", "f2")]
    others <- res[rn != rownames(res), c("f1", "f2")]
    res[rn, ]$optimal <- !any(others$f1 < this_sol$f1 & others$f2 < this_sol$f2)
  }
  res
})
```

Now we show the Pareto front in figure . The optimization found a total
of **5000** Pareto optimal solutions, out of **5000** potential
solutions.

![The Pareto optimal solutions for the first example
problem.](precondition-moop_files/figure-gfm/ex1-pareto-front-1.png)

### Pre-conditioning the problem

The goal is to approximate marginal distribution for each objective.
Each objective expresses a loss, so we will be using the ECCDF  later to
transform it into a score (i.e., low loss equals high score). For this,
we will uniformly sample from the hyperparameter space. For this
problem, we have *i* parameters. The samples’ matrix will therefore have
dimensions *m* × *i*, where *m* is the number of samples we wish to
draw. This number should be  ≪ ∥ ℋ ∥, i.e., much fewer samples than the
hyperparameter space’s size. Here, we will draw 1, 000 samples.

For this, we will create another (but smaller) random grid that will be
used to sample from the solution space of
*f*<sub>1</sub>, *f*<sub>2</sub>.

``` r
set.seed(2002)
constellations <- 1e3

prec_weight_grid <- matrix(data = runif(n = constellations * i, min = lb, max = ub), ncol = i)
```

No we can come up with the empirical marginal distributions.

``` r
# Objective no. 1:
ex1_f1_ecdf_vals <- sapply(X = 1:nrow(prec_weight_grid), FUN = function(idx) {
  ex1_f1_prime(x = as.numeric(prec_weight_grid[idx, ]))
})
ex1_f1_ecdf <- stats::ecdf(ex1_f1_ecdf_vals)
ex1_f1_eccdf <- function(x) 1 - ex1_f1_ecdf(x)


# .. and no. 2:
ex1_f2_ecdf_vals <- sapply(X = 1:nrow(prec_weight_grid), FUN = function(idx) {
  ex1_f2_prime(x = as.numeric(prec_weight_grid[idx, ]))
})
ex1_f2_ecdf <- stats::ecdf(ex1_f2_ecdf_vals)
ex1_f2_eccdf <- function(x) 1 - ex1_f2_ecdf(x)
```

The ECCDF s of the two objectives in figure are clearly non-linear,
indicating that not every loss is equally likely. Since we drew only few
samples, the ECCDF s are not perfectly smooth.

``` r
par(mfrow = c(1, 2))

curve2(func = ex1_f1_eccdf, from = min(ex1_f1_ecdf_vals), to = max(ex1_f1_ecdf_vals), main = "ECCDF of f1", ylab = "f1(x)")
curve2(func = ex1_f2_eccdf, from = min(ex1_f2_ecdf_vals), to = max(ex1_f2_ecdf_vals), main = "ECCDF of f2", ylab = "f2(x)")
```

![The empirical complementary cumulative distribution functions for
losses as sampled from either objective from example
1.](precondition-moop_files/figure-gfm/ex1-eccdfs-1.png)

### Testing the pre-conditioned problem

Now that we have a working transformation from objective values (losses)
to corresponding linear scores, we can test how well optimization works
with these. In order to map a hyperparameter vector to a score, we need
to pass it to the surrogate function, of which we will pass the result
into the corresponding ECCDF .

``` r
ex1_scalarizer_ecdf <- function(x, w = c(1, 1)) {
  s1 <- ex1_f1_eccdf(ex1_f1_prime(x))
  s2 <- ex1_f2_eccdf(ex1_f2_prime(x))
  -1 * (w[1] * s1 + w[2] * s2) # minimization
}

set.seed(1)
res <- do_opt(fn = function(x) {
  ex1_scalarizer_ecdf(x = x)
})
res
```

    ## 
    ## Call:
    ## nloptr::nloptr(x0 = x0, eval_f = fn, eval_grad_f = grad, lb = lower, 
    ##     ub = upper, opts = list(maxeval = 1000, algorithm = "NLOPT_LD_TNEWTON_RESTART"))
    ## 
    ## 
    ## 
    ## Minimization using NLopt version 2.7.1 
    ## 
    ## NLopt solver status: 1 ( NLOPT_SUCCESS: Generic success return value. )
    ## 
    ## Number of Iterations....: 1 
    ## Termination conditions:  maxeval: 1000 
    ## Number of inequality constraints:  0 
    ## Number of equality constraints:    0 
    ## Optimal value of objective function:  -0.691 
    ## Optimal value of controls: -1.875931 -1.023009 0.5828269 3.265662 -2.386545 3.187117 3.557402 1.286382 
    ## 1.032912 -3.50571

``` r
c(ex1_f1(res$solution), ex1_f2(res$solution))
```

    ## [1] 1 1

We notice a problem here. When running this repeatedly (even with
different seeds), the optimizer stops after a single iteration. Also,
the losses are quite large. This indicates that the problem cannot
converge at all. Let’s check the gradients:

``` r
pracma::grad(f = function(x) {
  ex1_scalarizer_ecdf(x = x)
}, x0 = runif(n = i, min = lb, max = ub))
```

    ##  [1] 0 0 0 0 0 0 0 0 0 0

Here is a problem. The gradient is always zero for all variables. Let’s
attempt the following: Instead of estimating the ECDF, we will fit a
parametric distribution. In fact, let’s test the normal distribution,
since the ECDFs looked quite like the CDFs of a normal distribution. In
practice, one would perhaps be better off attempting to fit a wider
range of potential distributions, then going with the best (using, e.g.,
the *D*-statistic of the KS-test).

``` r
ex1_f1_eccdf_norm <- function(x) {
  1 - pnorm(q = x, mean = mean(ex1_f1_ecdf_vals), sd = sd(ex1_f1_ecdf_vals))
}
ex1_f2_eccdf_norm <- function(x) {
  1 - pnorm(q = x, mean = mean(ex1_f2_ecdf_vals), sd = sd(ex1_f2_ecdf_vals))
}
```

The CCDF s of figure look quite similar to the ECCDF s we got
empirically. They do have a two advantages, though. First, smoothness –
we should not end up with all zeros for the gradients. Second, an ECDF 
returns 0 for values less than any of the observed ones, and 1 for
values beyond the observed range. The parametric CDF  however will never
really reach 0 or 1, which means that even tiny differences in *x* will
result in tiny differences in *y*, before and beyond the actually
observed values.

``` r
par(mfrow = c(1, 2))

curve2(func = ex1_f1_eccdf_norm,
       from = min(ex1_f1_ecdf_vals) - 10,
       to = max(ex1_f1_ecdf_vals) + 10,
       main = "ECCDF (normal) of f1", ylab = "f1(x)")

curve2(func = ex1_f2_eccdf_norm,
       from = min(ex1_f2_ecdf_vals) - 10,
       to = max(ex1_f2_ecdf_vals) + 10, main = "ECCDF (normal) of f2", ylab = "f2(x)")
```

![Fitted parametric CCDFs for samples as taken from the two objectives
of example 1.](precondition-moop_files/figure-gfm/ex1-eccdfs-norm-1.png)

So let’s test the gradient again:

``` r
ex1_scalarizer_ecdf_norm <- function(x, w = c(1, 1)) {
  s1 <- ex1_f1_eccdf_norm(ex1_f1_prime(x))
  s2 <- ex1_f2_eccdf_norm(ex1_f2_prime(x))
  -1 * (w[1] * s1 + w[2] * s2) # minimization
}

pracma::grad(f = function(x) {
  ex1_scalarizer_ecdf_norm(x = x)
}, x0 = runif(n = i, min = lb, max = ub))
```

    ##  [1]  0.32030983 -0.20746536  0.11357762 -0.27071066 -0.16723511 -0.08039282
    ##  [7] -0.35263819 -0.08311447  0.27282194 -0.11382062

Much better! Now we are perhaps able to make the gradient-based
optimization work well.

``` r
res <- do_opt(fn = function(x) {
  ex1_scalarizer_ecdf_norm(x = x, w = c(1, 1))
})

print(res)
```

    ## 
    ## Call:
    ## nloptr::nloptr(x0 = x0, eval_f = fn, eval_grad_f = grad, lb = lower, 
    ##     ub = upper, opts = list(maxeval = 1000, algorithm = "NLOPT_LD_TNEWTON_RESTART"))
    ## 
    ## 
    ## 
    ## Minimization using NLopt version 2.7.1 
    ## 
    ## NLopt solver status: 1 ( NLOPT_SUCCESS: Generic success return value. )
    ## 
    ## Number of Iterations....: 97 
    ## Termination conditions:  maxeval: 1000 
    ## Number of inequality constraints:  0 
    ## Number of equality constraints:    0 
    ## Optimal value of objective function:  -1.99913640061011 
    ## Optimal value of controls: 0.0018264 0.001827616 0.00183782 0.001828869 0.001816696 0.00182731 0.001830566 
    ## 0.001827715 0.00184777 0.001826697

``` r
print(c(ex1_f1(res$solution), ex1_f2(res$solution)))
```

    ## [1] 0.6278511 0.6363654

``` r
print(c(ex1_f2_eccdf_norm(ex1_f1_prime(res$solution)), ex1_f2_eccdf_norm(ex1_f2_prime(res$solution))))
```

    ## [1] 0.9995724 0.9995702

This is an excellent result, as we are quite close to the desired 50/50
trade-off for the first time. With equal weights (actually, both  = 1),
the theoretical optimal value of the scalarizer is 2, as we are dealing
now with linear scores that have a range of $\\interval{0}{1}$. The
result shows that we are quite close to it.

As before, we should run a few tests to see if we are able now to get
acceptable trade-offs.

``` r
set.seed(1337)
temp <- matrix(data = c(1,1, 1,2, 1,3, 1,5, 1,10, 2,1, 3,1, 5,1, 10,1, runif(10)), ncol = 2, byrow = TRUE)

temp1 <- matrix(data = sapply(X = 1:nrow(temp), FUN = function(idx) {
  res <- do_opt(fn = function(x) ex1_scalarizer_ecdf_norm(x = x, w = temp[idx, ]))
  c(res$objective / sum(temp[idx, ]), ex1_f1(res$solution), ex1_f2(res$solution))
}), ncol = 3, byrow = TRUE)

temp2 <- matrix(data = sapply(X = 1:nrow(temp), FUN = function(idx) {
  # Return wanted vs. gotten trade-off
  foo <- c(temp[idx, 1] / temp[idx, 2], temp1[idx, 3] / temp1[idx, 2])
  c(foo, abs(foo[1] - foo[2]) / (temp[idx, 1] / temp[idx, 2]))
}), ncol = 3, byrow = TRUE)

temp3 <- `colnames<-`(
  x = cbind(round(temp, 3), round(temp1, 5), round(temp2, 5)),
  value = c("w1", "w2", "res", "loss_f1", "loss_f2", "TO_wanted", "TO_gotten", "TO_diff_perc"))
temp3 <- temp3[order(temp3[, "res"]),]
```

|     w1 |     w2 |      res | loss_f1 | loss_f2 | TO_wanted | TO_gotten | TO_diff_perc |
|-------:|-------:|---------:|--------:|--------:|----------:|----------:|-------------:|
|  1.000 | 10.000 | -0.99962 | 0.94312 | 0.08986 |   0.10000 |   0.09529 |      0.04715 |
| 10.000 |  1.000 | -0.99962 | 0.08595 | 0.94446 |  10.00000 |  10.98820 |      0.09882 |
|  0.074 |  0.454 | -0.99961 | 0.91676 | 0.16404 |   0.16302 |   0.17893 |      0.09758 |
|  1.000 |  5.000 | -0.99960 | 0.90119 | 0.20474 |   0.20000 |   0.22719 |      0.13597 |
|  5.000 |  1.000 | -0.99960 | 0.19778 | 0.90392 |   5.00000 |   4.57030 |      0.08594 |
|  1.000 |  3.000 | -0.99959 | 0.84623 | 0.32902 |   0.33333 |   0.38881 |      0.16643 |
|  3.000 |  1.000 | -0.99959 | 0.32019 | 0.85058 |   3.00000 |   2.65652 |      0.11449 |
|  0.948 |  0.281 | -0.99959 | 0.28970 | 0.86501 |   3.37094 |   2.98586 |      0.11424 |
|  1.000 |  2.000 | -0.99958 | 0.78246 | 0.44296 |   0.50000 |   0.56611 |      0.13222 |
|  2.000 |  1.000 | -0.99958 | 0.43349 | 0.78838 |   2.00000 |   1.81868 |      0.09066 |
|  1.000 |  1.000 | -0.99957 | 0.62787 | 0.63635 |   1.00000 |   1.01351 |      0.01351 |
|  0.576 |  0.565 | -0.99957 | 0.62255 | 0.64157 |   1.02050 |   1.03056 |      0.00986 |
|  0.373 |  0.331 | -0.99957 | 0.59623 | 0.66634 |   1.12665 |   1.11759 |      0.00804 |
|  0.245 |  0.146 | -0.99957 | 0.48371 | 0.75556 |   1.68035 |   1.56202 |      0.07042 |

Desired and actual trade-offs for the pre-conditioned version of the
first example problem.

These results are good (table ). They indicate that, by an average
deviation of only ≈ **8.47%** (with a maximum deviation of 16.64%) from
the desired ratio, we can get very close to the desired trade-off. The
results are ordered best to worst solution, which is another feature of
pre-conditioning (orderable results). It appears that the most extreme
trade-offs are the best solutions. This confirms the conjecture we had
when introducing this problem, that the solution process most likely
jumps to one of the extremes (i.e., losses of \[0,1\] or \[1,0\]).

Those results would have perhaps be even better if we had sampled more
values from either loss, combined with some best-fitting parametric
probability distribution (instead of just assuming the normal
distribution).

### Convergence towards desired trade-off

We want to inspect how exactly the solutions converge from some random
starting point using some preference. For that, we need to track the
history of the process.

``` r
do_opt_hist <- function(fn, x0 = stats::runif(n = i, min = lb, max = ub), grad = function(x_) pracma::grad(f = fn, x0 = x_)) {
  vals <- matrix(ncol = 3 + length(x0), nrow = 0)
  colnames(vals) <- c("loss", "f1", "f2", paste0("par", 1:i))
  
  res <- stats::optim(
    par = x0,
    fn = function(x) {
      r <- fn(x)
      vals <<- rbind(vals, c(r, ex1_f1(x), ex1_f2(x), x))
      r
    },
    gr = grad,
    method = "L-BFGS-B",
    lower = rep(lb, i),
    upper = rep(ub, i)
  )
  res$history <- vals
  res
}
```

``` r
set.seed(1338)
temp <- matrix(data = c(1,1, 1,2, 1,3, 1,5, 1,10, 2,1, 3,1, 5,1, 10,1), ncol = 2, byrow = TRUE)

templ <- list()
for (idx in 1:nrow(temp)) {
  templ[[idx]] <- do_opt_hist(fn = function(x) {
    ex1_scalarizer_ecdf_norm(x = x, w = temp[idx, ])
  })
}
```

Figure demonstrates how pre-conditioning allows us to converge to a
desired trade-off between objectives.

![Working trade-offs using a pre-conditioned version of the first
example
problem.](precondition-moop_files/figure-gfm/ex1-working-tradeoffs-1.png)

It is worth noting that we actually did not reach the 50/50 trade-off of
the numerical transformed problem, but we were close to it. This is very
likely due to the fitted CDF , for which we only observed few values and
which is perhaps not using the most ideal distribution.

### Bonus: Estimate nadir vector

It appears the pre-conditioned problem of the first example works well.
Optimization literature tells us that objectives better be on the same
scale, by dividing it by the extend of utopian- and nadir-vectors. We
will obtain these, and then attempt to obtain desired trade-offs, which
did not work well so far, even with the more numerically stable
surrogate objectives (it did not work at all with the original problem).

We do know that the utopian vector is {0, 0}<sup>⊤</sup>, i.e., a loss
of 0 for either objective. In order to obtain the nadir vector, we can
*maximize* each objective separately. For normalization, we simply
divide by each objective’s nadir value.

``` r
ex1_nadir <- -1 * c(
  do_opt(fn = function(x) -1 * ex1_f1_prime(x = x))$objective,
  do_opt(fn = function(x) -1 * ex1_f2_prime(x = x))$objective)

ex1_nadir
```

    ## [1] 167.0596 172.1193

``` r
set.seed(1339)

ex1_scalarizer_prime_nadir <- function(x, w) {
  w[1] * ex1_f1_prime(x) / ex1_nadir[1] + w[2] * ex1_f2_prime(x) / ex1_nadir[2]
}

temp <- matrix(data = c(1,1, 1,2, 1,3, 1,5, 1,10, 2,1, 3,1, 5,1, 10,1, runif(n = 10)), ncol = 2, byrow = TRUE)

temp1 <- matrix(data = sapply(X = 1:nrow(temp), FUN = function(idx) {
  res <- do_opt(fn = function(x) ex1_scalarizer_prime_nadir(x = x, w = temp[idx, ]))
  c(ex1_f1(res$solution), ex1_f2(res$solution))
}), ncol = 2, byrow = TRUE)

temp2 <- matrix(data = sapply(X = 1:nrow(temp), FUN = function(idx) {
  # Return wanted vs. gotten trade-off
  foo <- c(temp[idx, 1] / temp[idx, 2], temp1[idx, 2] / temp1[idx, 1])
  c(foo, abs(foo[1] - foo[2]) / (temp[idx, 1] / temp[idx, 2]))
}), ncol = 3, byrow = TRUE)

temp3 <- `colnames<-`(
  x = cbind(round(temp, 3), round(temp1, 5), round(temp2, 5)),
  value = c("w1", "w2", "loss_f1", "loss_f2", "TO_wanted", "TO_gotten", "TO_diff_perc"))
```

|     w1 |     w2 | loss_f1 | loss_f2 | TO_wanted | TO_gotten | TO_diff_perc |
|-------:|-------:|--------:|--------:|----------:|----------:|-------------:|
|  1.000 |  1.000 | 0.62106 | 0.64301 |   1.00000 |   1.03534 |      0.03534 |
|  1.000 |  2.000 | 0.82490 | 0.37022 |   0.50000 |   0.44881 |      0.10238 |
|  1.000 |  3.000 | 0.89099 | 0.23003 |   0.33333 |   0.25817 |      0.22549 |
|  1.000 |  5.000 | 0.93607 | 0.11020 |   0.20000 |   0.11773 |      0.41135 |
|  1.000 | 10.000 | 0.96266 | 0.03430 |   0.10000 |   0.03563 |      0.64373 |
|  2.000 |  1.000 | 0.34755 | 0.83686 |   2.00000 |   2.40785 |      0.20393 |
|  3.000 |  1.000 | 0.21260 | 0.89807 |   3.00000 |   4.22422 |      0.40807 |
|  5.000 |  1.000 | 0.10031 | 0.93951 |   5.00000 |   9.36577 |      0.87315 |
| 10.000 |  1.000 | 0.03082 | 0.96397 |  10.00000 |  31.27258 |      2.12726 |
|  0.614 |  0.740 | 0.68756 | 0.57216 |   0.82916 |   0.83215 |      0.00360 |
|  0.927 |  0.468 | 0.35083 | 0.83516 |   1.98264 |   2.38054 |      0.20069 |
|  0.280 |  0.873 | 0.89556 | 0.21883 |   0.32091 |   0.24435 |      0.23858 |
|  0.764 |  0.407 | 0.37158 | 0.82417 |   1.87753 |   2.21804 |      0.18136 |
|  0.559 |  0.846 | 0.75730 | 0.48119 |   0.66077 |   0.63541 |      0.03838 |

Desired and actual trade-offs (TO) for the first example problem, as
transformed to a more numerically stable version and using normalized
objectives as of the nadir-vector.

The results are shown in table . The mean deviation is slightly better
with ≈ **40.67%**, while the maximum deviation of 212.73% is worse. What
we clearly observe here, however, is that the more extreme the desired
trade-off, the bigger the difference between it and the actual
trade-off. This did not exactly hold for the pre-conditioned problem,
either.

``` r
set.seed(1338)
temp <- matrix(data = c(1,1, 1,2, 1,3, 1,5, 1,10, 2,1, 3,1, 5,1, 10,1), ncol = 2, byrow = TRUE)

templ <- list()
for (idx in 1:nrow(temp)) {
  templ[[idx]] <- do_opt_hist(fn = function(x) {
    ex1_scalarizer_prime_nadir(x = x, w = temp[idx, ])
  })
}
```

Figure shows the trade-offs as reachable by the normalized objectives,
using the numerically stabilized version of the problem. While it might
does not look so bad, recall that on average, each trade-off is off by ≈
40.67%. It also appears that each optimization went straight to the
solution. In the pre-conditioned version (figure ), convergence was not
straight, and one could clearly observe that the direction adapted and
improved towards the solution with the desired trade-off.

![Pareto trade-offs using the numerically stabilized version of the
first problem, together with normalized
objectives.](precondition-moop_files/figure-gfm/ex1-nadir-tradeoffs-1.png)

### Absolute best solution

There is one thing left to do. Not only do we want to find the
hyperparameters that maximize the scores, but we will now simultaneously
attempt to learn the weights using a weighted mean.

``` r
set.seed(7)
do_opt(fn = function(x) {
  w <- tail(x, 2)
  x <- head(x, i)
  -1 * (w[1] * ex1_f1_eccdf_norm(ex1_f1_prime(x)) + w[2] * ex1_f2_eccdf_norm(ex1_f2_prime(x))) / sum(w)
}, x0 = c(runif(n = i, min = lb, max = ub), runif(2)),
    lower = c(rep(lb, i), .Machine$double.eps, .Machine$double.eps), # do not allow 0
    upper = c(rep(ub, i), 1, 1)) # + weights
```

    ## 
    ## Call:
    ## nloptr::nloptr(x0 = x0, eval_f = fn, eval_grad_f = grad, lb = lower, 
    ##     ub = upper, opts = list(maxeval = 1000, algorithm = "NLOPT_LD_TNEWTON_RESTART"))
    ## 
    ## 
    ## 
    ## Minimization using NLopt version 2.7.1 
    ## 
    ## NLopt solver status: 1 ( NLOPT_SUCCESS: Generic success return value. )
    ## 
    ## Number of Iterations....: 153 
    ## Termination conditions:  maxeval: 1000 
    ## Number of inequality constraints:  0 
    ## Number of equality constraints:    0 
    ## Optimal value of objective function:  -0.999658878024273 
    ## Optimal value of controls: -0.3161922 -0.3162715 -0.3162628 -0.3162288 -0.3162624 -0.3161903 -0.3161934 
    ## -0.3162287 -0.3162651 -0.3162709 2.220446e-16 1

Now we can say with certainty that there are two equally absolute best
solutions for this problem, where one favors the first objective (the
second having no weight) and one favors the second objective. Using the
seed of 1 leads to weights \[1,0\], and using a seed of 7 leads to
weights \[0,1\]. No matter the seed, one weight is always (practically)
0, while the other is  \> 0. In a weighted mean, it does not matter by
how much the one weight is greater than zero, just that it is (as
normalization of a zero-weight will not make it larger).

# References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-fonseca1995" class="csl-entry">

Fonseca, Carlos M., and Peter J. Fleming. 1995. “An Overview of
Evolutionary Algorithms in Multiobjective Optimization.” *Evol. Comput.*
3 (1): 1–16. <https://doi.org/10.1162/evco.1995.3.1.1>.

</div>

<div id="ref-miettinen2008" class="csl-entry">

Miettinen, Kaisa. 2008. “Introduction to Multiobjective Optimization:
Noninteractive Approaches.” In *Multiobjective Optimization, Interactive
and Evolutionary Approaches \[Outcome of Dagstuhl Seminars\]*, edited by
Jürgen Branke, Kalyanmoy Deb, Kaisa Miettinen, and Roman Slowinski,
5252:1–26. Lecture Notes in Computer Science. Springer.
[https://doi.org/10.1007/978-3-540-88908-3\\\_1](https://doi.org/10.1007/978-3-540-88908-3\_1).

</div>

<div id="ref-roy1996theoretical" class="csl-entry">

Roy, Bernard, and Vincent Mousseau. 1996. “A Theoretical Framework for
Analysing the Notion of Relative Importance of Criteria.” *Journal of
Multi-Criteria Decision Analysis* 5 (2): 145–59.

</div>

</div>

[1] <https://en.wikipedia.org/wiki/Test_functions_for_optimization>
