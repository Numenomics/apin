# Invariance Argument (Involutive-MCMC Format)

## State space

$\mathcal{X} = \mathbb{R}^d$ equipped with target density $\pi(x) \propto \exp(\log p(x))$ with respect to Lebesgue measure.

## Aux variables and extended target

`aux_spec = {"kind": "none"}`. No persistent auxiliary state enters the transition distribution; the proposal innovation $u \sim \mathcal{N}(0, I_d)$ is drawn fresh each step and discarded.

## Involution and composition

Given $(x, u) \in \mathbb{R}^d \times \mathbb{R}^d$, define the symmetric map $T(x, u) = (x + \sigma u, -u)$ with fixed step size $\sigma > 0$ frozen after `adapt()`. Clearly $T(T(x, u)) = (x, u)$ and $|\det D T(x, u)| = 1$. Accepting $T(x, u)$ with Metropolis probability $\min(1, \pi(x + \sigma u) / \pi(x))$ leaves $\pi(x)\,\mathcal{N}(u; 0, I_d)$ invariant by detailed balance; composing independent steps preserves $\pi$.
