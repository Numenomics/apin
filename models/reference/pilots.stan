// The reference posterior for models/pilots.stan is built from this file.
// It is the same model with a and b non-centred, so it has the same posterior
// for a, b, mu_a, mu_b and the three scales, and NUTS samples it without the
// divergences the centred funnel causes. a and b become transformed
// parameters with the original names, so the reference keeps those names.
data {
  int<lower=0> N;
  int<lower=0> n_groups;
  int<lower=0> n_scenarios;
  array[N] int<lower=1, upper=n_groups> group_id;
  array[N] int<lower=1, upper=n_scenarios> scenario_id;
  vector[N] y;
}
parameters {
  vector[n_groups] a_raw;
  vector[n_scenarios] b_raw;
  real mu_a;
  real mu_b;
  real<lower=0, upper=100> sigma_a;
  real<lower=0, upper=100> sigma_b;
  real<lower=0, upper=100> sigma_y;
}
transformed parameters {
  vector[n_groups] a = 10 * mu_a + sigma_a * a_raw;
  vector[n_scenarios] b = 10 * mu_b + sigma_b * b_raw;
}
model {
  mu_a ~ normal(0, 1);
  a_raw ~ std_normal();
  mu_b ~ normal(0, 1);
  b_raw ~ std_normal();
  y ~ normal(a[group_id] + b[scenario_id], sigma_y);
}
