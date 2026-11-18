Gate error rates by simulation: tolerance 0.1 SD, alpha_wrong 0.01 per model, alpha_right 0.05 per rep, R-hat < 1.01, ESS >= 400. 100 runs of 4 chains per row, a new reference every 20 runs. One rep per verdict.

| rho | draws per chain | ESS about | sampler | right | unproven | wrong |
|---|---|---|---|---|---|---|
| 0.0 | 1000 | 4000 | correct | 0.900 | 0.100 | 0.000 |
| 0.0 | 1000 | 4000 | error = tol | 0.040 | 0.960 | 0.000 |
| 0.0 | 1000 | 4000 | error = 2 tol | 0.000 | 0.060 | 0.940 |
| 0.0 | 5000 | 20000 | correct | 1.000 | 0.000 | 0.000 |
| 0.0 | 5000 | 20000 | error = tol | 0.000 | 1.000 | 0.000 |
| 0.0 | 5000 | 20000 | error = 2 tol | 0.000 | 0.000 | 1.000 |
| 0.5 | 1000 | 1333 | correct | 0.380 | 0.620 | 0.000 |
| 0.5 | 1000 | 1333 | error = tol | 0.020 | 0.980 | 0.000 |
| 0.5 | 1000 | 1333 | error = 2 tol | 0.000 | 0.510 | 0.490 |
| 0.5 | 15000 | 20000 | correct | 1.000 | 0.000 | 0.000 |
| 0.5 | 15000 | 20000 | error = tol | 0.000 | 1.000 | 0.000 |
| 0.5 | 15000 | 20000 | error = 2 tol | 0.000 | 0.000 | 1.000 |
| 0.9 | 1000 | 211 | correct | 0.000 | 1.000 | 0.000 |
| 0.9 | 1000 | 211 | error = tol | 0.000 | 1.000 | 0.000 |
| 0.9 | 1000 | 211 | error = 2 tol | 0.000 | 1.000 | 0.000 |
| 0.9 | 20000 | 4211 | correct | 1.000 | 0.000 | 0.000 |
| 0.9 | 20000 | 4211 | error = tol | 0.010 | 0.990 | 0.000 |
| 0.9 | 20000 | 4211 | error = 2 tol | 0.000 | 0.060 | 0.940 |
| 0.9 | 50000 | 10526 | correct | 1.000 | 0.000 | 0.000 |
| 0.9 | 50000 | 10526 | error = tol | 0.000 | 1.000 | 0.000 |
| 0.9 | 50000 | 10526 | error = 2 tol | 0.000 | 0.000 | 1.000 |
| 0.99 | 1000 | 20 | correct | 0.000 | 1.000 | 0.000 |
| 0.99 | 1000 | 20 | error = tol | 0.000 | 1.000 | 0.000 |
| 0.99 | 1000 | 20 | error = 2 tol | 0.000 | 1.000 | 0.000 |
| 0.99 | 50000 | 1005 | correct | 0.200 | 0.800 | 0.000 |
| 0.99 | 50000 | 1005 | error = tol | 0.010 | 0.990 | 0.000 |
| 0.99 | 50000 | 1005 | error = 2 tol | 0.000 | 0.670 | 0.330 |
| 0.999 | 1000 | 2 | correct | 0.000 | 1.000 | 0.000 |
| 0.999 | 1000 | 2 | error = tol | 0.000 | 1.000 | 0.000 |
| 0.999 | 1000 | 2 | error = 2 tol | 0.000 | 1.000 | 0.000 |
| 0.999 | 50000 | 100 | correct | 0.000 | 1.000 | 0.000 |
| 0.999 | 50000 | 100 | error = tol | 0.000 | 1.000 | 0.000 |
| 0.999 | 50000 | 100 | error = 2 tol | 0.000 | 1.000 | 0.000 |

Calibration test, groups of 5 reps (100 groups per row):

| rho | draws per chain | sampler | right | unproven | wrong | median ratio |
|---|---|---|---|---|---|---|
| 0.0 | 5000 | correct | 1.000 | 0.000 | 0.000 | 0.96 |
| 0.0 | 5000 | copies | 0.250 | 0.750 | 0.000 | 1.96 |
| 0.5 | 15000 | correct | 1.000 | 0.000 | 0.000 | 1.02 |
| 0.5 | 15000 | copies | 0.210 | 0.790 | 0.000 | 2.01 |

Halves of the reference: quantities disputed by chance (SPLIT_Z 5.0), 200 references of 10 chains x 1000 draws per row:

| rho of the reference chains | disputed | of quantities | share |
|---|---|---|---|
| 0.0 | 9 | 8800 | 0.1023% |
| 0.5 | 17 | 8800 | 0.1932% |
| 0.9 | 54 | 8800 | 0.6136% |

Correct runs against references that may miss rare excursions:

| case | rule | right | unproven | wrong |
|---|---|---|---|---|
| projection | none | 106 | 9 | 5 |
| projection | SPLIT_Z 5.0 | 106 | 14 | 0 |
| variable | none | 18 | 129 | 3 |
| variable | SPLIT_Z 5.0 | 18 | 129 | 3 |

Calibration test under an idealised Gaussian null with the correlations of a shipped reference (calibration_null): groups of 5 reps, every rep's ESS 20000, cutoff p < 0.001. Only groups in which every rep is right are tested.

| reference | quantities | tr(C^2) | scale of the correlations | groups | all reps right | refused | share |
|---|---|---|---|---|---|---|---|
| eight_schools_noncentered | 84 | 331.66 | 0.992 | 100000 | 99951 | 477 | 0.477% |
| hier_2pl | 2716 | 4725.18 | 0.589 | 20000 | 20000 | 31 | 0.155% |
