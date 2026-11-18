# Data sources

Every data set comes from posteriordb at commit 28f8d3d6e975315f42aa274a8399f21e07a43b30 (`posterior_database/data/data/<name>.json.zip`). Its description is in `posterior_database/data/info/<name>.info.json` at the same commit. `models/manifest.json` records the sha256 of each unzipped JSON file, and the harness checks it.

No posteriordb data info file has a licence field. apin ships a data set only when it is simulated, or is a short historical table (published in 1921 for the lynx and hare data, in 1981 for the eight schools data), or is the LSAT table from the WinBUGS examples. apin fetches the data sets taken from books, surveys and papers (`apin fetch-data`) and does not redistribute them. This split is a proposal. The owner must decide whether it is acceptable or whether all data must be fetched (see LICENSING.md).

| data set | models | origin (from the posteriordb info file) | in apin |
|---|---|---|---|
| eight_schools | eight_schools_centered, eight_schools_noncentered | Rubin (1981); Gelman et al., Bayesian Data Analysis | shipped |
| sblrc | blr | simulated: 100 observations, 5 correlated coefficients | shipped |
| lsat_data | lsat_model | LSAT response patterns, WinBUGS examples vol. 1 | shipped |
| gp_pois_regr | gp_regr | simulated, stan-dev/stat_comp_benchmarks | shipped |
| synthetic_grid_RBF_kernels | kronecker_gp | simulated GP on a grid, after Flaxman et al. (2015) | shipped |
| garch | garch11 | simulated GARCH(1,1), stat_comp_benchmarks | shipped |
| hudson_lynx_hare | lotka_volterra | Hudson's Bay Company pelts 1900 to 1920; Hewitt (1921), Howard (2009) | shipped |
| low_dim_gauss_mix | low_dim_gauss_mix | simulated, stat_comp_benchmarks | shipped |
| arma | arma11 | simulated ARMA(1,1), stat_comp_benchmarks | shipped |
| kidiq | kidscore_momiq | Gelman and Hill (2006), ARM chapter 3 | fetched |
| mesquite | logmesquite_logvash | Gelman and Hill (2006), ARM chapter 4 | fetched |
| wells_data | wells_dist100_model | Gelman and Hill (2006), ARM chapter 5 (Bangladesh wells) | fetched |
| diamonds | diamonds | ggplot2 diamonds data through brms::make_standata | fetched |
| radon_all | radon_partially_pooled_noncentered | Gelman and Hill (2006), radon, 386 counties | fetched |
| radon_mn | radon_variable_intercept_slope_noncentered | Gelman and Hill (2006), radon, Minnesota subset | fetched |
| dogs | dogs_hierarchical | Gelman and Hill (2006), dog shock study | fetched |
| pilots | pilots | Gelman and Hill (2006), ARM chapter 14 | fetched |
| sat | hier_2pl | grade 12 science assessment (Wood et al. 2003, via the mirt R package) | fetched |
| mcycle_gp | accel_gp | MASS mcycle data through brms::make_standata | fetched |
| traffic_accident_nyc | bym2_offset_only | NYC traffic accidents, DiMaggio (2015) | fetched |

Fetched files land in `.cache/data/` (or `$APIN_DATA_CACHE`). With the `upstream/posteriordb` submodule checked out, `apin fetch-data` reads them from the checkout and does not use the network.
