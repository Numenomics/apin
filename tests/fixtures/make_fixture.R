# Regenerates tiny/: four chains of synthetic draws and the values R's
# posterior package gives for them. Run: Rscript tests/fixtures/make_fixture.R
suppressMessages({library(posterior); library(jsonlite)})
set.seed(7)
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
n <- 101; chains <- 4
ar <- function(phi, mu) { x <- numeric(n); x[1] <- rnorm(1); for (i in 2:n) x[i] <- phi * x[i-1] + rnorm(1); x + mu }
arr <- array(NA, dim = c(n, chains, 4), dimnames = list(NULL, NULL, c("a", "b.1", "b.2", "c")))
for (ch in 1:chains) {
  arr[, ch, 1] <- ar(0.9, 0)
  arr[, ch, 2] <- rnorm(n)
  arr[, ch, 3] <- ar(0.5, ch * 0.3)       # chains disagree: R-hat > 1
  arr[, ch, 4] <- rexp(n)
  df <- data.frame(arr[, ch, ], lp__ = rnorm(n), n_leapfrog__ = 7, check.names = FALSE)
  write.csv(df, file.path(here, "tiny", sprintf("chain_%d.csv", ch - 1)), row.names = FALSE, quote = FALSE)
}
# Score the files as written (write.csv keeps 15 digits), so Python and R see the same numbers.
back <- arr
for (ch in 1:chains) back[, ch, ] <- as.matrix(read.csv(file.path(here, "tiny", sprintf("chain_%d.csv", ch - 1)),
                                                       check.names = FALSE)[, dimnames(arr)[[3]]])
s <- summarise_draws(as_draws_array(back), "ess_bulk", "ess_tail", "rhat")
write(toJSON(list(variable = s$variable, ess_bulk = s$ess_bulk, ess_tail = s$ess_tail, rhat = s$rhat,
                  posterior_version = as.character(packageVersion("posterior"))), digits = NA),
      file.path(here, "tiny_expected.json"))
