ex2_tradeoff_metrics <- function(to_wanted, to_gotten) {
  # Normalize both
  v1 <- as.numeric(to_wanted / max(to_wanted))
  v2 <- as.numeric(to_gotten / max(to_gotten))
  stopifnot(length(v1) == length(v2))
  
  `colnames<-`(x = matrix(
    nrow = 1,
    data = c(
      Metrics::mae(actual = v2, predicted = v1),
      Metrics::rmse(actual = v2, predicted = v1),
      abs(100 * (1 - v1/v2)),
      v2 - v1)
  ), value = c("MAE", "RMSE", paste0("PERC_DIFF_", 1:length(v1)), paste0("DEVIATION_", 1:length(v1))))
}


ex2_tradeoff_metrics_all <- function(to_wanted, to_gotten) {
  m <- NULL
  df <- NULL
  for (i in 1:nrow(to_wanted)) {
    temp <- ex2_tradeoff_metrics(to_wanted = to_wanted[i,], to_gotten = to_gotten[i,])
    if (i == 1) {
      m <- matrix(nrow = nrow(to_wanted), ncol = ncol(temp))
      colnames(m) <- colnames(temp)
    }
    m[i,] <- temp
  }
  df <- as.data.frame(m)
  df[df <= -Inf | df >= Inf] <- NA_real_
  df
}

make_smooth_ecdf <- function(values, slope = 0.025) {
  r <- range(values)
  e <- stats::ecdf(values)
  x <- sort(unique(values))
  y <- e(x)
  if (slope > 0) {
    ext <- r[2] - r[1]
    # Add a sllight slope before and after for numeric stability.
    x <- c(r[1] - ext, x, r[2] + ext)
    y <- c(0 - slope, y, 1 + slope)
  }
  `attributes<-`(x = stats::approxfun(x = x, y = y, yleft = y[1], yright = y[length(y)]), value = list(
    "min" = min(values),
    "max" = max(values),
    "range" = range(values),
    
    "slope_min" = min(x),
    "slope_max" = max(x),
    "slope_range" = range(x)
  ))
}

make_inverse_ecdf <- function(values, inverse_eccdf = FALSE) {
  e <- ecdf(values)
  y <- e(values)
  if (inverse_eccdf) {
    y <- 1 - y
  }
  stats::approxfun(x = y, y = values, yleft = if (inverse_eccdf) max(values) else min(values), yright = if (inverse_eccdf) min(values) else max(values))
}
