doWithParallelCluster <- function(expr, errorValue = NULL, numCores = parallel::detectCores()) {
  cl <- parallel::makePSOCKcluster(numCores)
  doSNOW::registerDoSNOW(cl)
  mev <- missing(errorValue)
  
  result <- tryCatch(expr, error=function(cond) {
    if (!mev) {
      return(errorValue)
    }
    return(cond)
  }, finally = {
    parallel::stopCluster(cl)
    foreach::registerDoSEQ()
    cl <- NULL
    gc()
  })
  return(result)
}

loadResultsOrCompute <- function(file, computeExpr) {
  file <- base::normalizePath(file, mustWork = FALSE)
  if (file.exists(file)) {
    return(base::readRDS(file))
  }
  
  res <- base::tryCatch(
    expr = computeExpr, error = function(cond) cond)
  
  # 'res' may have more than one class.
  if (any(class(res) %in% c("simpleError", "error", "condition"))) {
    print(traceback())
    stop(paste0("The computation failed: ", res))
  }
  
  base::saveRDS(res, file)
  return(res)
}

curve2 <- function(func, from, to, col = "black", lty = 1, lwd = 1, add = FALSE, xlab = NULL, ylab = NULL, xlim = NULL, ylim = NULL, main = NULL, ...) {
  f <- function(x) func(x)
  curve(expr = f, from = from, to = to, col = col, lty = lty, lwd = lwd, add = add, xlab = xlab, ylab = ylab, xlim = xlim, ylim = ylim, main = main, ... = ...)
}

