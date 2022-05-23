library(R6)

StandardScaler <- R6::R6Class(
  classname = "StandardScaler",
  
  private = list(
    wasFit = NULL,
    
    requireFitted = function() {
      if (!private$wasFit) {
        stop("This scaler was not previously fit.")
      }
    }
  ),
  
  public = list(
    center = NULL, # mean
    scale = NULL, # sd
    
    initialize = function() {
      private$wasFit <- FALSE
    },
    
    fit_transform = function(data) {
      self$fit(data)$transform(data)
    },
    
    fit = function(data) {
      if (private$wasFit) {
        stop("This scaler was not previously fit. Do not reuse scalers.")
      }
      
      s <- scale(x = data, center = TRUE, scale = TRUE)
      self$center <- attr(s, "scaled:center")
      self$scale <- attr(s, "scaled:scale")
      private$wasFit <- TRUE
      invisible(self)
    },
    
    transform = function(data) {
      private$requireFitted()
      
      (data - self$center) / self$scale
    },
    
    inverse_transform = function(data) {
      private$requireFitted()
      
      scale(
        scale(data, center = FALSE, scale = 1 / self$scale),
        center = -1 * self$center, scale = FALSE)
    }
  )
)


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

doWithParallelClusterExplicit <- function(cl, expr, errorValue = NULL, stopCl = TRUE) {
  doSNOW::registerDoSNOW(cl = cl)
  mev <- missing(errorValue)
  
  tryCatch(expr, error = function(cond) {
    if (!mev) {
      return(errorValue)
    }
    return(cond)
  }, finally = {
    if (stopCl) {
      parallel::stopCluster(cl)
      foreach::registerDoSEQ()
      gc()
    }
  })
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

