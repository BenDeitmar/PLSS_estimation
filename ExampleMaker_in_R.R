## ---------------------------------------------
## Utilities (no tensorflow; cached MNIST)
## ---------------------------------------------

pullHaarOd <- function(d) {
  X <- matrix(rnorm(d * d), nrow = d, ncol = d)
  eigen(X %*% t(X), symmetric = TRUE)$vectors
}

# Cached MNIST loader (downloads once to ~/.keras/datasets/, then reuses)
getMNIST_data <- function(d) {
  cache_dir <- file.path(path.expand("~"), ".keras", "datasets")
  if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  
  base_url <- "https://storage.googleapis.com/cvdf-datasets/mnist/"
  files <- c(
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
  )
  paths <- file.path(cache_dir, files)
  
  # Download missing files once (quietly)
  for (i in seq_along(files)) {
    if (!file.exists(paths[i])) {
      download.file(paste0(base_url, files[i]), destfile = paths[i], mode = "wb", quiet = TRUE)
    }
  }
  
  read_idx_images <- function(fname) {
    con <- gzfile(fname, "rb")
    on.exit(close(con))
    magic <- readBin(con, "integer", n = 1, size = 4, endian = "big")
    n     <- readBin(con, "integer", n = 1, size = 4, endian = "big")
    nrow  <- readBin(con, "integer", n = 1, size = 4, endian = "big")
    ncol  <- readBin(con, "integer", n = 1, size = 4, endian = "big")
    x     <- readBin(con, "integer", n = n * nrow * ncol, size = 1, signed = FALSE)
    matrix(x, nrow = nrow * ncol, ncol = n)  # (784, n)
  }
  
  train <- read_idx_images(paths[1])
  test  <- read_idx_images(paths[3])
  data_full <- cbind(train, test) / 255  # (784, 70000)
  
  if (d > nrow(data_full)) stop("d must be <= 784 for MNIST.")
  dims_chosen <- sample.int(nrow(data_full), size = d, replace = FALSE)
  
  list(
    data = data_full[dims_chosen, , drop = FALSE],  # (d, 70000)
    dims_chosen = dims_chosen
  )
}

PopEV_Y_maker <- function(d, n, ExampleNumber) {
  if (ExampleNumber == 1) {
    k <- d %/% 2
    PopEV <- c(rep(0.5, k), rep(1.0, d - k))
    Y <- diag(sqrt(PopEV), nrow = d, ncol = d) %*% matrix(rnorm(d * n), nrow = d, ncol = n)
    return(list(PopEV = PopEV, Y = Y))
  }
  
  if (ExampleNumber == 2) {
    k <- d %/% 2
    PopEV <- c(rep(0.5, k), 0.5 + (0:(d - k - 1)) / d)
    H <- pullHaarOd(d)
    Rademacher <- matrix(sample(c(-1, 1), size = d * n, replace = TRUE), nrow = d, ncol = n)
    Y <- H %*% diag(sqrt(PopEV), nrow = d, ncol = d) %*% Rademacher
    return(list(PopEV = PopEV, Y = Y))
  }
  
  if (ExampleNumber == 3) {
    mn <- getMNIST_data(d)
    data <- mn$data
    dims_chosen <- mn$dims_chosen
    N <- ncol(data)
    
    # Row-wise centering (safe, no recycling)
    X_full <- sweep(data, 1, rowMeans(data), "-")
    
    # Surrogate covariance eigenvalues (ascending, like numpy.linalg.eigh)
    Sigma_surrogate <- (X_full %*% t(X_full)) / N
    ev <- eigen(Sigma_surrogate, symmetric = TRUE, only.values = TRUE)$values
    PopEV_surrogate <- sort(ev, decreasing = FALSE)
    
    if (n > N) stop("n must be <= number of MNIST images (70000).")
    cols_chosen <- sample.int(N, size = n, replace = FALSE)
    Y <- X_full[, cols_chosen, drop = FALSE]
    
    return(list(
      PopEV_surrogate = PopEV_surrogate,
      Y = Y,
      Sigma = Sigma_surrogate,
      dims_chosen = dims_chosen,  # selected pixel rows
      cols_chosen = cols_chosen   # selected image columns
    ))
  }
  
  stop("ExampleNumber must be 1, 2, or 3.")
}

## ----------------------
## Example: quick sanity
## ----------------------
## set.seed(123)
##out3 <- PopEV_Y_maker(d = 50, n = 200, ExampleNumber = 3)
##str(out3$dims_chosen)
## str(out3$cols_chosen)
