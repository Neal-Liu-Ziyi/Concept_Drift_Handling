set.seed(1)
 
generate_random_arma_parameters <- function(lags, maxRoot)  {
  
  if(maxRoot <= 1.1) stop("maxRoot has to be bigger than 1.1")
  
  l <- lags
  s <- sign(runif(l,-1,1))
  
  # the AR process is stationary if the absolute value of all the roots of the characteristic polynomial are greater than 1
  polyRoots <- s*runif(l,1.1,maxRoot)
  
  #calculate coefficients from the roots of the characteristic polynomial
  coeff <- 1
  for(root in polyRoots) coeff <- c(0,coeff) - c(root*coeff,0)
  
  nCoeff <- coeff / coeff[1]
  params <- - nCoeff[2:length(nCoeff)]
  
  return(params)
}