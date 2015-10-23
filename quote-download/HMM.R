library(depmixS4)
library(TTR)
library(ggplot2)
library(reshape2)
library(xts)

hmmplot <- function(datafilename, imagefilename, tick, n=14) {
  data <- read.csv(datafilename, sep=' ')
  price <- data[,2]
  price <- xts(price, order.by=as.Date(data[,1]))
  
  # Build a data frame for ggplot
  datadf <- data.frame(price)
  datadf$ret <- coredata(log(price) - log(lag(price)))[,1]
  datadf$date <-as.Date(index(price))
  datadf$mva <- EMA(datadf$price, n)
  datadf$mva[1:n] = datadf$mva[n+1]
  datadf$retmva <- c(0, diff(log(datadf$mva)))
  
  # Plot the S&P 500 returns
  ggplot( datadf, aes(date) ) + geom_line( aes( y = ret ) ) + labs( title = "S&P 500 log Returns")
  
  # Construct and fit a regime switching model
  mod <- depmix(list(ret ~ 1, retmva ~ 1), 
                family =list(gaussian(), gaussian()), 
                nstates = 2, data = datadf)
  set.seed(1)
  fm2 <- fit(mod, verbose = FALSE)
  #
  summary(fm2)
  print(fm2)
  
  # Classification (inference task)
  probs <- posterior(fm2) # Compute probability of being in each state
  
  # Plot 
  datadf$states <- apply(probs[,-1], 1, function(x) which.max(x))
  xstart <- which(diff(datadf$states) != 0)
  xend <- xstart-1
  xstart <- c(1, xstart[-length(xstart)])
  rects <- data.frame(minprice = min(datadf$price), maxprice =max(datadf$price),
                      xstart=datadf$date[xstart],
                      xend=datadf$date[xend], col=datadf$states[xstart])
  q <- ggplot() + geom_line(data=datadf, aes(x=date, y=price)) +
              geom_rect(data=rects, 
              aes(ymin=0.99 * minprice, ymax=1.1 * maxprice, xmin=xstart, xmax=xend, fill=factor(col)), 
              alpha=0.2) 
  q <- q + labs(title = paste(tick, "n=", n))
  ggsave(file=imagefilename)
}

metals = c("XAU", "XAG", "XPT", "XPT")
for(t in metals) {
  hmmplot(paste(t, ".csv", sep=''), paste(t, ".png", sep=''), t)
}
