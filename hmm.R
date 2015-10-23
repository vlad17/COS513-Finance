
#----------------------------------------------------------------
# Hiden Markov Model of S&P 500 log returns
# See documentation for depmixS4 package 
# http://cran.r-project.org/web/packages/depmixS4/depmixS4.pdf and presentation 
# on Singapore R Users Group Site on HMM February 14, 2014
# http://www.meetup.com/R-User-Group-SG/files/

library(depmixS4)
library(TTR)
library(ggplot2)
library(reshape2)
library(xts)

options(echo=TRUE) # if you want see commands in output file
args <- commandArgs(trailingOnly = TRUE)
print(args)

#setwd("~/Documents/Princeton/COS513")

hmmplot <- function(datafilename, output, tick, n=14, numstates=2) {
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

  # Construct and fit a regime switching model
  mod <- depmix(list(ret ~ 1, retmva ~ 1), 
                family =list(gaussian(), gaussian()), 
                nstates = numstates, data = datadf)
  set.seed(1)
  fm2 <- fit(mod, verbose = FALSE)
  # Classification (inference task)
  probs <- posterior(fm2) # Compute probability of being in each state
  write.csv(probs, output)
  plotting_function <- function() {
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
    return(q)
  }
}

datafilename <-args[1]
output <- args[2]
hmmplot(datafilename, output)


