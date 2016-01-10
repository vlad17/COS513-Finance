library(quantmod)
#XAU (gold) , XAG (silver), XPD (palladium), or XPT (platinum)
metals = c("XAU", "XAG", "XPD", "XPT")
for(t in metals) {
  cat(paste("Downloading", t, "\n"))
  getMetals(t, from="2004-01-01", to='2015-11-01')
  eval(parse(text=paste('data <- ', t, "USD", sep='')))
  write.zoo(data, file=paste(t, '.csv', sep=''))
}

