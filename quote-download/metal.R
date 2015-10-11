library(quantmod)
#XAU (gold) , XAG (silver), XPD (palladium), or XPT (platinum)
metals = c("XAU", "XAG", "XPD", "XPT")
for(t in metals) {
  cat(paste("Downloading", t, "\n"))
  getMetals(t)
  eval(parse(text=paste('data <- ', t, "USD", sep='')))
  write.zoo(data, file=paste(t, '.csv', sep=''))
}

