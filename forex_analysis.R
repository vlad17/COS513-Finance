library(ggplot2)
library(dplyr)
library(tidyr)
library(lattice)
library(quantmod)
# Library for SVM
library(kernlab)


setwd('\Users\Panz\SkyDrive\Princeton Stuff\2015\Fall\COS 513\COS513-Finance')
processed_news_data <- read.csv("preprocessed_data_2005_heuristic_2.csv",
                                header=TRUE, sep='\t') # read data
processed_news_data$date <- as.Date(as.character(processed_news_data$SQLDATE), "%Y%m%d")
processed_news_data <- processed_news_data[-1]
news_data_ts <- xts(processed_news_data[-ncol(processed_news_data)], order.by=processed_news_data$date)

# Reading the forex data
forex_data <- read.csv("gbpusd_2005.csv", header=TRUE, sep=',') # read data
forex_data <- as.data.frame(forex_data)
forex_data$Date <- as.Date(as.character(forex_data$Date), "%Y%m%d")
forex_data_ts <- xts(forex_data[,-1], order.by=as.Date(forex_data$Date))
names(forex_data_ts) <- c('Rate')

chartSeries(forex_data_ts, name="GBP USD Forex")

# Calculate forex rate returns
# TODO: change this to calculate the next day's return
forex_ret <- forex_data_ts$Rate / lag(forex_data_ts$Rate, 1) - 1
colnames(forex_ret) <- "DailyReturn"
# Histogram for the returns
ggplot(forex_ret,aes(x=DailyReturn)) + geom_histogram(binwidth=0.001)

# Merge for return
merged_data <- merge(forex_ret, news_data_ts)
merged_data <- merged_data[!is.na(merged_data$DailyReturn)]
fit <- glm(merged_data$DailyReturn ~ merged_data)
summary(fit) # show results
plot(coredata(merged_data$DailyReturn), coredata(merged_data$GBR_empty_pos))


## Use SVM for classification
# Positive and negative examples
threshold = 0.005
positive_returns = merged_data[merged_data$DailyReturn > threshold]
positive_returns = data.matrix(as.data.frame.ts(positive_returns))
negative_returns = merged_data[merged_data$DailyReturn < -threshold]
negative_returns = data.matrix(as.data.frame.ts(negative_returns))

# Examples for training and testing
num_iterations = 100
accuracies <- c()
for(i in 1:num_iterations) 
{
  print(i)
  examples = rbind(positive_returns, negative_returns)
  num_train = round(nrow(examples) * 0.7)
  train_index <- sample(nrow(examples), num_train)
  x_train = examples[train_index,-1]
  x_test = examples[-train_index,-1]
  label <- function(ret) ifelse(ret > 0, 1, -1)
  y_train = label(examples[train_index,1])
  y_test = label(examples[-train_index,1])
  
  # RBF dot works!!!
  svp <- ksvm(x_train, y_train, type="C-svc",kernel='vanilladot',C=100,scaled=c())
  # svp
  
  # Use the built-in function to pretty-plot the classifier
  #plot(svp, data=x_train)
  
  # Predict labels on test
  y_pred = predict(svp, x_test)
  # table(y_test, y_pred)
  
  # Compute accuracy
  accuracies[i] = sum(y_pred==y_test)/length(y_test)
  
  # Compute at the prediction scores
  y_pred_score = predict(svp,x_test,type="decision")

}

mean(accuracies)

# Using 70% training and 30% test, the heuristics got the following accuracy (repeated 100 times):
# Default RBF Gaussian function
# h1: 49.7%
# h2: 51.7%
# h3: 54.3% 
# Default linear function
# h1: 51%
# h2: 58.4%
# h3: 55.9% 
