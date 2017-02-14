# Northwestern University
# PREDICT 454
# Loan Default Prediction

# Install packages
# Can comment out (#) if installed already
install.packages("pastecs", dependencies = TRUE)
install.packages("psych", dependencies = TRUE)
install.packages("lattice", dependencies = TRUE)
install.packages("plyr", dependencies = TRUE)
install.packages("corrplot", dependencies = TRUE)
install.packages("RColorBrewer", dependencies = TRUE)
install.packages("caTools", dependencies = TRUE)
install.packages("class", dependencies = TRUE)
install.packages("gmodels", dependencies = TRUE)
install.packages("C50", dependencies = TRUE)
install.packages("rpart", dependencies = TRUE)
install.packages("rpart.plot", dependencies = TRUE)
install.packages("modeest", dependencies = TRUE)
install.packages("randomForest", dependencies = TRUE)
install.packages("caret", dependencies = TRUE)
install.packages("ggplo2", dependencies = TRUE)
install.packages("readr", dependencies = TRUE)
install.packages("AUC", dependencies = TRUE)

# Include the following libraries
library(pastecs)
library(psych)
library(lattice)
library(plyr)
library(corrplot)
library(RColorBrewer)
library(caTools)
library(class)
library(gmodels)
library(C50)
library(rpart)
library(rpart.plot)
library(modeest)
library(randomForest)
library(caret)
library(ggplot2)
library(readr)
library(AUC)

# Get list of installed packages as check
search()

# Variable names start with Capital letters and for two words X_Y
# Examples: Path, Train_Data, Train_Data_Split1, etc.

# Set working directory for your computer
# Set path to file location for your computer
# Read in data to data file
setwd("C:/Users/Stephen D Young/Documents")
Path <- "C:/Users/Stephen D Young/Documents/Stephen D. Young/Northwestern/Predict 454/Project/Train Data/train_v2.csv"
Train_Data <- read.csv(file.path(Path,"train_v2.csv"), stringsAsFactors=FALSE)

# Get structure and dimensions of data file
str(Train_Data)
dim(Train_Data)

# Characteristics of data frame from above
# 105471 rows(records), 771 columns, 770 not including loss which is last column 
# there is no indicator variable for default but can be 0 or 1 for when there is
# an observed loss.  So we can create binary default flag.
# Summary statistics for loss variable
# options used to define number and decimal places
options(scipen = 100)
options(digits = 4)
summary(Train_Data$loss)
# Get mode using mlv from modeest package
mlv(Train_Data$loss, method = "mfv")
# Results of loss are min = 0, max = 100, mean = .8, mode = 0
# Loss is not in dollars but as percent of loan amount (e.g. 50 is loss of 
# 50 out of 100 loan amount)

# Check for missing values in loss column
sum(is.na(Train_Data$loss))
# There are no missing values in loss column

# Add in default indicator and determine number and proportion of defaults
# Use ifelse to set default to 1 if loss > 0 otherwise default is 0
# Get table of 0 and 1 values (i.e. no default, default)
Train_Data$default <- ifelse(Train_Data$loss>0,1,0)
table(Train_Data$default)
# Results are:
#     0     1 
# 95688 (No Default)  9783 (Default)
# 95,688 + 9,783 = 105,471 which is original number of records

# Get percentage of defaults
sum(Train_Data$default)/length(Train_Data$default)
# Percentage of defaults = 0.09275535

# Box plot and density plot of loss amount
# Lattice package for bwplot and densityplot
# Boxplot is for no default (0) and default (1)
# Boxplot should have zero loss for no default and range of loss upon default
bwplot(loss~factor(default), data=Train_Data, 
       main = "Box Plot for No Default (0) and Default (1)",
       xlab = "No Default (0), Default (1)",
       ylab = "Loss Amount (% of Loan)")

# Density plot of loss should skew right as 100% loss is max but uncommon
densityplot(~(loss[loss>0]), data = Train_Data, plot.points=FALSE, ref=TRUE,
            main = "Kernel Density of Loss Amount (% of Loan)",
            xlab = "Loss for Values Greater than Zero",
            ylab = "Density")

# Density plot of loss should skew right even with 40% as max value for plot
densityplot(~loss[loss>0 & loss<40], data = Train_Data, plot.points=FALSE, ref=TRUE,
            main = "Kernel Density of Loss Amount (% of Loan) up to 40%",
            xlab = "Loss for Values Greater than Zero and up to 40%",
            ylab = "Density")

# Basic summary statistics for loss values when greater than zero
# Get mode using mlv from modeest package
summary(Train_Data$loss[Train_Data$loss>0])
mlv(Train_Data$loss[Train_Data$loss>0], method = "mfv")
# Results of loss are min = 1, max = 100, mean = 8.62, mode = 2

# Function to compute summary statistics
myStatsCol <- function(x,i){
  
  # Nine statistics from min to na
  mi <- round(min(x[,i], na.rm = TRUE),4)
  q25 <- round(quantile(x[,i],probs = 0.25,  na.rm=TRUE),4)
  md <- round(median(x[,i], na.rm = TRUE),4)
  mn <- round(mean(x[,i], na.rm = TRUE),4)
  st <- round(sd(x[,i], na.rm = TRUE),4)
  q75 <- round(quantile(x[,i], probs = 0.75, na.rm = TRUE),4)
  mx <- round(max(x[,i], na.rm = TRUE))
  ul <- length(unique(x[complete.cases(x[,i]),i]))
  na <- sum(is.na(x[,i]))
  
  # Get results and name columns
  results <- c(mi, q25, md, mn, st, q75, mx, ul, na)
  names(results) <- c("Min.", "Q.25", "Median", "Mean",
                      "Std.Dev.", "Q.75", "Max.",
                      "Unique", "NA's")
  results
}

# Call to summary statistics function
# There are 9 summary statistics and 769 predictors excluding id and loss
Summary_Statistics <- matrix(ncol = 9, nrow = 769)
colnames(Summary_Statistics) <- c("Min.", "Q.25", "Median", "Mean",
                                  "Std.Dev.", "Q.75", "Max.",
                                  "Unique", "NA's")
row.names(Summary_Statistics) <- names(Train_Data)[2:770]

# Loop for each variable included in summary statistics calculation
for(i in 1:769){
  
  Summary_Statistics[i,] <- myStatsCol(Train_Data,i+1)
  
}

# Create data frame of results
Summary_Statistics <- data.frame(Summary_Statistics)
# View select records for reasonableness
head(Summary_Statistics,20)

# Output file of predictor variable summaries
write.csv(Summary_Statistics, file = file.path(Path,"myStats_Data1.csv"))

# Creates summary table from which we get n which is number of missing
# records (i.e. Summary$n)
Summary <- describe(Train_Data, IQR = TRUE, quant=TRUE)

# Calculate number of missing records for each variable
Missing_Var <- nrow(Train_Data) - Summary$n
plot(Missing_Var, main = "Plot of Missing Values", xlab = "Variable Index",
     ylab = "Number of Rows - Missing Records")

# Get variable names missing more than 5000 records as we may want to remove
# those with n missing > 5000 as imputation could be problematic
Lot_Missing <- colnames(Train_Data)[Missing_Var >= 5000]
print(Lot_Missing)
# Variables for which n missing > 5,000
# [1] "f72"  "f159" "f160" "f169" "f170" "f179" "f180" "f189" "f190" "f199"
# [11] "f200" "f209" "f210" "f330" "f331" "f340" "f341" "f422" "f586" "f587"
# [21] "f588" "f618" "f619" "f620" "f621" "f640" "f648" "f649" "f650" "f651"
# [31] "f653" "f662" "f663" "f664" "f665" "f666" "f667" "f668" "f669" "f672"
# [41] "f673" "f679" "f726"

# Find columns where all values are the same as we can remove these
Zero_Cols <- rownames(Summary)[Summary$range==0 & Summary$sd == 0]
print(Zero_Cols)
# [1] "f33"  "f34"  "f35"  "f37"  "f38"  "f678" "f700" "f701" "f702" "f736"
# [11] "f764"

# Get number of records that have no missing predictor variables
Complete_Records <- Train_Data[complete.cases(Train_Data),]
print(Complete_Records)
# If one creates a data set that has complete records for all predictors it
# would have 51,940 rows or 49% of total records (51,940/105,471 = .49)

# Principal Components Analysis (PCA) for Dimensionality Reduction
# Remove constant, columns with all zeros, and loss for PCA rescale
Train_Data_Adj <- Train_Data[,!names(Train_Data) %in% c("loss",Zero_Cols)]

# Run PCA removing observations with missing data as first pass and scaling
# which is to unit variance
PCs <- prcomp(Train_Data_Adj[complete.cases(Train_Data_Adj),], scale=TRUE)

#Variance explained by each principal component
PCs_Var <- PCs$sdev^2

#Proportion of variance explained 
Prop_Var_Expl <- PCs_Var / sum(PCs_Var)

# Plot of proportion of variance explained and cumulative proportion of variance explained
plot(Prop_Var_Expl, main = "Proportion of Variance Explained per PC", xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(Prop_Var_Expl), main = "Cumulative Proportion of Variance Explained",xlab=" Principal Component", ylab ="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
# Based on PCA may be able to reduce features to ~ 200

# Identify duplicate columns
names(Train_Data[duplicated(as.list(Train_Data))])
# Variables for which there are duplicate columns - should remove one of each
# [1] "f34"  "f35"  "f37"  "f38"  "f58"  "f86"  "f87"  "f88"  "f96"  "f97" 
# [11] "f98"  "f106" "f107" "f108" "f116" "f117" "f118" "f126" "f127" "f128"
# [21] "f155" "f156" "f157" "f165" "f166" "f167" "f175" "f176" "f177" "f185"
# [31] "f186" "f187" "f195" "f196" "f197" "f225" "f226" "f227" "f235" "f236"
# [41] "f237" "f245" "f246" "f247" "f255" "f256" "f257" "f265" "f266" "f267"
# [51] "f294" "f295" "f296" "f302" "f303" "f304" "f310" "f311" "f312" "f318"
# [61] "f319" "f320" "f326" "f327" "f328" "f345" "f354" "f362" "f371" "f379"
# [71] "f408" "f417" "f427" "f457" "f478" "f488" "f498" "f508" "f553" "f563"
# [81] "f573" "f582" "f599" "f700" "f701" "f702" "f729" "f741" "f764"
# Should not eliminate all of above but rather keep one of each duplicated
# variable (e.g. if X = Y = Z, keep X or Y or Z)
# Is done further down using correlation as measure of redundancy

# Identify constant columns
names(Train_Data[,apply(Train_Data, 2, var, na.rm=TRUE) == 0])
# Variables for which there are constant columns - should remove altogether
# [1] "f33"  "f34"  "f35"  "f37"  "f38"  "f678" "f700" "f701" "f702" "f736"
# [11] "f764"

# Remove constant columns
New_Train_Data <- Train_Data[,apply(Train_Data, 2, var, na.rm=TRUE) !=0]
# 11 variables have constants and so we go from 771 to 760

# Get table of new training data where constant valued variables are 
# eliminated
table(is.na(New_Train_Data))
# 79372006 (False) + 785954 (True) = 80157960 which is total number of 
# cells or 760 x 105471 = 80157960

# Sum if NA for all variables to get number of missing values for each 
# variable
Miss_Column <- sapply(New_Train_Data, function(x) sum(is.na(x)))

# Get the top 10 variables with missing values and amounts missing 
tail(sort(Miss_Column),10)
# f330  f331  f618  f619  f169  f170  f159  f160  f662  f663 
# 18067 18067 18407 18407 18417 18417 18736 18736 18833 18833 
# 184920 total cells with missing values for top 10

# Get summary for f633 to get number of NA's and calculate % missing
summary(New_Train_Data$f663)

# Percent missing values
18833/105471
# Missing value percentage is 17.86%, not high ( < 20% ). So, we will 
# not exclude any column but rather impute with median

# Impute missing values
Imputed_Data <- New_Train_Data
for(i in 1:760){
  
  Imputed_Data[is.na(New_Train_Data[,i]), i] = median(New_Train_Data[,i], na.rm = TRUE)

}

# Number of columns in Imputed_Data followed by rows, columns
ncol(Imputed_Data)
dim(Imputed_Data)
# 105471    760

# Remove highly correlated variables - should keep one of two as correlation
# is pairwise
High_Correlation <- cor(Imputed_Data)
High_Correlation_Data <- findCorrelation(High_Correlation,cutoff=0.99)
High_Correlation_Data
# All below columns are those that have high correlation with another 
# variable - 289 values from calculation - (760 - 289 = 471 columns)
# [1]  11  34  44  50  52  68 109 110 111 112 119 120 121 122 135 178 179 180
# [19] 181 188 189 190 191 248 249 250 251 258 259 260 261 274 337 339 346 348
# [37] 354 356 363 365 367 371 373 381 383 402 408 409 411 421 429 431 433 437
# [55] 447 451 456 459 465 468 473 474 475 476 478 480 481 483 484 485 487 488
# [73] 494 495 498 507 509 511 517 518 519 522 528 529 530 533 535 539 540 543
# [91] 545 549 553 555 558 559 560 561 562 563 565 567 568 569 570 571 572 573
# [109] 574 575 593 595 610 612 613 614 617 620 629 631 642 650 667 670 673 674
# [127] 675 676 678 689 690 691 692 693 694 695 696 697 701 702 705 711 712 713
# [145] 720 723 730 731 732 734 739 740 741 743 744 748 751  32  36  42  72  79
# [163]  80  81  89  90  91  99 100 101 131 139 148 149 150 151 158 159 160 161
# [181] 168 169 170 199 200 208 218 219 220 228 229 230 238 239 240 221 270 287
# [199] 288 289 295 296 297 303 304 305 311 312 313 319 320 321 331 342 350  46
# [217] 276  22 401 403 404 277 439   8  82 446 448 129 461 450  92 467 453 466
# [235] 477 471 470 102 493   9 534 231 538 544 241 548 541 542 554 531 532 564
# [253] 566 576 505 587 441  73 440 594 380 382 386 622  35  37  66 399  56 669
# [271] 671 672 677 680 681 686 687 688 703 714 735 736  20 663 398 400 750 752
# [289] 753

# Sorted Data
High_Correlation_Data <- sort(High_Correlation_Data)

# Reduced training data set where reductions include values that are constants
# and one of each highly correlated predictor which should include those
# predictors or columns that have same values (i.e. correlation of one)
Reduced_Training_Data <- Imputed_Data[,-c(High_Correlation_Data)]

# Adding default flag for loss > 0
Reduced_Training_Data$default <- ifelse(Reduced_Training_Data$loss>0,1,0)
dim(Reduced_Training_Data)
# 105471 by 472 from above
# Getting rid of loss value to focus on default
Training_Data_Default <- Reduced_Training_Data[,-c(471)]
dim(Training_Data_Default)
names(Training_Data_Default)
ncol(Training_Data_Default)
# Now there are 471 columns with last one loss amount
# We can use Reduced_Training_Data to create Training_Data_Loss for loss
# modeling purposes

# Dimensions are 105471 471
table(Training_Data_Default$default)
# 95688 (No Default) 9783 (Default)

# Partitioning training data for training and validation
# Will ultimately have three sets training, validation, testing

# Set seed for reproducibility
set.seed(998)
inTraining <- createDataPartition(Training_Data_Default$default, p = 0.70, list = F)
Training_Default <- Training_Data_Default[ inTraining,]
Validation_Default  <- Training_Data_Default[-inTraining,]
table(Training_Default$default)
ncol(Training_Default)
table(Validation_Default$default)
ncol(Validation_Default)
# Remember that last value (471) is default flag

# Get rid of ID
Training_Default <- Training_Default[,-1]
ncol(Training_Default)
Validation_Default <- Validation_Default[,-1]
ncol(Validation_Default)
# Now last column value (470) is default flag

# Check data to ensure loss is not included and default is included
names(Training_Default)

# Decision trees using rpart, rpart.plot and variable importance
# Using 10,000 records at a time to look for important variables
Decision_Tree1 <- rpart(default~., data=Training_Default[1:10000,])
rpart.plot(Decision_Tree1)
summary(Decision_Tree1)

Decision_Tree2 <- rpart(default~., data=Training_Default[10001:20000,])
rpart.plot(Decision_Tree2)
summary(Decision_Tree2)

Decision_Tree3 <- rpart(default~., data=Training_Default[20001:30000,])
rpart.plot(Decision_Tree3)
summary(Decision_Tree3)

Decision_Tree4 <- rpart(default~., data=Training_Default[30001:40000,])
rpart.plot(Decision_Tree4)
summary(Decision_Tree4)

Decision_Tree5 <- rpart(default~., data=Training_Default[40001:50000,])
rpart.plot(Decision_Tree5)
summary(Decision_Tree5)

Decision_Tree6 <- rpart(default~., data=Training_Default[50001:60000,])
rpart.plot(Decision_Tree6)
summary(Decision_Tree6)

Decision_Tree7 <- rpart(default~., data=Training_Default[60001:73829,])
rpart.plot(Decision_Tree7)
summary(Decision_Tree7)

# Variable importance
Decision_Tree1$variable.importance
Decision_Tree2$variable.importance
Decision_Tree3$variable.importance
Decision_Tree4$variable.importance
Decision_Tree5$variable.importance
Decision_Tree6$variable.importance
Decision_Tree7$variable.importance

par(mfrow=c(3,1))
barplot(Decision_Tree1$variable.importance, main="Variable Importance Plot - Tree1",
        xlab="Variables", col= "beige")
barplot(Decision_Tree2$variable.importance, main="Variable Importance Plot - Tree2",
        xlab="Variables", col= "beige")
barplot(Decision_Tree3$variable.importance, main="Variable Importance Plot - Tree3",
        xlab="Variables", col= "beige")

par(mfrow=c(3,1))
barplot(Decision_Tree4$variable.importance, main="Variable Importance Plot - Tree4",
        xlab="Variables", col= "beige")
barplot(Decision_Tree5$variable.importance, main="Variable Importance Plot - Tree5",
        xlab="Variables", col= "beige")
barplot(Decision_Tree6$variable.importance, main="Variable Importance Plot - Tree6",
        xlab="Variables", col= "beige")

par(mfrow=c(1,1))
barplot(Decision_Tree7$variable.importance, main="Variable Importance Plot - Tree7",
        xlab="Variables", col= "beige")

# There are 19 unique predictors that arise from the above plots.
# f766 f765 f281 f677 f400 f675 f322 f323 f333 f314 f402 f32 f64 f396
# f395 f397 f282 f25 f376

# Random Forest Model for Variable Importance EDA and Prediction on Validation
# Data used does not include loss which would predict default perfectly
# Need to set ntree value - computationally costly to use large number 
set.seed(100)
Random_Forest1 <- randomForest(as.factor(default) ~ ., data=Training_Default[1:10000,], importance=TRUE, ntree=50) 
VI_Fit1 <- importance(Random_Forest1)

Random_Forest2 <- randomForest(as.factor(default) ~ ., data=Training_Default[10001:20000,], importance=TRUE, ntree=50) 
VI_Fit2 <- importance(Random_Forest2)

Random_Forest3 <- randomForest(as.factor(default) ~ ., data=Training_Default[20001:30000,], importance=TRUE, ntree=50) 
VI_Fit3 <- importance(Random_Forest3)

Random_Forest4 <- randomForest(as.factor(default) ~ ., data=Training_Default[30001:40000,], importance=TRUE, ntree=50) 
VI_Fit4 <- importance(Random_Forest4)

Random_Forest5 <- randomForest(as.factor(default) ~ ., data=Training_Default[40001:50000,], importance=TRUE, ntree=50) 
VI_Fit5 <- importance(Random_Forest5)

Random_Forest6 <- randomForest(as.factor(default) ~ ., data=Training_Default[50001:60000,], importance=TRUE, ntree=50) 
VI_Fit6 <- importance(Random_Forest6)

Random_Forest7 <- randomForest(as.factor(default) ~ ., data=Training_Default[60001:70000,], importance=TRUE, ntree=50) 
VI_Fit7 <- importance(Random_Forest7)

# Plots from random forests runs on subsets of training data
par(mfrow=c(3,1))
varImpPlot(Random_Forest1)
varImpPlot(Random_Forest2)
varImpPlot(Random_Forest3)

par(mfrow=c(3,1))
varImpPlot(Random_Forest4)
varImpPlot(Random_Forest5)
varImpPlot(Random_Forest6)

par(mfrow=c(1,1))
varImpPlot(Random_Forest7)

# Importance variables - first N for each run of random forests
# can possibly use these to find important predictors to run 
# larger random forests and other models
rf1 <- Random_Forest1$importance[,4]
as.data.frame(rf1)

rf2 <- Random_Forest2$importance[,4]
as.data.frame(rf2)

rf3 <- Random_Forest3$importance[,4]
as.data.frame(rf3)

rf4 <- Random_Forest4$importance[,4]
as.data.frame(rf4)

rf5 <- Random_Forest5$importance[,4]
as.data.frame(rf5)

rf6 <- Random_Forest6$importance[,4]
as.data.frame(rf6)

rf7 <- Random_Forest7$importance[,4]
as.data.frame(rf7)


# Random Forest Model for Variable Importance EDA and Prediction on Validation
# Data used does not include loss which would predict default perfectly
# Need to set ntree value - computationally costly to use large number 
set.seed(100)
Random_Forest <- randomForest(as.factor(default) ~ ., data=Training_Default, importance=TRUE, ntree=200) 
print(Random_Forest)
plot(Random_Forest)
predictors(Random_Forest)
VI_Fit <- importance(fit)
varImpPlot(Random_Forest)

Random_Forest_Pred <- predict(Random_Forest, Validation_Default)
table(Random_Forest_Pred,Validation_Default$default)

# Plot ROC curve and calculate Area under curve for Logistic Regression
RF_Roc_Data <- cbind(Random_Forest_Pred,Validation_Default$default)
RF_ROC <- roc(predictions = RF_Roc_Data[,1] , labels = factor(Validation_Default$default))
RF_AUC <- auc(RF_ROC)
# Plot curve for Logistic Regression
plot(RF_ROC, col = "red", lty = 1, lwd = 2, main = "ROC Curve for Random Forest Model")

# Logistic Regression Model and predictions for default
Logistic_Regression <- glm(default ~., data=Training_Default, family = binomial)
Logistic_Regression_Pred <- predict(Logistic_Regression,Validation_Default, type="response")
Logistic_Default_Pred <- ifelse(Logistic_Regression_Pred >= 0.5,1,0)
table(Logistic_Default_Pred,Validation_Default$default)

# Plot ROC curve and calculate Area under curve for Logistic Regression
Log_Roc_Data <- cbind(Logistic_Default_Pred,Validation_Default$default)
Log_ROC <- roc(predictions = Log_Roc_Data[,1] , labels = factor(Validation_Default$default))
Log_AUC <- auc(Log_ROC)
# Plot curve for Logistic Regression
plot(Log_ROC, col = "red", lty = 1, lwd = 2, main = "ROC Curve for Logistic Regression")
