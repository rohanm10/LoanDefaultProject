# Clear workspace
rm(list=ls())
# Install packages
# Can comment out (#) if installed already
#install.packages("pastecs")
#install.packages("psych")
#install.packages("lattice")
#install.packages("plyr")
#install.packages("corrplot")
#install.packages("RColorBrewer")
#install.packages("caTools")
#install.packages("class")
#install.packages("gmodels")
#install.packages("C50")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("modeest")
#install.packages("randomForest")
#install.packages("caret")

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

# Libraries used for analysis
suppressMessages(library(pastecs))
suppressMessages(library(psych))
suppressMessages(library(lattice))
suppressMessages(library(plyr))
suppressMessages(library(corrplot))
suppressMessages(library(RColorBrewer))
suppressMessages(library(caTools))
suppressMessages(library(class))
suppressMessages(library(gmodels))
suppressMessages(library(C50))
suppressMessages(library(rpart))
suppressMessages(library(rpart.plot))
suppressMessages(library(modeest))
suppressMessages(library(randomForest))

# Get list of installed packages as check
search()

# Variable names start with Capital letters and for two words X_Y
# Examples: Path, Train_Data, Train_Data_Split1, etc.

# Set working directory for your computer
# Set path to file location for your computer
# Read in data to data file
#setwd("C:/Users/Stephen D Young/Documents")
#Path <- "C:/Users/Stephen D Young/Documents/Stephen D. Young/Northwestern/Predict 454/Project/Train Data/train_v2.csv"
setwd("C:/Users/IBM_ADMIN/Northwestern/454/Project/train_v2.csv/")
Path = ("C:/Users/IBM_ADMIN/Northwestern/454/Project/train_v2.csv/")

Train_Data <- read.csv(file.path(Path,"train_v2.csv"), stringsAsFactors=FALSE)

# Get structure and dimensions of data file
#str(Train_Data)
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
# Loss is not in dollars but as percent of loan amount

# Check for missing values in loss column
sum(is.na(Train_Data$loss))
# There are no missing values in loss column

# Make first working data set and add in column for validation set split
Train_Data_Split1 <- Train_Data
Train_Data_Split1$split <- round(runif(n=nrow(Train_Data_Split1)),3)

# Add in default indicator and determine number and proportion of defaults
# Use ifelse to set default to 1 if loss > 0 otherwise default is 0
# Get table of 0 and 1 values (i.e. no default, default)
Train_Data_Split1$default <- ifelse(Train_Data_Split1$loss>0,1,0)
table(Train_Data_Split1$default)

# Results are:
#     0     1 
# 95688  9783
# 95,688 + 9,783 = 105,471 which is original number of records

# Get percentage of defaults
sum(Train_Data_Split1$default)/length(Train_Data_Split1$default)
# Percentage of defaults = 0.09275535

# Box plot and density plot of loss amount
# Lattice package for bwplot and densityplot
# Boxplot is for no default (0) and default (1)
# Boxplot should have zero loss for no default and range of loss upon default
bwplot(loss~factor(default), data=Train_Data_Split1, 
       main = "Box Plot for No Default (0) and Default (1)",
       xlab = "No Default (0), Default (1)",
       ylab = "Loss Amount (% of Loan)")

# Density plot of loss should skew right as 100% loss is max but uncommon
densityplot(~(loss[loss>0]), data = Train_Data_Split1, plot.points=FALSE, ref=TRUE,
            main = "Kernel Density of Loss Amount (% of Loan)",
            xlab = "Loss for Values Greater than Zero",
            ylab = "Density")

# Density plot of loss should skew right even with 40% as max value for plot
densityplot(~loss[loss>0 & loss<40], data = Train_Data_Split1, plot.points=FALSE, ref=TRUE,
            main = "Kernel Density of Loss Amount (% of Loan) up to 40%",
            xlab = "Loss for Values Greater than Zero and up to 40%",
            ylab = "Density")

# Basic summary statistics for loss values when greater than zero
# Get mode using mlv from modeest package
summary(Train_Data_Split1$loss[Train_Data_Split1$loss>0])
mlv(Train_Data_Split1$loss[Train_Data_Split1$loss>0], method = "mfv")
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
row.names(Summary_Statistics) <- names(Train_Data_Split1)[2:770]

# Loop for each variable included in summary statistics calculation
for(i in 1:769){
  
  Summary_Statistics[i,] <- myStatsCol(Train_Data_Split1,i+1)
  
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
Train_Data_Adj <- Train_Data_Split1[,!names(Train_Data_Split1) %in% c("loss",Zero_Cols)]

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

# Decision tree using rpart, rpart.plot and variable importance
# Decision tree on data but without standardization
# Note that loss is response but we should convert to default and predict
# loss amounts post prediction of defaults which are binary classifier
multi.class.model <- rpart(default~., data=Complete_Records[1:20000, 1:771])
rpart.plot(multi.class.model)
summary(multi.class.model)
multi.class.model$variable.importance

barplot(multi.class.model$variable.importance, main="Variable Importance Plot", 
        xlab="Variables", col= "beige")

sessionInfo()
RStudio.Version()
set.seed(1)
rf.naive <- randomForest(as.factor(loss) ~ ., data=Complete_Records[1:1000,], importance=TRUE) 
varImpPlot(rf.naive)


##########################################################################
# Shatabdi
##########################################################################
#identify duplicate columns
names(Train_Data[duplicated(as.list(Train_Data))])

#remove duplicate columns
df1 = Train_Data[!duplicated(as.list(Train_Data))]

dim(df1)
#105471    682

#identify constant columns

names(df1[,apply(df1, 2, var, na.rm=TRUE) == 0])

#remove constant columns
df2 = df1[,apply(df1, 2, var, na.rm=TRUE) != 0]

dim(df2)
#105471    679

#################         Missing value handling         #################

table(is.na(df2))
#70853599   761210 

# Missing value for each column
missColumn = sapply(Train_Data, function(x) sum(is.na(x)))

#see the top 10 fields based on number of missing values
tail(sort(missColumn),10)

summary(Train_Data$f663)
#f330  f331  f618  f619  f169  f170  f159  f160  f662  f663 
#18067 18067 18407 18407 18417 18417 18736 18736 18833 18833 

18833/105471
# missing value percentage is 17.86%, not high ( < 20% ). So, we will not exclude any column.

# Impute missing values.

#library(mice)
#md.pattern(raw_data)

#following code did not show error, but hung up after an hour.
#tempData = mice(raw_data[-raw_data$loss],m=500,seed=500, method='cart')
#summary(tempData)
#completedData = complete(tempData,1)

# Alternate (faster) approach to impute missing values.
# replace missing value with median

imputed_data = df2
for(i in 1:679){
  imputed_data[is.na(df2[,i]), i] = median(df2[,i], na.rm = TRUE)
}

table(is.na(imputed_data)) 
#FALSE 
#71614809 

dim(imputed_data)
#105471    679
#===============================================
# Outlier handling
#===============================================
# to do
#===============================================
# Correlation 
#===============================================
library(Hmisc)
correlations = rcorr(as.matrix(df2[1:1000, 1:679 ]))
correlations[is.na(correlations)] = 0

# identify variable pair with strong correlation
#correlations = unique(correlations)
#strong_corr = {}
strong_corr = vector()
for (i in 1:679){
  for (j in 1:679){
    if ( !is.na(correlations$P[i,j])){
      if ( correlations$P[i,j] > 0.999 ) {
        #strong_corr[j] = rownames(correlations$P)[i]
        strong_corr[i] = rownames(correlations$P)[i]
        print(paste(rownames(correlations$P)[i], "&" , 
                    colnames(correlations$P)[j], ": ", correlations$P[i,j]))
        strong_corr =  na.omit(strong_corr)
      }
    }
  }
}

# remove NA
strong_corr =  na.omit(strong_corr)

# remove duplicate features
strong_corr = unique(strong_corr)
strong_corr
strong_corr[1] 
strong_corr[91]

for ( i in 1:91)
{
  print(strong_corr[i])
}

reduced_Data = df2[,strong_corr]

dim(reduced_Data)

# replace missing value with median 

for(i in 1:91){
  reduced_Data[is.na(reduced_Data[,i]), i] = median(reduced_Data[,i], na.rm = TRUE)
}

table(is.na(reduced_Data)) 
#FALSE 
#9597861 

dim(reduced_Data)
#105471     91

# verify the variables
summary(reduced_Data$f3)
summary(Train_Data$f3)

summary(reduced_Data$f152)
summary(Train_Data$f152)

summary(reduced_Data$loss)
summary(Train_Data$loss)

#============ Exploratory Models using caret =====================

library(caret)


#http://topepo.github.io/caret/model-training-and-tuning.html
#========= Preparation for Modeling
# stratified random sample of the data into training and test sets
set.seed(998)
inTraining = createDataPartition(reduced_Data$loss, p = 0.70, list = F)
training <- reduced_Data[ inTraining,]
testing  <- reduced_Data[-inTraining,]

#Basic Parameter Tuning for 10-fold Cross-validation
fitControl <- trainControl(method = "repeatedcv", number = 10,repeats = 10)

#---------------------------------------
# Boosted Generalized Linear Model
#---------------------------------------
set.seed(1234)
glmboost = train(loss~., data = training, 
                 method = "glmboost", trControl = fitControl)

# View summary information
glmboost
glmboost$finalModel

# View variable importance
varImp(glmboost)
plot(varImp(glmboost))

# predict using test data
glmboost.pred <- predict(glmboost, testing)

#RMSE
postResample(pred = glmboost.pred, obs = testing$loss)
#70/30 split
#RMSE Rsquared 
#3.823503 0.002102 

#------------------
# Generalized Linear Model
#------------------
set.seed(1234)
glm = train(loss~., data = training, 
            method = "glm", trControl = fitControl)

# View summary information
glm
glm$finalModel

# View variable importance
varImp(glm)
plot(varImp(glm))

# predict using test data
glm.pred <- predict(glm, testing)

#RMSE
postResample(pred = glm.pred, obs = testing$loss)
#RMSE Rsquared 
# 


#------------------
# Random Forest 
#------------------
set.seed(1234)
randomforest = train(loss~., data = training, 
                     method = "rf", trControl = fitControl)

# View summary information
randomforest
randomforest$finalModel

plot(randomforest$finalModel)
# predict using test data
randomforest.pred <- predict(randomforest, testing)

#RMSE
postResample(pred = randomforest.pred, obs = testing$logprice)
#RMSE  Rsquared 
# 
par(mfrow=c(1, 2))
importance(randomforest$finalModel)
varImpPlot(randomforest$finalModel)
plot(randomforest$finalModel)
summary(randomforest$finalModel)
#--------------------------
# Gradient Boosting Machine
#--------------------------
# GBM
set.seed(825)
gbmFit1 <- train(loss ~ ., data = training[1:2000,], 
                 method = "gbm", 
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
# View summary information
gbmFit1

gbmFit1$finalModel

# View variable importance
varImp(gbmFit1)
plot(varImp(gbmFit1))

# predict using test data
gbmFit1.pred <- predict(gbmFit1, testing)

#RMSE
postResample(pred = gbmFit1.pred, obs = testing$loss)

#---------------------------------------
# Linear Discriminant Analysis
#---------------------------------------
set.seed(1234)
lda = train(loss~., data = training, 
                 method = "lda", trControl = fitControl)

# View summary information
lda
lda$finalModel

# View variable importance
varImp(lda)
plot(varImp(lda))

# predict using test data
lda.pred <- predict(lda, testing)

#RMSE
postResample(pred = lda.pred, obs = testing$loss)
#70/30 split
#RMSE Rsquared 
