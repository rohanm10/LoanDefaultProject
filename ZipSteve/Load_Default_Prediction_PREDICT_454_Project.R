# Northwestern University
# PREDICT 454
# Loan Default Prediction


# Install packages
# Can comment out (#) if installed already
install.packages("pastecs")
install.packages("psych")
install.packages("lattice")
install.packages("plyr")
install.packages("corrplot")
install.packages("RColorBrewer")
install.packages("caTools")
install.packages("class")
install.packages("gmodels")
install.packages("C50")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("modeest")
install.packages("randomForest")


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
multi.class.model <- rpart(loss~., data=Complete_Records[1:20000, 1:771])
rpart.plot(multi.class.model)
summary(multi.class.model)
multi.class.model$variable.importance

barplot(multi.class.model$variable.importance, main="Variable Importance Plot", 
        xlab="Variables", col= "beige")

