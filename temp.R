#sample before modeling
set.seed(1234)
smpl1 <- Training_Default[sample(nrow(Training_Default), 10000),]

t.test(Training_Default$default,smpl1$default)

set.seed(998)
inTraining <- createDataPartition(smpl1$default, p = 0.70, list = F)
Training_Default <- smpl1[ inTraining,]
Validation_Default  <- smpl1[-inTraining,]
table(smpl1$default)


#230 variables were chosen from seven random forest models
Random_Forest <- randomForest(as.factor(default) ~ f67 + f3 + f746 + f432 + f682 + f765 + f374 + 
                                f766 + f672 + f536 + f211 + f393 + f281 + f143 + f212 + f468 + 
                                f376 + f598 + f768 + f46 + f654 + f289 + f179 + f671 + f518 + 
                                f613 + f640 + f279 + f384 + f733 + f629 + f652 + f743 + f180 + 
                                f739 + f735 + f643 + f775 + f670 + f132 + f76 + f471 + f69 + 
                                f677 + f614 + f395 + f646 + f366 + f75 + f533 + f201 + f333 + 
                                f639 + f26 + f673 + f509 + f763 + f368 + f680 + f669 + f271 + 
                                f144 + f451 + f383 + f647 + f422 + f479 + f596 + f9 + f14 + 
                                f209 + f600 + f44 + f660 + f349 + f138 + f514 + f322 + f398 + 
                                f756 + f60 + f745 + f401 + f367 + f472 + f169 + f342 + f438 + 
                                f442 + f70 + f170 + f601 + f142 + f399 + f16 + f218 + f716 + 
                                f57 + f638 + f350 + f54 + f71 + f402 + f64 + f357 + f737 + 
                                f516 + f361 + f19 + f433 + f526 + f353 + f637 + f365 + f413 + 
                                f609 + f122 + f51 + f210 + f448 + f522 + f288 + f645 + f55 + 
                                f662 + f332 + f149 + f734 + f200 + f207 + f146 + f199 + f648 + 
                                f525 + f727 + f216 + f261 + f29 + f251 + f500 + f774 + f499 + 
                                f13 + f412 + f664 + f159 + f674 + f59 + f658 + f45 + f624 + 
                                f621 + f431 + f397 + f628 + f620 + f631 + f136 + f636 + f6 + 
                                f277 + f537 + f657 + f148 + f450 + f726 + f359 + f273 + f32 + 
                                f375 + f330 + f425 + f520 + f213 + f740 + f189 + f396 + f523 + 
                                f424 + f53 + f204 + f22 + f742 + f659 + f272 + f147 + f385 + 
                                f139 + f696 + f651 + f39 + f524 + f47 + f612 + f190 + f611 + 
                                f61 + f202 + f668 + f599 + f203 + f68 + f160 + f588 + f622 + 
                                f434 + f587 + f562 + f17 + f423 + f441 + f340 + f501 + f661 + 
                                f25 + f378 + f513 + f278 + f217 + f593 + f15 + f436 + f751 + 
                                f649 + f429 + f92 + f134 + f31 + f65 + f18, 
                              data=Training_Default, importance=TRUE, ntree=200) 

print(Random_Forest)
plot(Random_Forest)
predictors(Random_Forest)
VI_Fit <- importance(Random_Forest)
varImpPlot(Random_Forest)

Random_Forest_Pred <- predict(Random_Forest, Validation_Default)
table(Random_Forest_Pred,Validation_Default$default)
Random_Forest$confusion
# Plot ROC curve and calculate Area under curve for Logistic Regression
RF_Roc_Data <- cbind(Random_Forest_Pred,Validation_Default$default)
RF_ROC <- roc(predictions = RF_Roc_Data[,1] , labels = factor(Validation_Default$default))
RF_AUC <- auc(RF_ROC)
# Plot curve for Logistic Regression
plot(RF_ROC, col = "red", lty = 1, lwd = 2, main = "ROC Curve for Random Forest Model")

# Logistic Regression Model and predictions for default
#Logistic_Regression <- glm(default ~., data=Training_Default, family = binomial)
Logistic_Regression <- glm(as.factor(default) ~ f67 + f3 + f746 + f432 + f682 + f765 + f374 + 
                                f766 + f672 + f536 + f211 + f393 + f281 + f143 + f212 + f468 + 
                                f376 + f598 + f768 + f46 + f654 + f289 + f179 + f671 + f518 + 
                                f613 + f640 + f279 + f384 + f733 + f629 + f652 + f743 + f180 + 
                                f739 + f735 + f643 + f775 + f670 + f132 + f76 + f471 + f69 + 
                                f677 + f614 + f395 + f646 + f366 + f75 + f533 + f201 + f333 + 
                                f639 + f26 + f673 + f509 + f763 + f368 + f680 + f669 + f271 + 
                                f144 + f451 + f383 + f647 + f422 + f479 + f596 + f9 + f14 + 
                                f209 + f600 + f44 + f660 + f349 + f138 + f514 + f322 + f398 + 
                                f756 + f60 + f745 + f401 + f367 + f472 + f169 + f342 + f438 + 
                                f442 + f70 + f170 + f601 + f142 + f399 + f16 + f218 + f716 + 
                                f57 + f638 + f350 + f54 + f71 + f402 + f64 + f357 + f737 + 
                                f516 + f361 + f19 + f433 + f526 + f353 + f637 + f365 + f413 + 
                                f609 + f122 + f51 + f210 + f448 + f522 + f288 + f645 + f55 + 
                                f662 + f332 + f149 + f734 + f200 + f207 + f146 + f199 + f648 + 
                                f525 + f727 + f216 + f261 + f29 + f251 + f500 + f774 + f499 + 
                                f13 + f412 + f664 + f159 + f674 + f59 + f658 + f45 + f624 + 
                                f621 + f431 + f397 + f628 + f620 + f631 + f136 + f636 + f6 + 
                                f277 + f537 + f657 + f148 + f450 + f726 + f359 + f273 + f32 + 
                                f375 + f330 + f425 + f520 + f213 + f740 + f189 + f396 + f523 + 
                                f424 + f53 + f204 + f22 + f742 + f659 + f272 + f147 + f385 + 
                                f139 + f696 + f651 + f39 + f524 + f47 + f612 + f190 + f611 + 
                                f61 + f202 + f668 + f599 + f203 + f68 + f160 + f588 + f622 + 
                                f434 + f587 + f562 + f17 + f423 + f441 + f340 + f501 + f661 + 
                                f25 + f378 + f513 + f278 + f217 + f593 + f15 + f436 + f751 + 
                                f649 + f429 + f92 + f134 + f31 + f65 + f18, 
                              data=Training_Default, family = binomial) 

Logistic_Regression_Pred <- predict(Logistic_Regression,Validation_Default, type="response")
Logistic_Default_Pred <- ifelse(Logistic_Regression_Pred >= 0.5,1,0)
table(Logistic_Default_Pred,Validation_Default$default)

# Plot ROC curve and calculate Area under curve for Logistic Regression
Log_Roc_Data <- cbind(Logistic_Default_Pred,Validation_Default$default)
Log_ROC <- roc(predictions = Log_Roc_Data[,1] , labels = factor(Validation_Default$default))
Log_AUC <- auc(Log_ROC)
# Plot curve for Logistic Regression
plot(Log_ROC, col = "red", lty = 1, lwd = 2, main = "ROC Curve for Logistic Regression")
