# Load packet
library(randomForest)
library(e1071)
library(nnet)
library(xgboost)
library(dplyr)
library(NeuralNetTools) 
library(tidyverse)
library(skimr)
library(DataExplorer)
library(caret)
library(pROC)
library(readxl)

# Data loading
heart_train <- read_excel(file.choose())
heart_test <- read_excel(file.choose()) 
# Variable type characteristics
heart_train$Y <- as.factor(heart_train$Y)
heart_test$Y <- as.factor(heart_test$Y)
# Formulate the formula
colnames(heart_train)
form_cls <- as.formula(
  paste0("Y ~ ",paste(colnames(heart_train)[6:16],collapse = "+")))
form_cls

# Establish RF prediction model
set.seed(123)
train_RF <- randomForest(form_cls, data=heart_train,
                         mtry=4,
                         ntree=200,
                         importance = T, proximity= T)
train_RF
# Establish XGBoost prediction model
dvfunc <- dummyVars(~. ,data= heart_train[,6:16],fullRank = T)
data_trainx <- predict(dvfunc, newdata = heart_train[,6:16])
data_trainy <- ifelse(heart_train$Y == "0", 0 , 1)
data_testx <- predict(dvfunc,newdata=heart_test[,6:16])
data_testy <- ifelse(heart_test$Y == "0", 0 , 1)
dtrain <- xgb.DMatrix(data = data_trainx, label= data_trainy)
dtest <- xgb.DMatrix(data = data_testx, label = data_testy)
watchlist <- list(train =dtrain, test = dtest)
set.seed(42)
train_XGB <- xgb.train(data = dtrain, subsample=1, colsample_bytree =1, 
                       objective = "binary:logistic", nrounds =100, watchlist=watchlist, 
                       verbose =1, print_every_n = 100, early_stopping_rounds =200,
                       eta =0.3, gamma =3, max_depth=6)
# Establish SVM prediction model
set.seed(1234)
train_SVM <- svm(form_cls,data = heart_train,
                 kernel ="radial",
                 probability = T,
                 cost = 3, gamma = 0.05)
# Establish ANN prediction model
traindata <- data.frame(heart_train)
testdata <- data.frame(heart_test)
set.seed(1234)
train_ANN <- nnet(form_cls,data = traindata, maxit = 500,
                  size = 10, decay = 0.3)

# Model prediction
# RF
RF_testpredprob <- predict(train_RF, newdata = heart_test, type = "prob")
RF_testroc <- roc(response = heart_test$Y, predictor = RF_testpredprob[ ,2])
# XGBoost
XGB_testpredprob <-predict(train_XGB, newdata = dtest)
XGB_testroc <- roc(response = heart_test$Y, predictor = XGB_testpredprob)
# SVM
SVM_testpred <- predict(train_SVM,newdata = heart_test, probability = T)
SVM_testpredprob <- attr(SVM_testpred,"probabilities")
SVM_testroc <- roc(response = heart_test$Y, predictor = SVM_testpredprob[ ,2])
# ANN
ANN_testpredprob <- predict(train_ANN, newdata = heart_test, type = "raw")
ANN_testroc <- roc(response = heart_test$Y, predictor = ANN_testpredprob[ ,1])

# Draw the ROC curve
P_RF <- plot(RF_testroc, print.auc=TRUE,
             #print.thres=TRUE, 
             col="#0091DA",
             legacy.axes = TRUE,
             print.auc.y=0.5, print.auc.x=0.5,
             tck=0.02,
             main="Repetition 5",
             family="serif",
             identity.col="black", 
             identity.lwd = 1.4, 
             lty = 1,
             lwd=3,
             print.auc.cex = 1.4,
             cex.main= 1.4,cex.axis = 1.4, cex.lab=1.4)   
P_XGB <- plot(XGB_testroc,col="#FEB113",print.auc.y=0.42, print.auc.x=0.5,print.auc.cex = 1.4,lwd=3,
              family="serif",lty=1, print.auc=TRUE, legacy.axes = TRUE, add=T)
P_SVM <- plot(SVM_testroc, col="#009240", print.auc.y=0.34, print.auc.x=0.5,print.auc.cex = 1.4,lwd=3,
              family="serif",lty=1, print.auc=TRUE, legacy.axes = TRUE, add=T)
P_ANN <- plot(ANN_testroc, col="#DC1B17", print.auc.y=0.26, print.auc.x=0.5,print.auc.cex = 1.4,lwd=3,
              family="serif",lty=1, print.auc=TRUE, legacy.axes = TRUE, add=T)

# Add Legends
op <- par(family="serif")
legend(0.28,0.23,
       legend =c("RF","XGBoost","SVM","ANN"),
       col= c("#0091DA","#FEB113","#009240","#DC1B17"),
       box.col = "white",
       lwd =3,
       cex =1.1)
