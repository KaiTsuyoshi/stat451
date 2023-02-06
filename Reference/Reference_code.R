## load these packages

library(prospectr)
library(pls)
library(Cubist)
library(ithir)
library(caret)
library(FactoMineR)


## Load the csv files "Soil.csv" and "VNIR_Spectra.csv"
## The soil properties are stored in the Soil.csv. The reflectance spectra are stored in the VNIR_Spectra.csv.
soil <- read.csv("Soil.csv", header = T)
ref <- read.csv("VNIR_Spectra.csv", header = T)

soil_var <- soil[,3:5] #select only y-variable

## Preprocess the vis-NIR spectra using the following steps:
## 1) Transform the reflectance spectra to absorbance spectra
## 2) Savitzky-Golay filter with the parameters: w = 11, p = 2, m = 0
## 3) Resample the spectra to 500 to 2450 nm with the spectral resolution of 10 nm
## 4) Normalize the spectra
abs <- log(100/ref) #step 1
colnames(abs) <- seq(350, 2500, by = 1)
matplot(as.numeric(colnames(abs)), t(abs), lty=1, pch=".", xlab="Wavelength (nm)", ylab="log(1/R)")

abs_sg <- savitzkyGolay(abs, w = 11, p = 2, m = 0)   #step 2
dim(abs_sg)

wave <- as.numeric(colnames(abs_sg))
new.wave <- seq(500, 2450, by = 10)
abs_new <- resample(abs_sg, wave, new.wave, interpol = 'spline')   #step 3

abs_std <- standardNormalVariate(abs_new)  #step 4

wv <- seq(500, 2450, by = 10)
matplot(wv, t(abs_std), type = 'l', lty = 1, pch = '.', xlab = 'Wavelength (nm)', ylab = 'Log(1/R)')  #normalized spectra visualization

## Randomly split the data into calibration (60%) and validation (40%) on both soil data and spectra data

set.seed(123)
cl <- sample(1:length(soil_var[,1]), round(length(soil_var[,1])*0.6))

cal_soil <- soil_var[cl,]
val_soil <- soil_var[-cl,]

cal_spek <- abs_std[cl,]
val_spek <- abs_std[-cl,]

## Use train function and pls method to develop a PLSR regression to predict Clay from processed vis-NIR spectra
## Use 10-fold CV and repeated 10 times as tuning parameters
## Select the best model using smallest RMSE value
## The maximum number of pls components set 30


Bestpls <- function(soil_variable) {
  index_col <- which(colnames(data.frame(soil_var)) == soil_variable)
  
  train_respon <- cal_soil[, index_col] # subset response data from cal_soil
  test_respon <- val_soil[, index_col] # 
  
  train_pred <- cal_spek[complete.cases(train_respon),]
  test_pred <- val_spek[complete.cases(test_respon),]
  
  train_respon <- train_respon[complete.cases(train_respon)]
  test_respon <- test_respon[complete.cases(test_respon)]
  
  fitControl <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 10)
  
  pls_model <- train(x = train_pred, y = train_respon,
                     na.action = na.omit,
                     trControl = fitControl,
                     method = 'pls',
                     tuneLength = 30,
                     metric = 'RMSE')
  
  plot(pls_model)
  
  bt = as.numeric(pls_model$bestTune)
  
  RMSE.c <- pls_model$results[which(pls_model$results$ncomp == bt),]$RMSE
  R2.c <- pls_model$results[which(pls_model$results$ncomp == bt),]$Rsquared
  RPD.c <- sd(train_respon)/RMSE.c
  RPIQ.c <- (quantile(train_respon)[4] - quantile(train_respon)[2])/RMSE.c
  
  predict.pls <- predict(pls_model, newdata = test_pred)
  
  RMSE.v <- sqrt(mean((predict.pls - test_respon)^2))
  R2.v <- cor(predict.pls, test_respon)^2
  RPD.v <- sd(test_respon)/RMSE.v
  RPIQ.v <- (quantile(test_respon)[4] - quantile(test_respon)[2])/RMSE.v
  
  
  
  return(list(all = pls_model, bt=bt, RMSE.c=RMSE.c, R2.c=R2.c,
              RPD.c=RPD.c, RPIQ.c=RPIQ.c, RMSE.v=RMSE.v, R2.v=R2.v, 
              RPD.v=RPD.v, RPIQ.v, RPIQ.v))
  
}

## Get the best number of pls components
pls.clay <- Bestpls("Clay")

pls.clay2 <- data.frame('Clay' = as.numeric(pls.clay[2:10]))

## Print the calibration and validation statistics
rownames(pls.clay2) <- c("bt", "RMSE.c","R2.c", "RPD.c", "RPIQ.c", "RMSE.v", "R2.v", "RPD.v", "RPIQ.v")
round(pls.clay2, 2)


## Check the variable importance of the PLSR model

varImp(pls.clay[[1]])


