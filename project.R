rm(list=ls()) #clear all variables in workspace

# Gym data exercise
library(caret)
library(kernlab)

# Load training data
gymData = read.csv('pml-training.csv')
# Partition into training and cross-validation data
trainingIndex = createDataPartition(gymData$classe,p=0.6,list=FALSE)

# Training and Cross-validation data
trainingData = gymData[trainingIndex,]
CVData = gymData[-trainingIndex,]

# Remove near-zero variance variables
nsv = nearZeroVar(trainingData)
trainingData = trainingData[,-nsv]
CVData = CVData[,-nsv]

# Remove timestamp variables, user name and X since we feel as they are not useful
trainingData = trainingData[,-c(1:5)]
CVData = CVData[,-c(1:5)]

pca_thresh = 0.95 # PCA threshold
# Last few cols  is the "classe" varable. We remove them and do a pca analysis
# We also perform k-nearest neighbor imputation in parallel
preProc = preProcess(trainingData[-ncol(trainingData)],method=c("knnImpute","pca"),thresh=pca_thresh)

# Determine fit coefficients on train data
trainingPC = predict(preProc,trainingData[-ncol(trainingData)])
modelFit = train(trainingData$classe~.,method="rf",data=trainingPC,trControl=trainControl(method="cv",number= 2))

###############################################
# Determine accuracy on CV data
CVPC = predict(preProc,CVData[-ncol(CVData)])
CVPrediction = predict(modelFit,CVPC)
print(confusionMatrix(CVData$classe,CVPrediction))

###############################################
# Predict on test set
testData = read.csv("pml-testing.csv")
# Remove irrelevant columns
testData = testData[-c(nsv,1:5)]
testPC = predict(preProc,testData[-ncol(testData)])
testPredictions = predict(modelFit,testPC)

###############################################
# Writing predictions to a file
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
# pml_write_files(testPredictions) # uncommment for writing out to files