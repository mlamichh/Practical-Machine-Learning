# Loading data
testData = read.csv("pml-testing.csv", na.strings=c("NA", ""))
trainData = read.csv("pml-training.csv", na.strings=c("NA", ""))

# Original dimension of data
dim(testData)
dim(trainData)

# Removing columns with all NA

naCols_train = apply(trainData, 2, FUN=function(x){sum(is.na(x))})

# naCols has either value 19216 or 0. Since dim of trainData is 19622 * 160.
# It implires that in some columns have most of them are NAs and remaining columns have only few NAs.

# Now we can select only thos columns which have no NAs.

selTrainData = trainData[, c(names(naCols_train)[naCols_train == 0])]

dim(selTrainData)
# It reduces dimension to 19622 * 60.

# Since some of the variables like "X", "user_name", "raw_timestamp_part_1"
# "raw_timeStamp_part_2", "cvtd_temestamp", "new_window", and "num_window"
# does not play roles in the outcome they have been removed from training set.

dummyVar = c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window","num_window")
finalTrainData = selTrainData[, !(colnames(selTrainData) %in% dummyVar)]
dim(finalTrainData)

# Finding correlation between variables.
library(corrplot)
corMatrix = cor(finalTrainData[, -53])
corrplot(corMatrix, method="square", tl.cex=0.5)

# We can see only few variables are highly correlated.
# Removing variables with corr > 0.9
library(caret)
cutCor = findCorrelation(corMatrix, cutoff=0.9)
reducedTrain = finalTrainData[, -c(cutCor)]
dim(reducedTrain) # Checking dimension of the reduced set.

# Lets select the train (70%) and test (30%) data from finalTrainData.
inTrain = createDataPartition(reducedTrain$classe, p=0.7, list=FALSE)
sampleTrain = reducedTrain[inTrain,]
sampleTest = reducedTrain[-inTrain,]

# Analysis with tree package
library(tree)
treeFit = tree(classe ~ ., data=sampleTrain)
plot(treeFit)
text(treeFit, cex = 0.5)

treePred = predict(treeFit, sampleTest, type="class")
treeMatrix = table(sampleTest$classe, treePred)
accuracyTree = sum(diag(treeMatrix))/sum(as.vector(treeMatrix))
accuracyTree

# Analysis with Random Forest method.

library(randomForest)

rfFit = randomForest(classe ~ ., data=sampleTrain, ntree=60, importance=TRUE)
trainRF
# Relative importance of the components
varImpPlot(rfFit, cex = 0.6, main="Relative imprtance of different variables")
# Predictioning actual test data.
# Predictioning test data.
rfPred = predict(rfFit, sampleTest, type="class")
rfMatrix = table(rfPred, sampleTest$classe)
accuracyRF = sum(diag(rfMatrix))/sum(as.vector(rfMatrix))
accuracyRF

# Checking answer with the test data provided in the question

results = predict(rfFit, testData)
results


pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(results)
