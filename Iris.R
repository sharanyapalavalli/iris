# File : Iris.R
# Author : Sharanya Palavalli
# Date : 06/08/2017

# Read in `iris` data set
irisDS <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"),
                   header = FALSE)
# Print first few lines
head(irisDS)
# Add column names
names(irisDS) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
# View the result
irisDS
irisDS$Species <- gsub("Iris-","",irisDS$Species)

library(ggplot2)
library(GGally)
library(usdm)
library(dplyr)
library(gridExtra)

# Analyze relationships between variables
ggpairs(irisDS, aes(colour=Species), title="Iris Variable Relationships") +
  # change colors
  theme(plot.title = element_text(size=30, face='bold'))

# Petal Width vs. Petal.Length
plot1 <- ggplot(data=irisDS, aes(x=Petal.Length, y=Petal.Width, colour=Species)) +
  scale_color_manual(values=c('#9933CC','#00CC00','#FFCC00')) +
  geom_point(size=3) + geom_smooth() +
  ggtitle("Petal Width vs Petal Length") +
  xlab("Petal Length") + ylab("Petal Width") +
  annotate('text', x=2,y=2, label="Correlation : 0.963", size=6) +
  theme(axis.title.x = element_text(size=16,face='bold'),
        axis.text.x = element_text(size=12, face='bold'),
        axis.title.y = element_text(size=16,face='bold'),
        axis.text.y = element_text(size=12, face='bold'),
        plot.title = element_text(size=28,face='bold'))

# Petal Length vs. Sepal Length
plot2 <- ggplot(data=irisDS, aes(x=Petal.Length, y=Sepal.Length, colour=Species)) +
  scale_color_manual(values=c('#9933CC','#00CC00','#FFCC00')) +
  geom_point(size=3) + geom_smooth() +
  
  ggtitle("Sepal Length vs Petal Length") +
  xlab("Petal Length") + ylab("Sepal Length") +
  annotate('text', x=5.5,y=4.5, label="Correlation : 0.872", size=6) +
  
  theme(axis.title.x = element_text(size=16,face='bold'),
        axis.text.x = element_text(size=12, face='bold'),
        axis.title.y = element_text(size=16,face='bold'),
        axis.text.y = element_text(size=12, face='bold'),
        plot.title = element_text(size=28,face='bold'))

grid.arrange(plot1, plot2)

summary(irisDS)

sample.size <- floor(0.75 * nrow(irisDS))
# set seed
set.seed(321)
# create index for data partition
ind <- sample(seq_len(nrow(iris)), size=sample.size)
# train & test data frames
trainIris <- irisDS[ind,]
testIris <- irisDS[-ind,]

# Separate the input and target variables in the training and validation data sets
train.X <- irisDS[ind, 1:4]
train.y <- irisDS[ind, 5]
test.X <- irisDS[-ind, 1:4]
test.y <- irisDS[-ind, 5]

library(MASS)
# Set prior probability and build model
ldaModel <- lda(Species ~ ., data=trainIris, prior=c(1,1,1)/3)

# summarize LDA model
ldaModel

# Make predictions on test data
predModelLDA <- predict(ldaModel, newdata=testIris, type='class')

# Evaluate model
table(predModelLDA$class, test.y)
# Compute accuracy of predictions
ldaAccuracy <- sum(predModelLDA$class == test.y)/length(predModelLDA$class)
ldaAccuracy
# Visualize results using dataset 
datasetLDA <- data.frame(Species=test.y, lda=predModelLDA)
propModelLDA <- ldaModel$svd^2/sum(ldaModel$svd^2)

library(scales)

ggplot(data=datasetLDA, aes(x=lda.x.LD1, y=lda.x.LD2, colour=Species)) +
  geom_point(size=3) +
  
  xlab(paste("LD1 (", percent(propModelLDA[1]), ")", sep="")) +
  ylab(paste("LD2 (", percent(propModelLDA[2]), ")", sep="")) +
  ggtitle("Linear Discriminant Analysis") +
  scale_color_manual(values=c('#9933CC','#00CC00','#FFCC00')) +
  
  theme(plot.title = element_text(size=32, face='bold'),
        axis.title.x = element_text(size=16, face='bold'),
        axis.title.y = element_text(size=16, face='bold'))

# Using Random Forest to predict the flower class

library(randomForest)
# develop model
rfModel <- randomForest(Species ~ ., data = trainIris, proximity=T, ntree = 200,mtry = 2, nodesize = 1, importance = T)

# visualize model
plot(rfModel, log='y')
MDSplot(rfModel, train.y)

# predict class
rfPredModel <- predict(rfModel, testIris)
table(rfPredModel, test.y)
rfAccuracy <- sum(rfPredModel == test.y)/length(rfPredModel)
rfAccuracy

# Using SVM to predict the flower class
library(e1071)
# build model with train X & y
svmModel <- best.svm(Species ~ ., data=trainIris)

# summarize the model
svmModel
# generate predictions
svmPredModel <- predict(svmModel, test.X, type="class")
# view distribution of predictions
table(svmPredModel, test.y)
# accuracy of predictions
svmAccuracy <- sum(svmPredModel == test.y)/length(svmPredModel)
svmAccuracy


#Using knn for predicting the flower class
library(class)

# Building classifier using knn algorithm
knnModel <- knn(train = trainIris, test = testIris, cl = train.y, k=3)

# Inspect the model
knnModel

# Approach A
# Evaluate the model 
# Insert the test labels in a data frame

irisTestLabels <- data.frame(test.y)

# Merge predictions with the test labels
merge <- data.frame(knnModel, irisTestLabels)

# Rename columns of merged data frame
names(merge) <- c("Predicted Species", "Observed Species")

# Inspect merged data frame
merge

#install and import package gmodels to evaluate predictive model
library("gmodels");

CrossTable(x = test.y, y = knnModel, prop.chisq=FALSE)

#Approach B
library(caret)

# Train your model
knnModel2 <- train(trainIris[, 1:4], trainIris[, 5], method='knn')

# Predict the labels for the test set
predictedLabels <- predict(object=knnModel2,testIris[,1:4])

# Evaluate the predictions
table(predictedLabels)

# Confusion matrix
confusionMatrix(predictedLabels,testIris[,5])

###########  End of File ###############
