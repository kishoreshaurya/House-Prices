#+eval=FALSE
library(readr)
library(tidyverse)
library(mice)
library(corrplot)
library(caret)
library(glmnet)

train <- read.csv("train.csv",stringsAsFactors=T, header=T)
test <- read.csv("test.csv",stringsAsFactors=T, header=T)


test$SalePrice <- NA

main_df <- rbind(train,test)
test$SalePrice <- NULL
str(train)


rapply(train, class = "factor", f = levels, how = "list")

md.pattern(main_df)
as.data.frame(colSums(is.na(main_df)))

sum.na <- sort(sapply(train, function(x) { sum(is.na(x))}), decreasing=TRUE)
sum.na
sum.na.percent <- sort(sapply(train, function(x) { sum(is.na(x)/dim(train)[1])}), decreasing=TRUE)
as.data.frame(sum.na.percent)

contVar <- names(train)[which(sapply(train, is.numeric))]
trainCont <- train[, contVar]
correlations <- cor(trainCont, use = "pairwise.complete.obs")
corrplot(correlations, method = "square")

# look up top 10 feature correlated to SalePrice
cor <- as.data.frame(as.table(correlations))
cor <- subset(cor, cor$Var2 == "SalePrice")
cor <- cor[order(cor$Freq, decreasing = T)[1:10],]
cor

#train set
train$PoolQC <- NULL
train$MiscFeature <- NULL
train$Alley <- NULL
train$Utilities <- NULL

ggplot(train, aes(x = GrLivArea, y = SalePrice)) + geom_point() + title("Outliers")
plot(train$GrLivArea,train$SalePrice)
order(train$GrLivArea,decreasing = T)[1:2]
train <- train[-1299,]
train <- train[-524,]


#imputation with mean
imp <- mice(train, m=1, method="cart")
train_comp <- mice::complete(imp)
as.data.frame(colSums(is.na(train_comp)))

#log transformation of SalePrice in train
hist(train$SalePrice)
train_comp$SalePrice <- log(train_comp$SalePrice)

#test Set
test$PoolQC <- NULL
test$MiscFeature <- NULL
test$Alley <- NULL
test$Utilities <- NULL

#imputation with mean
imp1 <- mice(test, m=1, method="cart")
test_comp <- mice::complete(imp1)
as.data.frame(colSums(is.na(test_comp)))


test_comp$SalePrice <- NA
train_comp <- train_comp %>% mutate(GarageYrBlt= as.factor(GarageYrBlt),YrSold=as.factor(YrSold), YearBuilt =as.factor(YearBuilt),
                                    YearRemodAdd = as.factor(YearRemodAdd))
test_comp <- test_comp %>% mutate(GarageYrBlt= as.factor(GarageYrBlt),YrSold=as.factor(YrSold), YearBuilt =as.factor(YearBuilt),
                                  YearRemodAdd = as.factor(YearRemodAdd))


main_df <- rbind(train_comp,test_comp)
as.data.frame(colSums(is.na(main_df)))

df <- as.data.frame(main_df[,77])
main_df <- main_df[,-c(77)]

library(e1071)

classes <- lapply(main_df,function(x) class(x))
numeric_feats <- names(classes[classes=="integer" | classes=="numeric"])
factor_feats <- names(classes[classes=="factor"| classes=="character"])

skewed_feats <- sapply(numeric_feats, function(x) skewness(main_df[[x]]))
skewed_feats <- skewed_feats[abs(skewed_feats) > .50]
as.table(skewed_feats)
hist(main_df$KitchenAbvGr)


for (x in names(skewed_feats)) {main_df[[x]] <- log(main_df[[x]]+1)}
main_df <- cbind(main_df, df = df$`main_df[, 77]`)
colnames(main_df)[77] <- "SalePrice"

dmy <- dummyVars( ~ ., data = main_df)
dmy_predict <- data.frame(predict(dmy, newdata = main_df))

master_df <- dmy_predict
master_df_train <- master_df[1:1458,]
master_df_test <- master_df[1459:2917,]

#LASSO Regression

options(na.action='na.pass')
y<- master_df_train$SalePrice

X <- model.matrix(SalePrice ~.^2, master_df)[,-c(1)]

X.training<- subset(X,X[,1]< 1461)
X.prediction<- subset(X,X[,1]>=1461)

nlasso.fit<-glmnet(x = X.training, y = y, alpha = 1)
plot(nlasso.fit, xvar = "lambda")

crossval <-  cv.glmnet(x = X.training, y = y, alpha = 1) #create cross-validation data. By default, the function performs ten-fold cross-validation, though this can be changed using the argument nfolds. 
plot(crossval)
penalty.lasso <- crossval$lambda.min #determine optimal penalty parameter, lambda
log(penalty.lasso) #see where it was on the graph
lasso.opt.fit <-glmnet(x = X.training, y = y, alpha = 1, lambda = penalty.lasso) #estimate the model with the optimal penalty
coef <- coef(lasso.opt.fit) #resultant model coefficients


# predicting the performance on the testing set
predicted.prices.log.i.lasso <- exp(predict(lasso.opt.fit, s = penalty.lasso, newx =X.prediction))
write.csv(predicted.prices.log.i.lasso, file = "Predicted Sale Prices.csv")

#Ridge Regression
ridge.fit<-glmnet(x = X.training, y = y, alpha = 0)
plot(ridge.fit, xvar = "lambda")

#selecting the best penalty lambda
crossval1 <-  cv.glmnet(x = X.training, y = y, alpha = 0)
plot(crossval1)
penalty.ridge <- crossval1$lambda.min 
log(penalty.ridge) 
ridge.opt.fit <-glmnet(x = X.training, y = y, alpha = 0, lambda = penalty.ridge) #estimate the model with that
coef(ridge.opt.fit)

ridge.testing <- exp(predict(ridge.opt.fit, s = penalty.ridge, newx =X.prediction))
write.csv(ridge.testing, file = "Predicted Sale Prices Ridge.csv")
