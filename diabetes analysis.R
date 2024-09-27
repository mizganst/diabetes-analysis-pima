library(caret)
library(tidyverse)
library(MASS)
library(mlbench)
library(corrplot)
library(gridExtra)
library(timeDate)
library(pROC)
library(caTools)
library(rpart.plot)
library(e1071)
library(graphics)

data("PimaIndiansDiabetes")
df <- PimaIndiansDiabetes
str(df)
head(df)
summary(df)

partition <- caret::createDataPartition(y = df$diabetes, times = 1, p = 0.7, list = FALSE)
train_set <- df[partition,]
test_set <- df[-partition,]
str(train_set)


#Exploratory Analysis

ggplot(df, aes(diabetes, fill = diabetes))+ geom_bar()+theme_bw()+labs(title = "Diabetes Classification Count", x = "Diabetes") + theme(plot.title = element_text(hjust = 0.5))

cor_data <- cor(df[,setdiff(names(df), 'diabetes')])
cor_data
corrplot::corrplot(cor_data)
corrplot::corrplot(cor_data, type = "lower", method = "number")

ggplot(data = df, aes(x = pregnant, fill = diabetes)) +
    geom_bar(stat='count', position='dodge') +
    ggtitle("pregnant Vs Diabetes") +
    theme_bw() +
    labs(x = "pregnant") +
    theme(plot.title = element_text(hjust = 0.5))  

#ML modelling

#regression and classification trees

model_rpart <- caret::train(diabetes ~., data = train_set,
                            method = "rpart",
                            metric = "ROC",
                            tuneLength = 20,
                            trControl = trainControl(method = "cv", number = 10,
                                                     classProbs = T, summaryFunction = twoClassSummary),
                            preProcess = c("center","scale","pca"))
plot(model_rpart)
model_rpart$bestTune
model_rpart$finalModel
rpart.plot::rpart.plot(model_rpart$finalModel, type = 2, fallen.leaves = T, extra = 2, cex = 0.60)

#prediction on test set

pred_rpart <- predict(model_rpart, test_set)
cm_rpart <- confusionMatrix(pred_rpart, test_set$diabetes, positive="pos")
pred_prob_rpart <- predict(model_rpart, test_set, type="prob")
roc_rpart <- roc(test_set$diabetes, pred_prob_rpart$pos)

cm_rpart
caTools::colAUC(pred_prob_rpart$pos, test_set$diabetes, plotROC = T)

#viz of rsults

col <- c("#ed3b3b", "#0099ff")

graphics::fourfoldplot(cm_rf$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("Random Forest Accuracy(",round(cm_rf$overall[1]*100),"%)", sep = ""))
graphics::fourfoldplot(cm_xgb$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("XGB Accuracy(",round(cm_xgb$overall[1]*100),"%)", sep = ""))

graphics::fourfoldplot(cm_rpart$table, color = col, conf.level = 0.95, margin = 1, 
                       main = paste("Rpart DT Accuracy(",round(cm_rpart$overall[1]*100),"%)", sep = ""))


model_forest <- caret::train(diabetes ~., data = train_set,
                             method = "ranger",
                             metric = "ROC",
                             trControl = trainControl(method = "cv", number = 10,
                                                      classProbs = T, summaryFunction = twoClassSummary),
                             preProcess = c("center","scale","pca"))

pred_rf <- predict(model_forest, test_set)
cm_rf <- confusionMatrix(pred_rf, test_set$diabetes, positive="pos")
pred_prob_rf <- predict(model_forest, test_set, type="prob")
roc_rf <- roc(test_set$diabetes, pred_prob_rf$pos)
cm_rf
roc_rf
caTools::colAUC(pred_prob_rf$pos, test_set$diabetes, plotROC = T)

xgb_grid_1  <-  expand.grid(
    nrounds = 50,
    eta = c(0.03),
    max_depth = 1,
    gamma = 0,
    colsample_bytree = 0.6,
    min_child_weight = 1,
    subsample = 0.5
)

model_xgb <- caret::train(diabetes ~., data = train_set,
                          method = "xgbTree",
                          metric = "ROC",
                          tuneGrid=xgb_grid_1,
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = T, summaryFunction = twoClassSummary),
                          preProcess = c("center","scale","pca"))

pred_xgb <- predict(model_xgb, test_set)
cm_xgb <- confusionMatrix(pred_xgb, test_set$diabetes, positive="pos")
pred_prob_xgb <- predict(model_xgb, test_set, type="prob")
roc_xgb <- roc(test_set$diabetes, pred_prob_xgb$pos)
cm_xgb
roc_xgb
caTools::colAUC(pred_prob_xgb$pos, test_set$diabetes, plotROC = T)
