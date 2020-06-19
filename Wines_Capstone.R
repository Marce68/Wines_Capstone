
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ellipse)) install.packages("ellipse", repos = "http://cran.us.r-project.org")


options(digits=5)

# Importing data
rw <- read.csv("~/R/Data/winequality-red.csv")
str(rw)

rw %>% head(10) %>% knitr::kable()

###############################
# Exploration and visualization

# It is preferrable to treat quality score as a factor
rw$quality <- as.factor(rw$quality)
str(rw)

# Let's see if we can find any type of evident separation in the dataset
rw %>% pivot_longer(colnames(rw)[-which(colnames(rw) == "quality")],"CF") %>%
  ggplot(aes(quality, value, fill = quality)) +
  geom_boxplot() +
  facet_wrap(~CF, scales = "free", ncol = 3) +
  theme(axis.text.x = element_blank(), legend.position="bottom")

# Let's see which are the main average differences between bad and good red wines
avg_rw <- rw %>%
  group_by(quality) %>%
  summarize(fixed.acidity = sum(fixed.acidity)/n(),
            volatile.acidity = sum(volatile.acidity)/n(),
            citric.acid = sum(citric.acid)/n(),
            residual.sugar = sum(residual.sugar)/n(),
            chlorides = sum(chlorides)/n(),
            free.sulfur.dioxide = sum(free.sulfur.dioxide)/n(),
            total.sulfur.dioxide = sum(total.sulfur.dioxide)/n(),
            density = sum(density)/n(),
            pH = sum(pH)/n(),
            sulphates = sum(sulphates)/n(),
            alcohol = sum(alcohol)/n(),)

# three lines of code to fit the table in the page
avg_rw_t <- as.data.frame(as.matrix(t(avg_rw)))
colnames(avg_rw_t) <- rep("Q_Score",6)
avg_rw_t %>%  knitr::kable()
rm(avg_rw_t)

# The variables are scarcely correlated apart from those related to acidity
ctab <- cor(rw[,1:11])
round(ctab, 2)
plotcorr(ctab, mar = c(0.1, 0.1, 0.1, 0.1))

# Note that the dataset is quite unbalanced: low samples count in low and high quality classes
rw %>%
  group_by(quality) %>%
  summarize(Wines = n()) %>%
  knitr::kable()



####################################
# Let's build our train and test set
# Test set is 20% of the full set
set.seed(1, sample.kind="Rounding") # `set.seed(1)` for R version <= 3.5
tst_idx <- createDataPartition(y = rw$quality,
                               times = 1,
                               p = 0.2,
                               list = FALSE)
rw_train <- rw[-tst_idx, ]
rw_test <- rw[tst_idx, ]

# We have very few wines with 3 and 8 in the train set
rw_train %>%
  group_by(quality) %>%
  summarize(Wines = n()) %>%
  knitr::kable()

# And even less in the test set 
rw_test %>%
  group_by(quality) %>%
  summarize(Wines = n()) %>%
  knitr::kable()


#################################################
# Let's test now some algorithm to chose the best

# Let's try some trivial classification based on cutoffs
cut_offs <- (avg_rw$sulphates[1:5] + avg_rw$sulphates[2:6])/2
y_hat <- rep("0", length(rw_test$quality))

# Let's see if the variable sulphates contain the information we need
y_hat[which(rw_test$sulphates <= cut_offs[1])] <- 3
y_hat[which(rw_test$sulphates > cut_offs[1] & rw_test$sulphates <= cut_offs[2])] <- 4
y_hat[which(rw_test$sulphates > cut_offs[2] & rw_test$sulphates <= cut_offs[3])] <- 5
y_hat[which(rw_test$sulphates > cut_offs[3] & rw_test$sulphates <= cut_offs[4])] <- 6
y_hat[which(rw_test$sulphates > cut_offs[4] & rw_test$sulphates <= cut_offs[5])] <- 7
y_hat[which(rw_test$sulphates > cut_offs[5])] <- 8

y_hat <- as.factor(y_hat)

confusionMatrix(y_hat, rw_test$quality)$table
confusionMatrix(y_hat, rw_test$quality)$overall["Accuracy"]

# Let's try KNN although we know that it's not performing well on unbalanced datasets
rw_knn <- train(quality ~ .,
                method = "knn",
                data = rw_train)

ggplot(rw_knn, highlight = TRUE)
rw_knn$bestTune
rw_knn$finalModel

y_hat_knn <- predict(rw_knn, rw_test, type = "raw")
confusionMatrix(y_hat_knn, rw_test$quality)$overall["Accuracy"]
confusionMatrix(y_hat_knn, rw_test$quality)$table

# KNN & CV
control <- trainControl(method = "cv", number = 10, p = .9)
rw_knn_cv <- train(quality ~ ., method = "knn", 
                   data = rw_train,
                   tuneGrid = data.frame(k = seq(9, 50, 2)),
                   trControl = control)
ggplot(rw_knn_cv, highlight = TRUE)
rw_knn_cv$bestTune
rw_knn_cv$finalModel

y_hat_knn_cv <- predict(rw_knn_cv, rw_test, type = "raw")
confusionMatrix(y_hat_knn_cv, rw_test$quality)$overall["Accuracy"]
confusionMatrix(y_hat_knn_cv, rw_test$quality)$table


# Classification tree with some tuning
rw_rpart <- train(quality ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = rw_train)
plot(rw_rpart, highlight = TRUE)
rw_rpart$bestTune
rw_rpart$finalModel

y_hat_rpart <- predict(rw_rpart, rw_test, type = "raw")
confusionMatrix(y_hat_rpart, rw_test$quality)$overall["Accuracy"]
confusionMatrix(y_hat_rpart, rw_test$quality)$table


#####################
# Random Forest
set.seed(1, sample.kind="Rounding") # `set.seed(1)` for R version <= 3.5
nodesize <- seq(1, 51, 10)
acc <- sapply(nodesize, function(ns){
  train(quality ~ .,
        method = "rf",
        data = rw_train,
        tuneGrid = data.frame(mtry = 2),
        nodesize = ns)$results$Accuracy
})
qplot(nodesize, acc)

set.seed(1, sample.kind="Rounding") # `set.seed(1)` for R version <= 3.5
rw_rf <- train(quality ~ .,
                  method = "rf",
                  tuneGrid = data.frame(mtry = seq(1:5)),
                  nodesize = nodesize[which.max(acc)],
                  data = rw_train)
plot(rw_rf)
rw_rf$bestTune
rw_rf$finalModel

y_hat_rf <- predict(rw_rf, rw_test, type = "raw")
confusionMatrix(y_hat_rf, rw_test$quality)$overall["Accuracy"]
confusionMatrix(y_hat_rf, rw_test$quality)$table


# Let's do some oversampling
y <- rw$quality
x <- rw[ ,1:11]
usrw <- upSample(x, y, list=FALSE, yname = "quality")

# Now the dataset is perfectly balanced
usrw %>%
  group_by(quality) %>%
  summarize(Q = n()) %>%
  knitr::kable()

#############################################################
# Let's build our train and test set again after oversampling
set.seed(1968, sample.kind="Rounding") # `set.seed(1)` for R version <= 3.5
tst_idx <- createDataPartition(y = usrw$quality,
                               times = 1,
                               p = 0.2,
                               list = FALSE)
usrw_train <- usrw[-tst_idx, ]
usrw_test <- usrw[tst_idx, ]

# KNN after oversampling
usrw_knn <- train(quality ~ .,
                  method = "knn",
                  data = usrw_train)

ggplot(usrw_knn, highlight = TRUE)
usrw_knn$bestTune
usrw_knn$finalModel

y_hat_knn <- predict(usrw_knn, usrw_test, type = "raw")
confusionMatrix(y_hat_knn, usrw_test$quality)$overall["Accuracy"]
confusionMatrix(y_hat_knn, usrw_test$quality)$table


##################################
# Random Forest after oversampling
set.seed(1968, sample.kind="Rounding") # `set.seed(1)` for R version <= 3.5
usrw_rf <- train(quality ~ .,
                 method = "rf",
                 tuneGrid = data.frame(mtry = 2),
                 nodesize = 2, # 11?
                 data = usrw_train)
plot(usrw_rf)
usrw_rf$bestTune
usrw_rf$finalModel

y_hat_usrf <- predict(usrw_rf, usrw_test, type = "raw")
confusionMatrix(y_hat_usrf, usrw_test$quality)$overall["Accuracy"]
confusionMatrix(y_hat_usrf, usrw_test$quality)$table



