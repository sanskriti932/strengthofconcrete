install.packages("ggplot2")
install.packages("mice")
install.packages("caret")
install.packages("randomForest")
install.packages("rpart.plot")
library(rpart)
library(caret)
library(randomForest)
library(psych)
library('ggplot2')
library(mice)
library(mlbench)
library(tidyverse)
library(pROC)

#reading the data from csv
concretestrength_train <- read.csv("/Users/sanskritilamichhane/Downloads/concrete_strength_train.csv")
concretestrength_test <- read.csv("/Users/sanskritilamichhane/Downloads/concrete_strength_test.csv")

#training and testing column added
concretestrength_train$isTrain<-"yes"
concretestrength_test$isTrain<-"no"

#View(concretestrength_train)
#View(concretestrength_test)

#binding the testing and training dataset in concretestrdataset
concretestrdataset <- rbind(concretestrength_train,concretestrength_test)
#View(concretestrdataset)

#number of rows and columns in dataset
nrow(concretestrdataset)
ncol(concretestrdataset)

#number of dimension in dataset
dim(concretestrdataset)

#missing na values calculation
is.na(concretestrdataset)
sum(is.na(concretestrdataset))

#summary of the dataset
summary(concretestrdataset)

#description of dataset
describe(concretestrdataset)

#Structure of dataset
str(concretestrdataset)

#finding missing values
missingvalues <- (apply(concretestrdataset,2,function(x) sum(is.na(x)))/nrow(concretestrdataset))
missingvalues

#average of missing values
averagemissingvalues <- mean(missingvalues)
averagemissingvalues

#before imputation md.pattern
md.pattern(concretestrdataset)

#imputation of missing values via rf and calculating its md.pattern
imputedrf_concretestr<-mice(data=concretestrdataset,m=1,method="rf",maxit=10)
rf_concretestrimputed <- complete(imputedrf_concretestr)
md.pattern(rf_concretestrimputed)

#imputation of missing values via mean and calculating its md.pattern
imputedmean_concretestr<-mice(data=concretestrdataset,m=1,method="mean",maxit=10)
mean_concretestrimputed <- complete(imputedmean_concretestr)
md.pattern(mean_concretestrimputed)

#histogram from different imputation methods
hist(rf_concretestrimputed$Cement, main = "Cement Distribution (RF Imputed)", 
     xlab = "Cement", ylab = "Frequency", col = "orchid", border = "black")

hist(mean_concretestrimputed$Cement, main = "Cement Distribution (Mean Imputed)", 
     xlab = "Cement", ylab = "Frequency", col = "salmon", border = "black")


#Outlier detection
# Data frame creation for storing outliers
outliers <- data.frame()

# Looping through each variable and calculating Q1, Q3 and IQR
for (i in 1:9) {
  
  Q1 <- quantile(rf_concretestrimputed[[i]], 0.25)
  Q3 <- quantile(rf_concretestrimputed[[i]], 0.75)
  IQR <- Q3 - Q1
  
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  
  outliers <- rbind(outliers, rf_concretestrimputed[rf_concretestrimputed[[i]] < lower_bound | rf_concretestrimputed[[i]] > upper_bound, ])
}
# Removing outliers
outliers <- unique(outliers)
outliers

# Extract the first 9 columns of the imputed data
rf_concretestrimputed_numeric <- as.data.frame(rf_concretestrimputed)[, 1:9]


#box plot of outliers     
par(mfrow=c(2,5))         
for(i in 1:9){      
  boxplot(rf_concretestrimputed[i],col="cyan",main="Box Plot",xlab=colnames(rf_concretestrimputed)[[i]])
}

# Remove outliers from the original dataset
filtered_data <- rf_concretestrimputed[!(rownames(rf_concretestrimputed) %in% rownames(outliers)), ]
filtered_data
View(filtered_data)


#Model Predictions
#Model-1 
#Random Forest
# Missing Values Check
any(is.na(filtered_data))


#Training and Testing dataset splitted
train <- filtered_data[,1:9][filtered_data$isTrain=="yes",]
test <- filtered_data[,1:9][filtered_data$isTrain=="no",]


# Model Development using randomForest function
# Training control object created for 10 sets of cross fold validation
traincontrolobj <- trainControl(method = "cv", number = 10)

# Optimisation of parameters for modelling
oob_error <- double(9)
test_error <- double(9)

for(mtry in 1:9) {
  rf <- randomForest(Strength ~ ., train, mtry = mtry, ntree = 500)
  oob_error[mtry] <- rf$mse[500] #Fitted Trees Error
  prediction <- predict(rf, test) #Test Set Prediction for Trees
  test_error[mtry] <- mean( (test$Strength - prediction) ** 2) #Mean squared error
}

matplot(1:mtry , cbind(oob_error, test_error), pch = 19 , col = c("green", "yellow"), type = "b",
        ylab = "Mean Squared Error", xlab = "Number of Predictors Considered at each Split")
legend("topright", legend = c("Out of Bag Error", "Test Error"), pch = 19, col = c("green", "yellow"))

test_error
oob_error


#Minimum test error index
mintesterror_index <- which.min(test_error)

# Corresponding mtry value for minimum test error
bestmtryvalue <- mintesterror_index

#Mtry value obtained
print(paste("Best mtry value for lowest test error:", bestmtryvalue))

#Training model using random forest function for training dataset
filtered_data.rf <- randomForest(Strength ~ ., 
                                 data = train,
                                 method = "rf",
                                 trControl = traincontrolobj)


print(filtered_data.rf)

#Performance metrics for random forest accessed
performancemetrics_randomforest <- filtered_data.rf$results

#Random Forest model plotting
plot(filtered_data.rf)

#Test set prediction conducted
prediction <- predict(filtered_data.rf, test)

#Calculation of performance metrics like RMSE, R2 and MAE
rmseperformance_met = RMSE(prediction,test$Strength)
r2performance_met = R2(prediction, test$Strength)
maeperformance_met = MAE(prediction, test$Strength)
c(rmseperformance_met,r2performance_met,maeperformance_met)

#Rsquare Adjusted
n <- nrow(test)  
p <- ncol(test) - 1  
adjusted_r2 <- 1 - (1 - r2performance_met) * (n - 1) / (n - p - 1)

# Calculate Mean Squared Error (MSE)
mserror <- mean((test$Strength - prediction)^2)

# Combine all metrics
metricscalculated <- c(rmseperformance_met, r2performance_met, adjusted_r2, maeperformance_met, mserror)
metricscalculated

#Predicted data vs Observed data plotting
df <- data.frame(pred = prediction, obs = test$Strength)


# Plot predicted vs observed data with color
ggplot(df, aes(x = obs, y = pred, color = abs(obs - pred))) + 
  geom_point() +
  scale_color_gradient(low = "blue", high = "red", name = "Absolute Error") +
  labs(x = "Observed Strength", y = "Predicted Strength", title = "Predicted vs Observed Data")


#Analysis of residual data
prediction <- predict(filtered_data.rf, test)
residuals_data <- test$Strength - prediction


#Residual vs fitted values plotting
plot(x = prediction, y = residuals_data, main = "Residuals vs. Fitted Values", xlab = "Fitted Values", 
     ylab = "Residuals")
abline(h = 0, col = "cyan")

qqnorm(residuals_data)
qqline(residuals_data)


#Residual data plotting
par(mfrow = c(2, 5))
for (col in colnames(test)[-ncol(test)]) {
  plot(x = test[[col]], y = residuals_data, main = paste("Residuals vs.", col), xlab = col, ylab = "Residuals",col="lightgoldenrod")
}


#Model-2
#KNN

# Define normalize function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalize columns 1 to n-1 in the training set
train_data_norm <- as.data.frame(lapply(train[, -ncol(train)], normalize))

# Add the target variable to the normalized training set
train_data_norm$Strength <- train$Strength

# Normalize columns 1 to n-1 (excluding the target variable) in the testing set
test_data_norm <- as.data.frame(lapply(test[, -ncol(test)], normalize))

# Add the target variable to the testing set
test_data_norm$Strength <- test$Strength

# Dispersing data into features (X) and target variable (y)
X_train <- train_data_norm[, -ncol(train_data_norm)]  # Features for training
y_train <- train_data_norm$Strength  # Target variable for training
X_test <- test_data_norm[, -ncol(test_data_norm)]  # Features for testing
y_test <- test_data_norm$Strength  # Target variable for testing

kvalueoptimal <- round(sqrt(nrow(train_data_norm)))


# Train KNN regression model using cross-validation (5-fold)
set.seed(123)  # For reproducibility
knn_model <- train(
  x = X_train,                       # Features
  y = y_train,                       # Target variable
  method = "knn",                    # KNN regression method
  trControl = trainControl(method = "cv", number = 5),  # 5-fold cross-validation
  tuneGrid = expand.grid(k = kvalueoptimal)   # Search grid for K values from 1 to 30
)


# Print the optimal k value
cat("K value optimal:", optimal_k, "\n")

# Test Set Prediction
y_prediction <- predict(knn_model, newdata = X_test)

# Residuals calculation
residuals <- y_test - y_prediction
# RMSE Calculation
RMSE <- sqrt(mean(residuals^2))

# MAE Calculation
MAE <- mean(abs(residuals))

# MSE Calculation
MSE <- mean(residuals^2)

#Rsquared Calculation
R_squared <- 1 - sum(residuals^2) / sum((y_test - mean(y_test))^2)

# Calculate adjusted R-squared
n <- length(y_test)
p <- ncol(X_test)
adjusted_R_squared <- 1 - (1 - R_squared) * ((n - 1) / (n - p - 1))

# Create a data frame to store performance metrics
performance_metrics <- data.frame(
  Metric = c("RMSE", "MAE", "MSE", "R-squared", "Adjusted R-squared"),
  Value = c(RMSE, MAE, MSE, R_squared, adjusted_R_squared)
)

# Print the table of performance metrics
print(performance_metrics)

# Plot predicted vs. original values
plot(y_test, y_prediction, main = "Predicted vs. Original Values",
     xlab = "Original Values", ylab = "Predicted Values",col="violet")
abline(0, 1, col = "green")

# Plot residuals
plot(residuals, main = "Residuals Plot", xlab = "Index", ylab = "Residuals",col="red")


#Model-3
#Logistic Regression

train_data <- filtered_data[filtered_data$isTrain == "yes", ]
test_data <- filtered_data[filtered_data$isTrain == "no", ]


#considering the threshold of high and low strength

q1 <- quantile(train_data$Strength, 0.25)
q3 <- quantile(train_data$Strength, 0.75)
print(q3)


filtered_df <- train_data[train_data$Strength < q3 | train_data$Strength > q1, ]
filtered_df$Strength <- ifelse(filtered_df$Strength > q3, 1, 0)


datasetsregression = filtered_df
datasetsregression$Strength = filtered_df$Strength

datafortraining = datasetsregression

# Fit logistic regression model
logit <- glm(Strength ~ Cement, family = binomial, data = datasetsregression)
summary(logit)

# Create dataset for plotting
trainset <- mutate(datasetsregression, prob = ifelse(Strength == 1, 1, 0))

# Plot logistic regression model
ggplot(trainset, aes(Cement, prob)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(
    title = "Logistic Regression Model",
    x = "Cement Concentration",
    y = "Probability of High Strength"
  )



#checking the probability for test data
newdata <- data.frame(Cement = test_data$Cement)
probabilities <- predict(logit, newdata, type = "response")
probabilities

# Create a ROC curve object
roc_obj <- roc(ifelse(test_data$Strength > q3, 1, 0), probabilities)

# Plot the ROC curve
plot(roc_obj, main = "ROC Curve", col = "cyan")

# Compute the AUC
auc_result <- auc(roc_obj)
print(paste("AUC:", auc_result))

# Residual plot
plot(logit, which = 1, col = "orange", main = "Residual Plot")

# Original vs. Predicted plot
predicted <- predict(logit, newdata, type = "response")
original <- ifelse(test_data$Strength > q3, 1, 0)  # Assuming your original dataset contains the Strength column
plot(original ~ predicted, col = "red", main = "Original vs. Predicted", xlab = "Predicted Probabilities",
     ylab = "Original Strength")


# Performance metrics
# Confusion matrix
predicted_classes <- ifelse(predicted > 0.5, 1, 0)
actual_classes <- ifelse(test_data$Strength > q3, 1, 0)
confusionMatrix(table(predicted_classes, actual_classes))


# Accuracy
accuracy <- sum(predicted_classes == actual_classes) / length(predicted_classes)
print(paste("Accuracy:", accuracy))


# Sensitivity and Specificity
conf_matrix <- confusionMatrix(table(predicted_classes, actual_classes))
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
print(paste("Sensitivity:", sensitivity))


# Compute kappa statistic,False Positives,False Negatives
FN <- conf_matrix$table[2,1]
FP <- conf_matrix$table[1,2]
kappa <- conf_matrix$overall["Kappa"]

# Output FN, FP, and Kappa
print(paste("False Negatives (FN):", FN))
print(paste("False Positives (FP):", FP))
print(paste("Kappa Statistic:", kappa))

#Model-4
#Naive Bayes

nb_filtered_data <- filtered_data

# Calculate threshold value
threshold_value <- quantile(nb_filtered_data$Strength, 0.75)

# Create binary outcome variable
nb_filtered_data$HighStrength <- ifelse(nb_filtered_data$Strength >= threshold_value, "High", "Low")

# Set seed for reproducibility
set.seed(123)

# Specify proportion for training data
train_proption <- 0.8

# Calculate number of rows for training data
train_size <- floor(train_proption * nrow(nb_filtered_data))

# Randomly sample row indices for training data
train_index <- sample(seq_len(nrow(nb_filtered_data)), size = train_size)

# Subset data into training and testing sets
train_data <- nb_filtered_data[train_index, ]
test_data <- nb_filtered_data[-train_index, ]

# Train the Naive Bayes model using caret with cross-validation
nb_fit <- train(
  HighStrength ~ .,  # Use your binary outcome variable as the dependent variable
  data = train_data,
  method = "naive_bayes",  # Specify the method as "naive_bayes" for Naive Bayes
  trControl = trainControl(method = "cv", number = 5),  # Cross-validation with 5 folds
  tuneLength = 10  # Tune the model with 10 parameter combinations
)

# Generate predicted probabilities for the test set
nbtest_predictionprob <- predict(nb_fit, newdata = test_data, type = "prob")

# Extract predicted probabilities for the "High" class
nbtesthighprobprediction <- nbtest_predictionprob[, "High"]

# Convert HighStrength to factor with levels "Low" and "High"
test_data$HighStrength <- factor(test_data$HighStrength, levels = c("Low", "High"))

# Create ROC curve
roc_obj <- roc(test_data$HighStrength, nbtest_predictionprob[, "High"], levels = c("High", "Low"), direction = "<")

# Plot ROC curve
plot(roc_obj, main = "ROC Curve for Naive Bayes Model",col="violet")
auc_value <- auc(roc_obj)

# Make predictions on the test set using the trained model
nb_predictionsset <- predict(nb_fit, newdata = test_data)

# Convert predicted values to factor with levels "Low" and "High"
nb_predictionsset <- factor(nb_predictionsset, levels = c("Low", "High"))

# Create confusion matrix
conf_matrix <- confusionMatrix(nb_predictionsset, test_data$HighStrength)
 
# Print confusion matrix
print(conf_matrix)

# Create a table with model performance metrics
performance_table <- data.frame(
  AUC = auc_value,
  Accuracy = conf_matrix$overall["Accuracy"],
  Sensitivity = conf_matrix$byClass["Sensitivity"],
  Specificity = conf_matrix$byClass["Specificity"],
  Balanced_Accuracy = conf_matrix$byClass["Balanced Accuracy"]
)

# Print the performance table
print(performance_table)

# Plot predicted vs. original values
plot(test_data$Strength, nbtest_predictionprob[, "High"], xlab = "Original Values", ylab = "Predicted Values", 
     main = "Predicted vs. Original Values")
abline(0, 1, col = "yellow")



#Model-5
#Descision Tree

#splitting into train and test
train <- filtered_data[,1:9][filtered_data$isTrain == "yes", ]
test <- filtered_data[,1:9][filtered_data$isTrain == "no", ]

#Fitting model
cart_fit <- rpart(Strength ~ . , data = train)
summary(cart_fit)

rpart.plot(cart_fit)


#model evaluation
filtered_data_pred <- predict(cart_fit,test,type="vector")
filtered_data_pred

#modeltuning and parameter optimisation
cart_fit$cptable
plotcp(cart_fit)

opt_filtered_data<- which.min(cart_fit$cptable[,'xerror'])
cp_filtered_data <- cart_fit$cptable[opt_filtered_data,'CP']
cp_filtered_data

pruned_fit<-prune(cart_fit,cp_filtered_data)
rpart.plot(pruned_fit)

pruned_predict<-predict(pruned_fit,test,type = "vector")
pruned_rmse <- sqrt(mean((pruned_predict - test$Strength)^2))
pruned_rmse

#Original Vs Predicted
plot(test$Strength, filtered_data_pred, main = "Actual vs Predicted Strength",
     xlab = "Actual Strength", ylab = "Predicted Strength")
abline(0, 1, col = "red")  # Add a diagonal line for reference


# Calculate Root Mean Squared Error (RMSE)
# Performance metrics
rmse <- sqrt(mean((filtered_data_pred - test$Strength)^2))

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(pruned_predict - test$Strength))

# Calculate R-squared (coefficient of determination)
SST <- sum((test$Strength - mean(test$Strength))^2)
SSE <- sum((pruned_predict - test$Strength)^2)
rsquared <- 1 - SSE/SST

# Calculate Adjusted R-squared
n <- nrow(test)  # Number of observations in the test set
p <- length(coef(pruned_fit)) - 1  # Number of predictors (excluding intercept)
adjusted_rsquared <- 1 - (1 - rsquared) * (n - 1) / (n - p - 1)

# Calculate Mean Squared Error (MSE)
mse <- mean((pruned_predict - test$Strength)^2)

cat(
  "Mean Squared Error (MSE) for pruned model:", mse, "\n",
  "R-squared (R2) for pruned model:", rsquared, "\n",
  "Adjusted R-squared (Adj R2) for pruned model:", adjusted_rsquared, "\n",
  "Root Mean Squared Error (RMSE):", rmse, "\n",
  "Mean Absolute Error (MAE) for pruned model:", mae, "\n"
)

#residual analysis
residuals <- test$Strength - pruned_predict
plot(residuals, main = "Residuals Plot", xlab = "Observation", ylab = "Residuals",col="pink")
plot(pruned_predict, residuals, main = "Residuals vs Fitted Values", xlab = "Fitted Values", ylab = "Residuals",col="blue")
abline(h = 0, col = "yellow")

qqnorm(residuals)
qqline(residuals)

hist(residuals, prob = TRUE, main = "Histogram of Residuals", xlab = "Residuals")
lines(density(residuals), col = "lightgreen")


