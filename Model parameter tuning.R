#### Hyperparameter tuning of RF ####
# Load packet
library(randomForest)
library(readxl)
library(writexl)
library(ggplot2)
library(caret)

# Data loading
train <-read_excel(file.choose())
train $ Y <- as.factor(train $ Y)

# Parameter setting
mtry_values <- c(2, 4, 6, 8) 
ntree_max <- 1000  
results <- list()

# Build model and record the OOB error
for (mtry in mtry_values) {
  set.seed(123)  
  model <- randomForest(Y ~ ., data = train, mtry = mtry, ntree = ntree_max)
  results[[as.character(mtry)]] <- model$mse
}
# Display the results of hyperparameter tuning
print(model$mse)
# Draw curve graph
colors <- c("#67A9CF", "#2166AC", "#EF8A62", "#B2182B")
plot(1:ntree_max, results[[1]], type = "l", col = colors[1],
     xlab = "Number of Trees (ntree)", ylab = "OOB Error",
     ylim = range(unlist(results)), main = "The parameter tuning results of the RF")
for (i in 2:length(mtry_values)) {
  lines(1:ntree_max, results[[i]], col = colors[i])
}
legend("topright", legend = paste("mtry =", mtry_values),
       col = colors, lty = 1, cex = 0.8)


#### Hyperparameter tuning of XGboost ####
# Load packet
# install.packages("xgboost")
library(xgboost)
library(ggplot2)
library(readxl)
library(reshape2)

# Data loading
data <- read_excel(file.choose()) 
target_col <- "Y"  

# Divide the data into feature matrices and labels
features <- data[, !colnames(data) %in% target_col]
label <- data[[target_col]]

# Convert to DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(features), label = label)

#  Hyperparameter tuning settings
params_grid <- expand.grid(
  eta = seq(0.1, 1, by = 0.1),       
  gamma = seq(1, 5, by = 0.5),        
  max_depth = 6:8       
)
results <- data.frame()

# Hyperparameter tuning loop
for(i in 1:nrow(params_grid)){
  current_params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = params_grid$eta[i],
    gamma = params_grid$gamma[i],
    max_depth = params_grid$max_depth[i]
  )
  
  set.seed(123)
  cv_model <- xgb.cv(
    params = current_params,
    data = dtrain,
    nrounds = 200,
    nfold = 10,
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  best_iter <- cv_model$best_iteration
  best_logloss <- cv_model$evaluation_log$test_logloss_mean[best_iter]
  
  results <- rbind(results, data.frame(
    eta = params_grid$eta[i],
    gamma = params_grid$gamma[i],
    max_depth = params_grid$max_depth[i],
    logloss = best_logloss,
    nrounds = best_iter
  ))
}

# Draw a three-dimensional hyperparameter surface graph
ggplot(results, aes(x = eta, y = factor(max_depth), fill = logloss)) +
  geom_tile() +
  facet_wrap(~ gamma, nrow = 1) +
  scale_fill_gradient(low = "green", high = "red") +
  labs(x = "Learning Rate", y = "Max Depth", fill = "LogLoss") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#  Optimal hyperparameter output
optimal_params <- results[which.min(results$logloss), ]
cat("Optimal parameter combination：\n",
    "eta =", optimal_params$eta, "\n",
    "gamma =", optimal_params$gamma, "\n",
    "max_depth =", optimal_params$max_depth, "\n")


#### Hyperparameter tuning of SVM ####
# Load packet
library(e1071)
library(caret)
library(readxl)
library(writexl)

# Data loading
train1 <-read_excel(file.choose())
y <- train1 $ Y

# Data normalization
set.seed(123)
preProc <- preProcess(train1[, 1:18])
x <- predict(preProc, train1)

# Cross-validation
set.seed(123)
tune_result <- tune.svm(
  x = x,
  y = y,
  type = "eps-regression", 
  kernel = "radial",
  cost = seq(1, 10, by = 1), 
  gamma = seq(0.01, 0.1, by = 0.01),
  tunecontrol = tune.control(
    sampling = "cross",
    cross = 10
  )
)

print(tune_result$best.parameters)

# Optimal hyperparameter output
summary(tune_result)
best_model <- svm(
  x = x,
  y = y,
  type = "eps-regression",
  kernel = "radial",
  cost = tune_result$best.parameters$cost,
  gamma = tune_result$best.parameters$gamma)
summary(best_model)
results <- as.data.frame(tune_result$performances)
print(colnames(results))

# Generate heat map (including MSE values)
ggplot(results, aes(x = factor(cost), y = factor(gamma))) +
  geom_tile(aes(fill = error), color = "gray80") +
  geom_text(
    aes(label = sprintf("%.3f", error)), 
    color = "white",
    size = 3.5) +
  scale_fill_gradient(
    low = "#2c7bb6",
    high = "#d7191c",
    name = "MSE") +
  labs(
    x = "Cost",
    y = "Gamma",
    ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"), 
        panel.grid = element_blank()) 

#### Hyperparameter tuning of ANN ####
# Load packet
library(nnet)
library(ggplot2)
library(caret)

# Data loading
data <-read_excel(file.choose())

# Ensure that the target variable is a factor
target_col <- ncol(data)
data[[target_col]] <- as.factor(data[[target_col]])
formula <- reformulate(names(data)[-target_col], names(data)[target_col])

# Data normalization
preproc <- preProcess(data[-target_col], method = c("center", "scale", "zv"))
data_norm <- predict(preproc, data)

#  Hyperparameter tuning settings
param_grid <- expand.grid(
  size = 1:10,
  decay = seq(0.1, 0.5, by = 0.1)
)

# Cross-validation
k <- 10
set.seed(123)
folds <- createFolds(data_norm[[target_col]], k = k)

# Enhanced result storage
results <- data.frame(
  Size = numeric(),
  Decay = numeric(),
  Accuracy = numeric(),
  ErrorCount = numeric()
)

# Hyperparameter tuning loop
for(i in 1:nrow(param_grid)){
  current_size <- param_grid$size[i]
  current_decay <- param_grid$decay[i]
  accuracies <- numeric(k)
  error_count <- 0
  for(j in 1:k){
    tryCatch({
      train_data <- data_norm[-folds[[j]], ]
      test_data <- data_norm[folds[[j]], ]
      ann_model <- nnet(
        formula,
        data = train_data,
        size = current_size,
        decay = current_decay,
        linout = FALSE, 
        trace = FALSE,
        maxit = 1000,
        MaxNWts = 1000
      )
      if(exists("ann_model")){
        pred <- predict(ann_model, test_data, type = "class")
        accuracies[j] <- mean(pred == test_data[[target_col]])
      } else {
        accuracies[j] <- NA
        error_count <- error_count + 1
      }
    }, error = function(e) {
      message(sprintf("parameter combination size=%d decay=%.1f The %d fold is incorrect: %s",
                      current_size, current_decay, j, e$message))
      accuracies[j] <<- NA
      error_count <<- error_count + 1
    })
  }
  
  # object record
  valid_acc <- accuracies[!is.na(accuracies)]
  mean_acc <- ifelse(length(valid_acc) > 0, mean(valid_acc), NA)
  
  results <- rbind(results, data.frame(
    Size = current_size,
    Decay = current_decay,
    Accuracy = mean_acc,
    ErrorCount = error_count
  ))
}

clean_results <- results[complete.cases(results), ]

# Visual analysis: Heat map
ggplot(clean_results, aes(x = factor(Size), y = factor(Decay), fill = Accuracy)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.3f\n(E=%d)", Accuracy, ErrorCount)),
            color = "black", size = 3) +
  scale_fill_gradient(low = "#2166AC", high = "#B2182B") +
  labs(x = "The number of neurons in the hidden layer (decay)",
       y = "Weight attenuation coefficient (size)") +
  theme_bw()

# Optimal parameter output
optimal_params <- clean_results[which.max(clean_results$Accuracy), ]
cat("Optimal parameter combination：\n",
    "Size =", optimal_params$Size, "\n",
    "Decay =", optimal_params$Decay, "\n",
    "Accuracy  =", round(optimal_params$Accuracy, 4),
    "(Number of errors ：", optimal_params$ErrorCount, ")")
