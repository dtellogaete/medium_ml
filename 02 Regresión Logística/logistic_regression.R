# Regresion logística

# Importar dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Selección conjunto de entrenamiento y test
library(caTools)
set.seed(0)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training = subset(dataset, split == TRUE)
testing = subset(dataset, split == FALSE)

# Escalado de variables
training[1:2] = data.frame(scale(training[1:2]))
testing[1:2] = data.frame(scale(testing[1:2]))


# Aplicación de modelo de regresión logística con la función glm
logistic = glm(formula = Purchased ~ .,
                 data = training,
                 family = binomial)

# Predicción de los resultados conjunto de testing
prob_pred = predict(logistic, type = "response",
                    newdata = testing[,-3])

y_predict = ifelse(prob_pred>=0.5, 1, 0)

# Matriz de confusión
cm = table(testing[,3], y_predict)

