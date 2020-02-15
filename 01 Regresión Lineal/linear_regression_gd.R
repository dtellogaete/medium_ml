# Función de Descenso de Gradiente

LinearRegressionGD = function(lrate = 0.1, niter = 10000,
                              X, y, theta){
  const = lrate*(1/length(X))
  for(i in 1:niter){
    h = X*theta[2]+theta[1]
    theta[1] = theta[1]-const*(sum(h-y))
    theta[2] = theta[2]-const*(sum(h-y))*X
  }
  return(theta)
}

# Importar el dataset
dataset = read.csv('Admission_Predict_Ver1.1.csv', sep = ",")
dataset = dataset[1:length(dataset$GRE.Score), c(2,9)]

# Selección conjunto de entrenamiento y test
library(caTools)
set.seed(0)
split = sample.split(dataset$GRE.Score, SplitRatio = 0.75)
training = subset(dataset, split == TRUE)
testing = subset(dataset, split == FALSE)

# Escalado de variables
training = data.frame(scale(training))
testing = data.frame(scale(testing))

# Aplicación del modelo
linearRegression = LinearRegressionGD(lrate = 0.1, niter = 20000,
                                      X = as.numeric(training$GRE.Score),
                                      y = as.numeric(training$Chance.of.Admit),
                                      c(0.1, 1.0))

ypred = testing$GRE.Score*linearRegression[2]+
  linearRegression[1]

# Aplicación del modelo con la librería de R 
regression = lm(formula =  Chance.of.Admit ~ GRE.Score,
               data = training)

# Representación Grafíca
library(ggplot2)

ggplot()+
  geom_point(aes(x=testing$GRE.Score,
                 y=testing$Chance.of.Admit),
             colour = "red")+
  geom_line(aes(x=testing$GRE.Score,
                y=ypred, colour = "blue"),
             alpha = 1,
            size= 0.8)+
  geom_line(aes(x=testing$GRE.Score,
                 y=predict(regression, newdata = testing),
                 colour = "green"), alpha = 1,
             size= 0.8)+
  scale_color_discrete(name = "Modelo", labels = c("Descenso por gradiente", 
                                                   "Regresión Lineal (lm)"))+
  ggtitle("Probabilidad admisión vs GRE Score (Conjunto de Test)")+
  xlab("GRE Score")+
  ylab("Probabilidad admisión a Postgrado")
 

