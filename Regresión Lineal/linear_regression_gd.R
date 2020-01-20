# Función de Descenso de Gradiente

LinearRegressionGD = function(lrate = 0.1, niter = 10000,
                              X, y, theta){
  for(i in 1:niter){
    h = X*theta[2]+theta[1]
    const = lrate*(1/length(X))
    theta[1] = theta[1]-const*(sum(h-y))
    theta[2] = theta[2]-const*(sum(h-y))*X
  }
  return(theta)
}

# Importar el dataset
dataset = read.csv('consumo_cerveja.csv', sep = ",")
dataset = dataset[1:365, c(2,7)]
dataset$Temperatura.Media..C. = as.numeric(dataset$Temperatura.Media..C.)/10

# Selección conjunto de entrenamiento y test
library(caTools)
set.seed(0)
split = sample.split(dataset$Temperatura.Media..C., SplitRatio = 0.75)
training = subset(dataset, split == TRUE)
testing = subset(dataset, split == FALSE)



# Aplicación del modelo
linearRegression = LinearRegressionGD(lrate = 0.0005, niter = 100000,
                                      X = training$Temperatura.Media..C.,
                                      y = training$Consumo.de.cerveja..litros.,
                                      c(18, 0.8))

ypred = testing$Temperatura.Media..C.*linearRegression[2]+
  linearRegression[1]

# Aplicación del modelo con la librería de R 
regression = lm(formula =  Consumo.de.cerveja..litros. ~ Temperatura.Media..C.,
               data = training)



# Representación Grafíca
library(ggplot2)

ggplot()+
  geom_point(aes(x=testing$Temperatura.Media..C.,
                 y=testing$Consumo.de.cerveja..litros.),
             colour = "red")+
  geom_line(aes(x=testing$Temperatura.Media..C.,
                y=ypred, colour = "blue"),
             alpha = 1,
            size= 0.8)+
  geom_line(aes(x=testing$Temperatura.Media..C.,
                y=predict(regression, newdata = testing),
                colour = "forestgreen"), alpha = 1,
            size= 0.8)+
  scale_color_discrete(name = "Modelo", labels = c("Descenso por gradiente", 
                                                   "Regresión Lineal (lm)"))+
  ggtitle("Consumo de Cerveza vs Temperatura Media (Conjunto de testing) ")+
  xlab("Temperatura Media en Sao Paulo (°C)")+
  ylab("Consumo de cerveza (L)")
 

