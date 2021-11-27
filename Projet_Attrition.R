#-----------------------#
#PREPARATION DES DONNES #
#-----------------------#

#Chargement des données

attrition <- read.csv("Data_Projet_1.csv", header = TRUE, sep = ',', dec = ".", stringsAsFactors = T)

# Verification de la bonne lecture du fichier
names(attrition)

# Verification des modes (types) de chaque variable
str(attrition)

#----------------------------------------------------#
# INSTALLATION ET ACTIVATION DE LA LIBRAIRIE ggplot2 #
#----------------------------------------------------#

install.packages("ggplot2")

library(ggplot2)


#----------------------------------------#
# STATISTIQUES GENERALES SUR LES DONNEES #
#----------------------------------------#

# Quartiles et moyenne des variables quantitatives et effectifs par valeur des variables qualitatives

summary(attrition)

#-----------------------------------#
# VISUALISATION MONODIMENSIONNELLES #
#-----------------------------------#

# Diagrammes circulaire en secteurs de variables discretes

pie(table(attrition$Attrition), main = "Répartition des classes")

#-------------------------------------------------------------------#
#Relation entre les variales explicatives et la variable a expliquer#
#-------------------------------------------------------------------#

#RELATION EXISTANT ENTRE LA VARIABLE  AGE ,Monthly_Rate ET ATTRITION 

#Boite a moustache 

boxplot(attrition$Age~attrition$Attrition , data = attrition , col="bisque")

# Histogramme d'effectifs de variables continues

qplot(attrition$Age, data=attrition, main="Distibution de Age", xlab="Valeur de Age", ylab="Nombre d'instances", binwidth=9, fill=attrition$Attrition)

# Points en couleurs selon leur classe attrition=Oui ou attrition=Non

qplot(attrition$Age, attrition$MonthlyRate, data=attrition, main="Nuage de point de Monthly rate et Age", xlab="Valeur de Age", ylab="Valeur de Monthly_Rate", color=attrition$Attrition)


#RELATION EXISTANT ENTRE LES VARIABLE  Employeecount,over18,StandarHours ET ATTRITION 


# Histogramme d'effectifs de variables continues

qplot(attrition$EmployeeCount, data=attrition, main="Distibution de Employeecount", xlab="Valeur de Employeecount", ylab="Nombre d'instances", binwidth=9, fill=attrition$Attrition)

qplot(attrition$Over18, data=attrition, main="Distibution de over18", xlab="Valeur de over18", ylab="Nombre d'instances", binwidth=9, fill=attrition$Attrition)

qplot(attrition$StandardHours, data=attrition, main="Distibution de standarHours", xlab="Valeur de standarHours", ylab="Nombre d'instances", binwidth=9, fill=attrition$Attrition)

#RELATION EXISTANT ENTRE Les VARIABLEs  Job_Role ,Hourly_Rate ET ATTRITION 

# Points en couleurs selon leur classe attrition=Oui ou attrition=Non

qplot(attrition$HourlyRate, attrition$JobRole, data=attrition, main="Nuage de point de Job_Role et Hourly_Rate", xlab="Valeur de Hourly_Rate", ylab="Valeur de Job_Rolle", color=attrition$Attrition)+ geom_jitter(reight = 0.3,height = 0.3)

#RELATION EXISTANT ENTRE Les VARIABLEs  MaritalStatus ,Monthly_Income ET ATTRITION 

# Points en couleurs selon leur classe attrition=Oui ou attrition=Non

qplot(attrition$MonthlyIncome, attrition$MaritalStatus, data=attrition, main="Nuage de point de MaritalStatus et Monthly_Income", xlab="Valeur de MonthlyRate", ylab="Valeur de MaritalStatus", color=attrition$Attrition)+ geom_jitter(reight = 0.5,height = 0.3)

#RELATION EXISTANT ENTRE Les VARIABLEs  DistanceFromeHome,  JobSatisfaction ET ATTRITION 

# Points en couleurs selon leur classe attrition=Oui ou attrition=Non

qplot(attrition$JobSatisfaction, attrition$DistanceFromHome, data=attrition, main="Nuage de point de DistanceFromHome et JobSatisfaction", xlab="Valeur de Job_Satisfaction", ylab="Valeur de DistanceFromHome", color=attrition$Attrition)+ geom_jitter(reight = 0.5,height = 0.3)

#----------------------------------------------------------------------------------#
# CREATION DES ENSEMBLES D'APPRENTISSAGE ET DE TEST AVEC ET SANS Quotient_Familial #
#----------------------------------------------------------------------------------#

#Création de l'ensemble d'apprentissage et de l'ensemble de test

#l'ensemble d'apprentissage aura 980 instances, ce qui correspond aux 2/3 de l'nsemble produit
#l'ensemble de test aura donc 490 instances, ce qui correspond au 1/3 de l'ensemble produit
attrition_EA <- attrition[1:980,]
attrition_ET <- attrition[981:1470,]

#Suppression des variables n'ayant pas d'incidence sur la prédiction

attrition_EA <- subset(attrition_EA, select=-EmployeeCount)
attrition_EA <- subset(attrition_EA, select=-Over18)
attrition_EA <- subset(attrition_EA, select=-StandardHours)

attrition_ET <- subset(attrition_ET, select=-EmployeeCount)
attrition_ET <- subset(attrition_ET, select=-Over18)
attrition_ET <- subset(attrition_ET, select=-StandardHours)

#-----------------------------------------#
#Installation et chargement des librairies#
#-----------------------------------------#
install.packages("rpart")
library(rpart)

install.packages("ROCR")
library(ROCR)

install.packages("e1071")
library(e1071)

install.packages("naivebayes")
library(naivebayes)

install.packages("nnet")

library(nnet)

install.packages("randomForest")
library(randomForest)


#-------------------------#
# ARBRE DE DECISION RPART #
#-------------------------#

# Definition de la fonction d'apprentissage,matrice de confusion, test et evaluation par courbe ROC
test_rpart <- function(arg1, arg2, arg3, arg4){
  
  # Apprentissage du classifeur
  dt <- rpart(Attrition~., attrition_EA, parms = list(split = arg1), control = rpart.control(minbucket = arg2))
  
  # Tests du classifieur : classe predite
  dt_class <- predict(dt, attrition_ET, type="class")
  
  # Matrice de confusion
  Mc_rpart <- table(attrition_ET$Attrition , dt_class)
  print(Mc_rpart)
  
  # Calcul du taux de succes du classifieur
  succes1 <- (Mc_rpart[1,1] + Mc_rpart[2,2]) / nrow(attrition_ET)
  cat("succes1 = ",succes1)
  
  # Calcul de la mesure du Taux de Vrais Négatifs pour evaluer la fiabilité des prédictions négatives du classifieur:VN/(VN+FN)
  TVN_rpart <- Mc_rpart[1,1] / (Mc_rpart[1,1] + Mc_rpart[2,1])
  cat("\n TVN_rpart = " ,TVN_rpart)
  
  
  
  # Tests du classifieur : probabilites pour chaque prediction
  dt_prob <- predict(dt, attrition_ET, type="prob")
  
  # Courbes ROC
  dt_pred <- prediction(dt_prob[,2], attrition_ET$Attrition)
  dt_perf <- performance(dt_pred,"tpr","fpr")
  plot(dt_perf, main = "Arbres de decision rpart()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  dt_auc <- performance(dt_pred, "auc")
  cat("\n AUC = ", as.character(attr(dt_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-------------------------------------------------#
# APPRENTISSAGE DES CONFIGURATIONS ALGORITHMIQUES #
#-------------------------------------------------#

# Arbres de decision
test_rpart("gini", 10, FALSE, "red")
test_rpart("gini", 5, TRUE, "blue")
test_rpart("information", 10, TRUE, "green")
test_rpart("information", 5, TRUE, "orange")



#----------------#
# RANDOM FORESTS #
#----------------#

# Definition de la fonction d'apprentissage, matrice de confusion, test et evaluation par courbe ROC
test_rf <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  rf <- randomForest(Attrition~., attrition_EA, ntree = arg1, mtry = arg2)
  
  # Test du classifeur : classe predite
  rf_class <- predict(rf,attrition_ET, type="response")
  
  # Matrice de confusion
  Mc_rf <- table(attrition_ET$Attrition, rf_class)
  print(Mc_rf)
  
  # Calcul du taux de succes du classifieur
  succes2 <- (Mc_rf[1,1] + Mc_rf[2,2]) / nrow(attrition_ET)
  cat("succes2 = ",succes2)
  
  # Calcul de la mesure du Taux de Vrais Négatifs pour evaluer la fiabilité des prédictions négatives du classifieur:VN/(VN+FN)
  TVN_rf <- Mc_rf[1,1] / (Mc_rf[1,1] + Mc_rf[2,1])
  cat("\n TVN_rf = ",TVN_rf)
  
  # Test du classifeur : probabilites pour chaque prediction
  rf_prob <- predict(rf, attrition_ET, type="prob")
  
  # Courbe ROC
  rf_pred <- prediction(rf_prob[,2], attrition_ET$Attrition)
  rf_perf <- performance(rf_pred,"tpr","fpr")
  plot(rf_perf, main = "Random Forests randomForest()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  rf_auc <- performance(rf_pred, "auc")
  cat("\n AUC = ", as.character(attr(rf_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-------------------------------------------------#
# APPRENTISSAGE DES CONFIGURATIONS ALGORITHMIQUES #
#-------------------------------------------------#

# Forets d'arbres decisionnels aleatoires
test_rf(300, 3, FALSE, "red")
test_rf(300, 5, TRUE, "blue")
test_rf(500, 3, TRUE, "green")
test_rf(500, 5, TRUE, "orange")

#-------------------------#
# SUPPORT VECTOR MACHINES #
#-------------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_svm <- function(arg1, arg2, arg3){
  # Apprentissage du classifeur
  svm <- svm(Attrition~., attrition_EA, probability=TRUE, kernel = arg1, maxister=500)
  
  # Test du classifeur : classe predite
  svm_class <- predict(svm, attrition_ET, type="response")
  
  # Matrice de confusion
  Mc_svm <- table(attrition_ET$Attrition,svm_class)
  print(Mc_svm)
  
  # Calcul du taux de succes du classifieur
  succes3 <- (Mc_svm[1,1] + Mc_svm[2,2]) / nrow(attrition_ET)
  cat("\n succes3=",succes3)
  
  # Calcul de la mesure du Taux de Vrais Négatifs pour evaluer la fiabilité des prédictions négatives du classifieur:VN/(VN+FN)
  TVN_svm <- Mc_svm[1,1] / (Mc_svm[1,1] + Mc_svm[2,1])
  cat("\n TVN_svm = ",TVN_svm)
  
  
  # Test du classifeur : probabilites pour chaque prediction
  svm_prob <- predict(svm, attrition_ET, probability=TRUE)
  
  # Recuperation des probabilites associees aux predictions
  svm_prob <- attr(svm_prob, "probabilities")
  
  # Courbe ROC 
  svm_pred <- prediction(svm_prob[,1], attrition_ET$Attrition)
  
  svm_perf <- performance(svm_pred,"tpr","fpr")
  
  plot(svm_perf, main = "Support vector machines svm()", add = arg2, col = arg3)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  
  svm_auc <- performance(svm_pred, "auc")
  
  cat("\n AUC = ", as.character(attr(svm_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}


#-------------------------------------------------#
# APPRENTISSAGE DES CONFIGURATIONS ALGORITHMIQUES #
#-------------------------------------------------#

# Support vector machines
test_svm("linear", FALSE, "red")
test_svm("polynomial", TRUE, "blue")
test_svm("radial", TRUE, "green")
test_svm("sigmoid", TRUE, "orange")



#-----------------#
# NEURAL NETWORKS #
#-----------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_nnet <- function(arg1, arg2, arg3, arg4, arg5){
  # Redirection de l'affichage des messages intermédiaires vers un fichier texte
  sink('output.txt', append=T)
  
  # Apprentissage du classifeur 
  nn <- nnet(Attrition~., attrition_EA, size = arg1, decay = arg2, maxit=arg3,MaxNWts=84581)
  
  # Réautoriser l'affichage des messages intermédiaires
  sink(file = NULL)
  
  # Test du classifeur : classe predite
  nn_class <- predict(nn, attrition_ET, type="class")
  
  # Matrice de confusion
  
  Mc_nnet<-table(attrition_ET$Attrition,nn_class)
  print(Mc_nnet)
  
  # Calcul du taux de succes du classifieur
  succes4 <- (Mc_nnet[1,1] + Mc_nnet[2,2]) / nrow(attrition_ET)
  cat("succes4",succes4)
  
  # Calcul de la mesure du Taux de Vrais Négatifs pour evaluer la fiabilité des prédictions négatives du classifieur:VN/(VN+FN)
  TVN_nnet <- Mc_nnet[1,1] / (Mc_nnet[1,1] + Mc_nnet[2,1])
  cat("\n TVN_nnet=",TVN_nnet)
  
  
  # Test des classifeurs : probabilites pour chaque prediction
  nn_prob <- predict(nn, attrition_ET, type="raw")
  
  # Courbe ROC 
  nn_pred <- prediction(nn_prob[,1], attrition_ET$Attrition)
  nn_perf <- performance(nn_pred,"tpr","fpr")
  plot(nn_perf, main = "Réseaux de neurones nnet()", add = arg4, col = arg5)
  
  # Calcul de l'AUC
  nn_auc <- performance(nn_pred, "auc")
  cat("\n AUC = ", as.character(attr(nn_auc, "y.values")))
  
  # Return ans affichage sur la console
  invisible()
}

#-------------------------------------------------#
# APPRENTISSAGE DES CONFIGURATIONS ALGORITHMIQUES #
#-------------------------------------------------#

# Réseaux de neurones nnet()
test_nnet(50, 0.01, 100, FALSE, "red")
test_nnet(50, 0.01, 300, TRUE, "tomato")
#test_nnet(25, 0.01, 100, TRUE, "blue")
test_nnet(25, 0.01, 300, TRUE, "purple")
test_nnet(50, 0.001, 100, TRUE, "green")
test_nnet(50, 0.001, 300, TRUE, "turquoise")
test_nnet(25, 0.001, 100, TRUE, "grey")
test_nnet(25, 0.001, 300, TRUE, "black")


#-------------#
# NAIVE BAYES #
#-------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_nb <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur 
  nb <- naive_bayes(Attrition~., attrition_EA, laplace = arg1, usekernel = arg2)
  
  # Test du classifeur : classe predite
  nb_class <- predict(nb, attrition_ET, type="class")
  
  # Matrice de confusion
  Mc_nb <- table(attrition_ET$Attrition,nb_class)
  print(Mc_nb)
  print(Mc_nb[2,1])
  
  # Calcul du taux de succes du classifieur
  succes5 <- (Mc_nb[1,1] + Mc_nb[2,2]) / nrow(attrition_ET)
  cat("succes5 = ",succes5)
  
  # Calcul de la mesure du Taux de Vrais Négatifs pour evaluer la fiabilité des prédictions négatives du classifieur:VN/(VN+FN)
  TVN_nb <- Mc_nb[1,1] / (Mc_nb[1,1] + Mc_nb[2,1])
  cat("\n TVN_nb = ",TVN_nb)
  
  # Test du classifeur : probabilites pour chaque prediction
  nb_prob <- predict(nb, attrition_ET, type="prob")
  
  # Courbe ROC
  nb_pred <- prediction(nb_prob[,2], attrition_ET$Attrition)
  nb_perf <- performance(nb_pred,"tpr","fpr")
  plot(nb_perf, main = "Classifieurs bayésiens naïfs naiveBayes()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  nb_auc <- performance(nb_pred, "auc")
  cat("\n AUC = ", as.character(attr(nb_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-------------------------------------------------#
# APPRENTISSAGE DES CONFIGURATIONS ALGORITHMIQUES #
#-------------------------------------------------#

# Naive Bayes
test_nb(0, FALSE, FALSE, "red")
test_nb(200, FALSE, TRUE, "blue")
test_nb(0, TRUE, TRUE, "green")
test_nb(200, TRUE, TRUE, "orange")

#----------------------------------------------#
#DEFINITION DU CLASSIFIEUR RETENU : NAIVE BAYES#
#----------------------------------------------#

# Apprentissage du classifeur 
C_retenu <- naive_bayes(Attrition~., attrition_EA, laplace = 0, usekernel = FALSE)

#-------------------------------#
# PREDICTIONS par  Naive Bayes() #
#-------------------------------#

# Application de Naive Bayes aux prospects
attrition_new <- read.csv("Data_Projet_1_New.csv", header = TRUE, sep = ",", dec = ".",stringsAsFactors = T)
str(attrition_new)

# Application du classifieur a attrition_pro : classe predite
C_retenu_pred <- predict(C_retenu, attrition_new, type="class")
C_retenu_pred
table(C_retenu_pred)

# Application du classifieur a attrition_pro : Probabilités
C_retenu_prob <- predict(C_retenu, attrition_new, type="prob")
C_retenu_prob

# Ajout dans le data frame attrition_pro d'une colonne Attrition contenant la classe predite 
attrition_new$Attrition <- C_retenu_pred

# Ajout dans le data frame attrition_pro d'une colonne Probabilities contenant les  
# probabilités de la classe predite 
attrition_new$Probabilities <- C_retenu_prob

summary(attrition_new)


# Ecriture du fichier avec Attrition et Probabilities
write.table(attrition_new, "Data_Attrition_New.csv", sep=",", dec=".", row.names=F) 

data<-read.csv(file = "Data_Attrition_New.csv",header = TRUE,sep = ",",dec = ".")

View(data)
