##### Pearson Correlation Analysis ####
# Load package
# install.packages("psych")
library(readxl)
library(corrplot)
library(psych)

# Data loading
bc1 <- read_excel(file.choose())
bc2 <- read_excel(file.choose())
mtcars1 <- data.frame(bc1)
mtcars2 <- data.frame(bc2)

# Calculate the correlation coefficient and p-value
M1 <- cor(mtcars1)
M2 <- cor(mtcars2)
testRes1 <- cor.mtest(mtcars1, conf.level = 0.95)
testRes2 <- cor.mtest(mtcars2, conf.level = 0.95)
head(testRes1)
head(testRes2)

nature_col <- colorRampPalette(c("#2166AC", "#67A9CF", "#D1E5F0", 
                                 "#FDDBC7", "#EF8A62", "#B2182B"))(200)
par(family = "serif")

# Paint
cor.plot1 <- corrplot(corr = M1, p.mat = testRes1$p, type = "lower",
                      tl.col = "black",  
                      tl.cex= 1, tl.srt =360,
                      insig = "label_sig", sig.level = c(.01, .05),
                      pch.cex =1, pch.col = "black")
cor.plot2 <- corrplot(corr = M2, p.mat = testRes2$p, type = "lower",
                      tl.col = "black",tl.cex= 1,tl.srt =360,
                      insig = "label_sig", sig.level = c(.01, .05),
                      pch.cex = 1, pch.col = "black")
# Merge
par(family= "serif") 
par("family") 
cor.plot3 <- corrplot(corr = M1, p.mat = testRes1$p, type = "lower",
                      tl.pos = "lt", tl.col = "black",tl.cex= 1.05,
                      insig = "label_sig", sig.level = c(.01, .05),
                      pch.cex =1, pch.col = "black")
cor.plot4 <- corrplot(corr = M2, p.mat = testRes2$p, type = "upper", 
                      tl.col = "black",tl.cex= 1,tl.pos = "n",
                      add = T,#cl.pos = "n",
                      insig = "label_sig", sig.level = c(.01, .05),
                      pch.cex = 1, pch.col = "black",tl.srt =90)

#####Logistic Regression####
# Load package
library(readxl)
library(MASS)
# Import data
train <- read_excel(file.choose())
colnames(train)
form_cls <- as.formula(
  paste0(
    "Y ~ ",
    paste(colnames(train)[1:18],collapse = "+") #[1:18]为选择自变量的列
  ))
form_cls  
# Model Building 
set.seed(1234)
heart_model <- glm(form_cls,data=train, family=binomial("logit"))
summary(heart_model) 
logit.step <- step(heart_model,direction="both")  
summary(logit.step)  


#####Random Forest Variable Importance Measures####
# Load package
library(randomForest) 
library(datasets) 
library(tidyverse)
library(rfPermute) 
library(A3) 
library(readxl)
library(writexl)

# 2001-2008年
train1 <-read_excel(file.choose())
colnames(train1)
form_cls1 <- as.formula(paste0(
  "Y ~",
  paste(colnames(train1)[1:18],collapse = "+")))
form_cls1
# Model Building
set.seed(123) 
rf1 <- randomForest(form_cls1, 
                    data = train1, 
                    ntree = 200, mtry = 4,
                    importance = TRUE, 
                    keep.forest = TRUE)
print(rf1)

set.seed(123) 
imp1 <- importance(rf1,scale = TRUE)
imp1

imp.df1 <- data.frame(imp1, check.names = FALSE) 
imp.df1$Y <- row.names(imp.df1) 
imp.df1

p1 <- imp.df1 %>% 
  ggplot(aes(x = reorder(Y, `%IncMSE`), y = `%IncMSE`)) + 
  geom_bar(stat = "identity", aes(fill = `%IncMSE`)) + 
  scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 20)) + 
  geom_text(aes(label = paste(round(`%IncMSE`, 2), "%"), hjust = -0.3)) +
  scale_fill_gradient(
    low = "#0072B5", 
    high = "#BC3C29",
    name = "Importance (%)"
  ) + 
  labs(x = "", y = "Increase in MSE (%)") + 
  coord_flip() + 
  theme_bw() + 
  theme(
    legend.position = c(0.85, 0.2),  
    legend.direction = "vertical", 
    legend.key.height = unit(1, "cm") 
  )
print(p1)

ggsave("4RFM_legend.tiff", 
       plot = p1, 
       device = "tiff",
       dpi = 300, 
       width = 10, 
       height = 6, 
       units = "in")

# 2009-2020
train2 <-read_excel(file.choose())

colnames(train2)
form_cls2 <- as.formula(paste0(
  "Y ~",
  paste(colnames(train2)[1:18],collapse = "+")))
form_cls2

set.seed(123) 
rf2 <- randomForest(form_cls2, 
                    data = train2, 
                    ntree = 200, mtry = 4,
                    importance = TRUE, 
                    keep.forest = TRUE)
set.seed(123) 
imp2 <- importance(rf2,scale = TRUE)
imp2

imp.df2 <- data.frame(imp2, check.names = FALSE) 
imp.df2$Y <- row.names(imp.df2) 
imp.df2

p2 <- imp.df2 %>% 
  ggplot(aes(x=reorder(Y,`%IncMSE`),y=`%IncMSE`))+ 
  geom_bar(stat = "identity",aes(fill=`%IncMSE`))+ 
  scale_y_continuous(limits = c(0,100),breaks = seq(0,100, by=20))+ 
  geom_text(aes(label = paste(round(`%IncMSE`,2),"%"),hjust = -0.3))+
  scale_fill_gradient(low = "#0072B5",high = "#BC3C29")+ 
  labs(x="",y="Increase in MSE(%)")+ 
  coord_flip()+ 
  theme_bw()+ 
  theme(legend.position = "none" )
p2
