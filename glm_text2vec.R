#===================== import data ====================================================

library(readxl)
df <- read_excel("D:/NLP-ticketing/df2018.xlsx")

summary(df)
str(df)
df$time <- format(as.POSIXct(df$TIME) ,format = "%H:%M:%S") #split date to time/drop date

df$SEX<-as.factor(df$SEX)
df$`RESPONDENT 1`<-as.factor(df$`RESPONDENT 1`)
df$`RESPONDENT 2`<-as.factor(df$`RESPONDENT 2`)
df$CHANNEL<-as.factor(df$CHANNEL)
df$CITY<-as.factor(df$CITY)
df$CARD<-as.factor(df$SSB_CARD)
df$CATEGORY<-as.factor(df$CATEGORY)
df$SUBCATEGORY<-as.factor(df$SUBCATEGORY)

summary(df)



#===================== subset data - "product range" =====================================


df_gama<-df[which(df$CATEGORY=='product range'),]
summary(df_gama)
summary(df_gama$SUBCATEGORY)
df_gama<-df_gama[which(df_gama$SUBCATEGORY!="range nonexistent"),]
df_gama$SUBCATEGORY<-droplevels(df_gama$SUBCATEGORY)

library(text2vec)
library(data.table)
typeof(df_gama)

#====================== tokenize ========================================================

df_gama<-setDT(df_gama)
setkey(df_gama, ID_call)

ALL_IDS<-df_gama$ID_call
train_ids = sample(ALL_IDS, 35)
test_ids = setdiff(ALL_IDS, train_ids)
train = df_gama[J(train_ids)]
test = df_gama[J(test_ids)]

prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(train$CONTENT, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train$ID_call, 
                  progressbar = FALSE)
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)

#======================= document to term matrix =============================

dtm_train = create_dtm(it_train, vectorizer)
dim(dtm_train)
levels(df_gama$SUBCATEGORY)



#============== glm model =====================================================

NFOLDS<-5

set.seed(23)
glmnet_classifier<-cv.glmnet(x = dtm_train, y = train$SUBCATEGORY, 
                             family = 'multinomial', 
                             nfolds = NFOLDS, 
                             alpha = 1,
                             type.measure = 'class')
print(paste("max misclassification error =", round(max(glmnet_classifier$cvm), 4)))

it_test = test$CONTENT %>% 
  prep_fun %>% 
  tok_fun %>% 
  itoken(ids = test$ID_call, 
         progressbar = FALSE)

dtm_test = create_dtm(it_test, vectorizer)

preds = predict.cv.glmnet(glmnet_classifier, dtm_test, type = 'response')
preds2 <- as.data.frame(preds)
colnames(preds2)
levels(df_card$SUBCATEGORY)
colnames(preds2)


preds2["prediction"]<-colnames(preds2)[max.col(preds2,ties.method="first")]

test<-as.data.frame(test)
test["pred"]<-preds2$prediction

#write.csv(test, "test2.csv")

test["check"]<-vector()
test$check<-ifelse(test$SUBCATEGORY==test$pred, 1,0)
test$check<-as.numeric(test$check)
count(test[which(test$check==1),])
count(test[which(test$check==0),])
count(test[which(test$check==1),])/nrow(test)

cat(sprintf("prediction of SUBCATEGORY 4 cat.: %f...\n", count(test[which(test$check==1),])/nrow(test)))





