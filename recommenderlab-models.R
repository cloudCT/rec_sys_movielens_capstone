


######################## Recommenderlab #################





# Libraries needed:
library(recommenderlab)
library(reshape2)




############
######## Creating Matrix

# Creating realRatingMatrix and creating subset of data with only 
# relevant data

edx.copy <- edx

edx.copy$userId <- as.factor(edx.copy$userId)
edx.copy$movieId <- as.factor(edx.copy$movieId)



sparse_dat <- sparseMatrix(i = edx.copy$userId,
                           j = edx.copy$movieId,
                           x = edx.copy$rating,
                           dims = c(length(unique(edx.copy$userId)),
                                    length(unique(edx.copy$movieId))),
                           dimnames= list(paste("u", 
                                                1:length(unique(edx.copy$userId)),
                                                sep = ""),
                                          paste("m",
                                                1:length(unique(edx.copy$movieId)), 
                                                sep = "")))

rm(edx.copy)

sparse_dat[1:10,1:10]

r_mat <- as(sparse_dat, "realRatingMatrix")


# Selecting the most relevant users and movies:

# Determining the minimum number of movies per user:
min_movies <- quantile(rowCounts(r_mat),0.9)
min_movies

min_users <- quantile(colCounts(r_mat),0.9)

# Only having upper bound of users and movies

rel_r_mat <- r_mat[rowCounts(r_mat) > min_movies,
                   colCounts(r_mat) > min_users]






############
######## Models stacked against each other

set.seed(1)


models_train_scheme <- rel_r_mat%>%
  evaluationScheme(method = 'cross-validation',
                   given = -5, 
                   k = 10)


models_to_try <- list(
  `IBCF Cosinus` = list(name = "IBCF",
                        param = list(method = "cosine")),
  `IBCF Pearson` = list(name = "IBCF",
                        param = list(method = "pearson")),
  `UBCF Cosinus` = list(name = "UBCF",
                        param = list(method = "cosine")),
  `UBCF Pearson` = list(name = "UBCF",
                        param = list(method = "pearson"))
)



results <- evaluate(models_train_scheme, method = models_to_try,
                    type = "ratings")


plot(results)





############
######## UBCF Model Cosine




set.seed(1)
model_train_scheme <- rel_r_mat %>%
  evaluationScheme(method = "split", # single train/test split
                   train = 0.75, # proportion of rows to train.
                   given = -5) 

# Model



ubcf_model <- getData(model_train_scheme, "train") %>% #only fit on the 75% training data.
  Recommender(method = "UBCF", param = list(method = "Cosine",
                                            nn = 150,
                                            normalize = "center"))



# Prediction

ubcf_pred <- predict(ubcf_model, getData(model_train_scheme, "known"), type = "ratings")
ubcf_pred




test_error_ubcf_c <- calcPredictionAccuracy(ubcf_pred, getData(model_train_scheme, "unknown"))[1]

recom_results <- data_frame(method = "UBCF Cosine", RMSE = test_error_ubcf_c)




############
######## UBCF Model Pearson


set.seed(1)
model_train_scheme <- rel_r_mat %>%
  evaluationScheme(method = "split",
                   train = 0.75, 
                   given = -5) 
# Model


ubcf_model <- getData(model_train_scheme, "train") %>%
  Recommender(method = "UBCF", param = list(method = "Pearson",
                                            nn = 200,
                                            normalize = "center"))



# Prediction

ubcf_pred <- predict(ubcf_model, getData(model_train_scheme, "known"), type = "ratings")
ubcf_pred




test_error_ubcf_p <- calcPredictionAccuracy(ubcf_pred, getData(model_train_scheme, "unknown"))[1]

recom_results <- bind_rows(recom_results,
                           data_frame(method = "UBCF Pearson",
                                      RMSE = test_error_ubcf_p))




############
######## IBCF Model Cosine


set.seed(1)

model_train_scheme <- rel_r_mat %>%
  evaluationScheme(method = 'split', # single train/test split
                   train = 0.75, 
                   given = -5,
                   k = 1)


# Model

model_params <- list(method = "Cosine",
                     k = 275, # Based on cross-validation
                     normalize = "center")

ibcf_model <- getData(model_train_scheme, "train") %>% #only fit on the 75% training data.
  Recommender(method = "IBCF", param = model_params)



# Prediction

ibcf_pred <- predict(ibcf_model, getData(model_train_scheme, "known"), type = "ratings")
ibcf_pred




test_error_ibcf_c <- calcPredictionAccuracy(ibcf_pred, getData(model_train_scheme, "unknown"))[1]

recom_results <- bind_rows(recom_results,
                           data_frame(method = "IBCF Cosine", 
                                      RMSE = test_error_ibcf_c))



############
######## IBCF Model Pearson


set.seed(1)

model_train_scheme <- rel_r_mat %>%
  evaluationScheme(method = 'split', # single train/test split
                   train = 0.75, 
                   given = -5,
                   k = 1)


# Model

model_params <- list(method = "Pearson",
                     k = 50, # Based on cross-validation
                     normalize = "center")

ibcf_model <- getData(model_train_scheme, "train") %>% #only fit on the 75% training data.
  Recommender(method = "IBCF", param = model_params)



# Prediction

ibcf_pred <- predict(ibcf_model, getData(model_train_scheme, "known"), type = "ratings")
ibcf_pred




test_error_ibcf_p <- calcPredictionAccuracy(ibcf_pred, getData(model_train_scheme, "unknown"))[1]

recom_results <- bind_rows(recom_results,
                           data_frame(method = "IBCF Pearson", 
                                      RMSE = test_error_ibcf_p))





############
######## Ensemble / Hybrid Recommender



set.seed(1)

model_train_scheme <- rel_r_mat %>%
  evaluationScheme(method = 'split', # single train/test split
                   train = 0.75, 
                   given = -5,
                   k = 1)


# Model

recommenders <- list(
  UBCF = list(name = "UBCF", param = list(method = "Cosine",
                                       nn = 150,
                                       normalize = "center")),
  IBCF = list(name = "IBCF", param = list(method = "Pearson",
                                          k = 50, 
                                          normalize = "center"))
)



hyb_recom <- getData(model_train_scheme, "train") %>% 
  Recommender(method = "HYBRID", parameter = list(recommenders = recommenders, 
                                                  weights = c(0.5,0.5)))



# Prediction

hyb_pred <- predict(hyb_recom, getData(model_train_scheme, "known"), type = "ratings")
hyb_pred




test_error_hyb <- calcPredictionAccuracy(hyb_pred, getData(model_train_scheme, "unknown"))[1]

recom_results <- bind_rows(recom_results,
                           data_frame(method = "HYBRID",
                                      RMSE = test_error_hyb))



############
######## Visualization of results
library(kableExtra)

kable(recom_results)

recom_results %>% 
  kbl() %>% 
  kable_material_dark(full_width = F) %>% 
  column_spec(2, color = "white", 
              background = spec_color(recom_results$RMSE[1:5], end = 0.7,
                                      direction = -1))



# Higher is worse
recom_results %>% 
  ggplot(aes(method,RMSE, fill = RMSE))+
  geom_bar(stat = "identity")+
  theme(axis.text.x = element_text(angle=70,hjust = 1,size = 5))+
  coord_cartesian(ylim = c(0.5,1.25))+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  xlab("Method")+
  ggtitle("Recommenderlab Models")+
  scale_fill_gradient2(low="coral",high = "azure4", mid = "azure3",
                       midpoint = 0.8)


# Performance based
recom_results %>% 
  mutate(RMSE = 1.5- RMSE) %>% 
  arrange(desc(RMSE)) %>% 
  ggplot(aes(RMSE,reorder(method,RMSE), fill = RMSE))+
  geom_bar(stat = "identity")+
  theme(axis.title.y = element_blank())+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  coord_cartesian(xlim = c(0.5,0.8))+
  scale_x_continuous(breaks= c(0.5,0.6,0.7,0.8), labels= c(1,0.9,0.8,0.7))+
  scale_fill_gradient2(low="azure4",high = "coral", mid = "azure3",
                       midpoint = 0.7 ,breaks= c(0.71,0.7,0.69), 
                       labels= c(0.79,0.8,0.81))





################
########## UBCF Model with different dataset sizes

# We will be using different cutoffs to demonstrate the effect on the RMSE
# Using more of the data results in increasingly higher RMSE and model run time



edx.copy <- edx

edx.copy$userId <- as.factor(edx.copy$userId)
edx.copy$movieId <- as.factor(edx.copy$movieId)



sparse_dat <- sparseMatrix(i = edx.copy$userId,
                           j = edx.copy$movieId,
                           x = edx.copy$rating,
                           dims = c(length(unique(edx.copy$userId)),
                                    length(unique(edx.copy$movieId))),
                           dimnames= list(paste("u", 
                                                1:length(unique(edx.copy$userId)),
                                                sep = ""),
                                          paste("m",
                                                1:length(unique(edx.copy$movieId)), 
                                                sep = "")))

rm(edx.copy)

sparse_dat[1:10,1:10]

r_mat <- as(sparse_dat, "realRatingMatrix")


############### 90% quantile cutoff

# Determining the minimum number of movies per user:
min_movies <- quantile(rowCounts(r_mat),0.9)
min_movies

min_users <- quantile(colCounts(r_mat),0.9)

# Only having upper bound of users and movies

rel_r_mat_90 <- r_mat[rowCounts(r_mat) > min_movies,
                   colCounts(r_mat) > min_users]


############### 85% quantile cutoff

# Determining the minimum number of movies per user:
min_movies <- quantile(rowCounts(r_mat),0.85)
min_movies

min_users <- quantile(colCounts(r_mat),0.85)

# Only having upper bound of users and movies

rel_r_mat_85 <- r_mat[rowCounts(r_mat) > min_movies,
                      colCounts(r_mat) > min_users]


############### 80% quantile cutoff

# Determining the minimum number of movies per user:
min_movies <- quantile(rowCounts(r_mat),0.8)
min_movies

min_users <- quantile(colCounts(r_mat),0.8)

# Only having upper bound of users and movies

rel_r_mat_80 <- r_mat[rowCounts(r_mat) > min_movies,
                      colCounts(r_mat) > min_users]



############### 75% quantile cutoff

# Determining the minimum number of movies per user:
min_movies <- quantile(rowCounts(r_mat),0.75)
min_movies

min_users <- quantile(colCounts(r_mat),0.75)

# Only having upper bound of users and movies

rel_r_mat_75 <- r_mat[rowCounts(r_mat) > min_movies,
                      colCounts(r_mat) > min_users]


############### 70% quantile cutoff

# Determining the minimum number of movies per user:
min_movies <- quantile(rowCounts(r_mat),0.7)
min_movies

min_users <- quantile(colCounts(r_mat),0.7)

# Only having upper bound of users and movies

rel_r_mat_70 <- r_mat[rowCounts(r_mat) > min_movies,
                      colCounts(r_mat) > min_users]



###########
############ Models

set.seed(1)
model_train_scheme <- rel_r_mat_90 %>%
  evaluationScheme(method = "split",
                   train = 0.75,
                   given = -5) 

ubcf_model <- getData(model_train_scheme, "train") %>% 
  Recommender(method = "UBCF", param = list(method = "Cosine",
                                            nn = 50,
                                            normalize = "center"))

pred_90 <- predict(ubcf_model, getData(model_train_scheme, "known"), type = "ratings")


rmse_90 <- calcPredictionAccuracy(pred_90, getData(model_train_scheme, "unknown"))[1]

cutoff_comp <- data_frame(cutoff = "90%", RMSE = rmse_90)


model_train_scheme <- rel_r_mat_85 %>%
  evaluationScheme(method = "split",
                   train = 0.75,
                   given = -5) 

ubcf_model <- getData(model_train_scheme, "train") %>% 
  Recommender(method = "UBCF", param = list(method = "Cosine",
                                            nn = 50,
                                            normalize = "center"))

pred_85 <- predict(ubcf_model, getData(model_train_scheme, "known"), type = "ratings")


rmse_85 <- calcPredictionAccuracy(pred_85, getData(model_train_scheme, "unknown"))[1]

cutoff_comp <- bind_rows(cutoff_comp,
                         data_frame(cutoff = "85%", 
                                    RMSE = rmse_85))


model_train_scheme <- rel_r_mat_80 %>%
  evaluationScheme(method = "split",
                   train = 0.75,
                   given = -5) 

ubcf_model <- getData(model_train_scheme, "train") %>% 
  Recommender(method = "UBCF", param = list(method = "Cosine",
                                            nn = 50,
                                            normalize = "center"))

pred_80 <- predict(ubcf_model, getData(model_train_scheme, "known"), type = "ratings")


rmse_80 <- calcPredictionAccuracy(pred_80, getData(model_train_scheme, "unknown"))[1]

cutoff_comp <- bind_rows(cutoff_comp,
                         data_frame(cutoff = "80%", 
                                    RMSE = rmse_80))

model_train_scheme <- rel_r_mat_75 %>%
  evaluationScheme(method = "split",
                   train = 0.75,
                   given = -5) 

ubcf_model <- getData(model_train_scheme, "train") %>% 
  Recommender(method = "UBCF", param = list(method = "Cosine",
                                            nn = 50,
                                            normalize = "center"))

pred_75 <- predict(ubcf_model, getData(model_train_scheme, "known"), type = "ratings")


rmse_75 <- calcPredictionAccuracy(pred_75, getData(model_train_scheme, "unknown"))[1]

cutoff_comp <- bind_rows(cutoff_comp,
                         data_frame(cutoff = "75%", 
                                    RMSE = rmse_75))


model_train_scheme <- rel_r_mat_70 %>%
  evaluationScheme(method = "split",
                   train = 0.75,
                   given = -5) 

ubcf_model <- getData(model_train_scheme, "train") %>% 
  Recommender(method = "UBCF", param = list(method = "Cosine",
                                            nn = 50,
                                            normalize = "center"))

pred_70 <- predict(ubcf_model, getData(model_train_scheme, "known"), type = "ratings")


rmse_70 <- calcPredictionAccuracy(pred_70, getData(model_train_scheme, "unknown"))[1]

cutoff_comp <- bind_rows(cutoff_comp,
                         data_frame(cutoff = "70%", 
                                    RMSE = rmse_70))



### NO CUTOFF
# Do not run this code. It takes about 8 hours and achieves a horrendous RMSE
# RMSE : 


# 
# set.seed(1)
# model_train_scheme <- r_mat %>%
#   evaluationScheme(method = "split",
#                    train = 0.75,
#                    given = -5) 
# 
# ubcf_model <- getData(model_train_scheme, "train") %>% 
#   Recommender(method = "UBCF", param = list(method = "Cosine",
#                                             nn = 50,
#                                             normalize = "center"))
# 
# pred <- predict(ubcf_model, getData(model_train_scheme, "known"), type = "ratings")
# 
# 
# rmse <- calcPredictionAccuracy(pred, getData(model_train_scheme, "unknown"))[1]



cutoff_comp <-bind_rows(cutoff_comp,
                        data_frame(cutoff = "Complete Dataset", 
                                   RMSE = 1.225969))




############ Results Cutoff Comparison

library(knitr)
library(kableExtra)

kable(cutoff_comp)

cutoff_comp %>% 
  kbl() %>% 
  kable_material_dark(full_width = F) %>% 
  column_spec(2, color = "white", 
              background = spec_color(cutoff_comp$RMSE[1:6], end = 0.7,
                                      direction = -1,option="E"))




cutoff_comp %>% 
  mutate(RMSE = 1.5- RMSE) %>% 
  arrange(desc(RMSE)) %>% 
  ggplot(aes(RMSE,reorder(cutoff,RMSE), fill = RMSE))+
  geom_bar(stat = "identity")+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  ylab("Cutoff")+
  coord_cartesian(xlim = c(0.2,0.8))+
  scale_x_continuous(breaks= c(0.2,0.5,0.6,0.7,0.8), labels= c(1.3,1,0.9,0.8,0.7))+
  scale_fill_gradient2(low="azure4",high = "coral", mid = "azure3",
                       midpoint = 0.65 ,breaks= c(0.3,0.6,0.7), 
                       labels= c(1.2,0.9,0.80))









