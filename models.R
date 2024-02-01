

## To make computation faster:
library(doParallel)
# detectCores()
registerDoParallel(cores = 8)



################ Base-Models / Models to beat



### Loss Function (RMSE)

RMSE <- function(actual_ratings, predicted_ratings){
  sqrt(mean((actual_ratings - predicted_ratings)^2))
}



### Naive Models (predicting mu and just 3)

mu <- edx_train %>% summarize(mu = mean(rating)) %>% pull(mu)
mu

naive_rmse <- RMSE(edx_test$rating, mu)
naive_rmse

rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)



naiver_rmse <- RMSE(edx_test$rating,3)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Guessing 3",
                                     RMSE = naiver_rmse ))


### Random guessing

range <- seq(0.5,5,0.5)
guess_right <- function(x,y){
  mean(y==x)
}

set.seed(1)

simulation <- replicate(10000, {
  i <- sample(edx_train$rating, 1000, replace = T)
  sapply(range,guess_right,i)
})

guess_prob <- c()
for(i in 1:nrow(simulation)){
  guess_prob <- append(guess_prob,mean(simulation[i,]))
}

random_preds <- sample(range,
                       size = nrow(edx_test),
                       replace = T,
                       prob = guess_prob)

guessing_rmse <- RMSE(random_preds,edx_test$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Random guessing",
                                     RMSE = guessing_rmse ))




### Movie + User effects base model

mu <- mean(edx_train$rating) 
movie_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))
user_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_2_rmse <- RMSE(predicted_ratings, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))

rmse_results


### Regularization

# Lambda cross validation (Just movie effect)

lambdas <- seq(0, 10, 0.25)
mu <- mean(edx_train_l$rating)
just_the_sum <- edx_train_l %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- edx_test_l %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
})

library(gghighlight)
ggplot(data =data.frame(rmses=rmses,lambdas =lambdas),
       aes(x = lambdas, y = rmses))+
  geom_point(col ="coral")+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  xlab("Lambdas")+
  ylab("Rmse")+
  gghighlight(rmses == min(rmses))

  

lambdas[which.min(rmses)]






# Regularized movie effects

lambda <- lambdas[which.min(rmses)]
movie_reg_avgs <- edx_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

predicted_ratings <- edx_test %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

model_3_rmse <- RMSE(predicted_ratings, edx_test$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_3_rmse ))

## Summary first base models

library(knitr)
kable(rmse_results)



















#### Regularized bi and bu


regularization <- function(lambda,edx_train_l,edx_test_l){
  mu <- mean(edx_train_l$rating)
  b_i <- edx_train_l %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx_train_l %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  predicted_ratings <- 
    edx_test_l %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
}


lambdas <- seq(0,10,0.25)

lambdas_rmse <- sapply(lambdas,
                       regularization,
                       edx_train_l = edx_train_l,
                       edx_test_l = edx_test_l)

lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)

lambdas_tibble


lambda <- lambdas[which.min(lambdas_rmse)]


mu <- mean(edx_train$rating)

b_i_reg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))


preds <- edx_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  mutate(prediction = mu + b_i +b_u) %>% 
  pull(prediction)

reg_model_rmse <- RMSE(preds, edx_test$rating)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect + User Effect Model",  
                                     RMSE = reg_model_rmse ))



##### Regularized with genre effect


regularization <- function(lambda,edx_train_l,edx_test_l){
  mu <- mean(edx_train_l$rating)
  b_i <- edx_train_l %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx_train_l %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <-edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    mutate(genres =str_split(genres, "\\|")) %>% 
    unnest(cols = c(genres)) %>% 
    group_by(genres) %>% 
    mutate(b_g = sum(rating- mu - b_i - b_u)/(n()+lambda)) %>% 
    ungroup() %>% 
    group_by(movieId) %>% 
    summarize(b_g = mean(b_g))
  predicted_ratings <- 
    edx_test_l %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "movieId") %>% 
    filter(!is.na(b_i), !is.na(b_u),!is.na(b_g)) %>% 
    mutate(pred = mu + b_i + b_u +b_g) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
}


lambdas <- seq(0,10,0.25)

lambdas_rmse <- sapply(lambdas,
                       regularization,
                       edx_train_l = edx_train_l,
                       edx_test_l = edx_test_l)

lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)

lambdas_tibble



lambda <- lambdas[which.min(lambdas_rmse)]


mu <- mean(edx_train$rating)

b_i_reg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_g_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  mutate(b_g = sum(rating- mu - b_i - b_u)/(n()+lambda)) %>% 
  ungroup() %>% 
  group_by(movieId) %>% 
  summarize(b_g = mean(b_g))

preds <- edx_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_g_reg, by = "movieId") %>% 
  mutate(prediction = mu + b_i +b_u+b_g) %>% 
  pull(prediction)

reg_model_2_rmse <- RMSE(preds, edx_test$rating)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized with split Genre Effects Model",  
                                     RMSE = reg_model_2_rmse ))



##### Regularized with joint genres

regularization <- function(lambda,edx_train_l,edx_test_l){
  mu <- mean(edx_train_l$rating)
  b_i <- edx_train_l %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx_train_l %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <-edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))
  predicted_ratings <- 
    edx_test_l %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>% 
  filter(!is.na(b_i), !is.na(b_u),!is.na(b_g)) %>% 
    mutate(pred = mu + b_i + b_u +b_g) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
}


lambdas <- seq(0,10,0.25)

lambdas_rmse <- sapply(lambdas,
                       regularization,
                       edx_train_l = edx_train_l,
                       edx_test_l = edx_test_l)

lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)

lambdas_tibble



lambda <- lambdas[which.min(lambdas_rmse)]


mu <- mean(edx_train$rating)

b_i_reg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_g_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))

preds <- edx_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_g_reg, by = "genres") %>% 
mutate(prediction = mu + b_i +b_u+b_g) %>% 
  pull(prediction)

reg_model_3_rmse <- RMSE(preds, edx_test$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized with joint Genre Effect Model",  
                                     RMSE = reg_model_3_rmse ))




#### Regularized with time between movie release and rating

regularization <- function(lambda,edx_train_l,edx_test_l){
  mu <- mean(edx_train_l$rating)
  b_i <- edx_train_l %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx_train_l %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <-edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))
  b_t <- edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>% 
    mutate(year = str_extract(title, "(\\(\\d{4}\\))"),
           title = str_replace(title,"(\\(\\d{4}\\))$","")) %>%
    mutate(year = as.numeric(str_extract(year,"\\d{4}"))) %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = year(timestamp)) %>% 
    mutate(years_between = timestamp - year) %>% 
    group_by(years_between) %>% 
    summarize(b_t = sum(rating -mu-b_i-b_u-b_g)/(n()+lambda))

  predicted_ratings <- 
    edx_test_l %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(year = str_extract(title, "(\\(\\d{4}\\))"),
           title = str_replace(title,"(\\(\\d{4}\\))$","")) %>%
    mutate(year = as.numeric(str_extract(year,"\\d{4}"))) %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = year(timestamp)) %>% 
    mutate(years_between = timestamp - year) %>% 
    left_join(b_t, by ="years_between") %>% 
  filter(!is.na(b_i), !is.na(b_u),!is.na(b_u),!is.na(b_t)) %>% 
    mutate(pred = mu + b_i + b_u +b_g+b_t) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
}


lambdas <- seq(0,10,0.25)

lambdas_rmse <- sapply(lambdas,
                       regularization,
                       edx_train_l = edx_train_l,
                       edx_test_l = edx_test_l)

lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)

lambdas_tibble



lambda <- lambdas[which.min(lambdas_rmse)]


mu <- mean(edx_train$rating)

b_i_reg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_g_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating -b_i-mu)/(n()+lambda))

b_t_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>%
  left_join(b_g_reg, by = "genres") %>%
  mutate(year = str_extract(title, "(\\(\\d{4}\\))"),
         title = str_replace(title,"(\\(\\d{4}\\))$","")) %>%
  mutate(year = as.numeric(str_extract(year,"\\d{4}"))) %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  mutate(years_between = timestamp - year) %>% 
  group_by(years_between) %>% 
  summarize(b_t = sum(rating -mu-b_i-b_u-b_g)/(n()+lambda))


preds <- edx_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_g_reg, by = "genres") %>% 
  mutate(year = str_extract(title, "(\\(\\d{4}\\))"),
         title = str_replace(title,"(\\(\\d{4}\\))$","")) %>%
  mutate(year = as.numeric(str_extract(year,"\\d{4}"))) %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  mutate(years_between = timestamp - year) %>% 
  left_join(b_t_reg, by ="years_between") %>% 
  mutate(prediction = mu + b_i +b_u+b_g+b_t) %>% 
  pull(prediction)

reg_model_4_rmse <- RMSE(preds, edx_test$rating)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized with \"years between\" Effect Model",  
                                     RMSE = reg_model_4_rmse ))




##### Regularized with hours


regularization <- function(lambda,edx_train_l,edx_test_l){
  mu <- mean(edx_train_l$rating)
  b_i <- edx_train_l %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx_train_l %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <-edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))
  b_h <- edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = hour(timestamp)) %>% 
    group_by(timestamp) %>% 
    summarize(b_h = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))
    
  predicted_ratings <- 
    edx_test_l %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = hour(timestamp)) %>% 
    left_join(b_h, by = "timestamp") %>% 
    filter(!is.na(b_i), !is.na(b_u),!is.na(b_g),!is.na(b_h)) %>% 
    mutate(pred = mu + b_i + b_u +b_g+b_h) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
}


lambdas <- seq(0,10,0.25)

lambdas_rmse <- sapply(lambdas,
                       regularization,
                       edx_train_l = edx_train_l,
                       edx_test_l = edx_test_l)

lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)

lambdas_tibble



lambda <- lambdas[which.min(lambdas_rmse)]


mu <- mean(edx_train$rating)

b_i_reg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_g_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))

b_h_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  left_join(b_g_reg, by ="genres") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = hour(timestamp)) %>% 
  group_by(timestamp) %>% 
  summarize(b_h = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))


preds <- edx_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_g_reg, by = "genres") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = hour(timestamp)) %>% 
  left_join(b_h_reg, by = "timestamp") %>% 
  mutate(prediction = mu + b_i +b_u+b_g+b_h) %>% 
  pull(prediction)

reg_model_5_rmse <- RMSE(preds, edx_test$rating)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized with Hour Effect Model",  
                                     RMSE = reg_model_5_rmse ))



##### Regularized with Year Effect



regularization <- function(lambda,edx_train_l,edx_test_l){
  mu <- mean(edx_train_l$rating)
  b_i <- edx_train_l %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx_train_l %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <-edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))
  b_y <- edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = year(timestamp)) %>% 
    group_by(timestamp) %>% 
    summarize(b_y = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))
  
  predicted_ratings <- 
    edx_test_l %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = year(timestamp)) %>% 
    left_join(b_y, by = "timestamp") %>% 
    filter(!is.na(b_i), !is.na(b_u),!is.na(b_g),!is.na(b_y)) %>% 
    mutate(pred = mu + b_i + b_u +b_g+b_y) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
}


lambdas <- seq(0,10,0.25)

lambdas_rmse <- sapply(lambdas,
                       regularization,
                       edx_train_l = edx_train_l,
                       edx_test_l = edx_test_l)

lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)

lambdas_tibble



lambda <- lambdas[which.min(lambdas_rmse)]


mu <- mean(edx_train$rating)

b_i_reg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_g_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))

b_y_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  left_join(b_g_reg, by ="genres") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  group_by(timestamp) %>% 
  summarize(b_y = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))


preds <- edx_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_g_reg, by = "genres") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  left_join(b_y_reg, by = "timestamp") %>% 
  mutate(prediction = mu + b_i +b_u+b_g+b_y) %>% 
  pull(prediction)

reg_model_6_rmse <- RMSE(preds, edx_test$rating)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized with Year Effect Model",  
                                     RMSE = reg_model_6_rmse ))



#### Regularized Month Effect


regularization <- function(lambda,edx_train_l,edx_test_l){
  mu <- mean(edx_train_l$rating)
  b_i <- edx_train_l %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx_train_l %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <-edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))
  b_m <- edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = month(timestamp)) %>% 
    group_by(timestamp) %>% 
    summarize(b_m = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))
  
  predicted_ratings <- 
    edx_test_l %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = month(timestamp)) %>% 
    left_join(b_m, by = "timestamp") %>% 
    filter(!is.na(b_i), !is.na(b_u),!is.na(b_g),!is.na(b_m)) %>% 
    mutate(pred = mu + b_i + b_u +b_g+b_m) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
}


lambdas <- seq(0,10,0.25)

lambdas_rmse <- sapply(lambdas,
                       regularization,
                       edx_train_l = edx_train_l,
                       edx_test_l = edx_test_l)

lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)

lambdas_tibble



lambda <- lambdas[which.min(lambdas_rmse)]


mu <- mean(edx_train$rating)

b_i_reg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_g_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))

b_m_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  left_join(b_g_reg, by ="genres") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = month(timestamp)) %>% 
  group_by(timestamp) %>% 
  summarize(b_m = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))


preds <- edx_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_g_reg, by = "genres") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = month(timestamp)) %>% 
  left_join(b_m_reg, by = "timestamp") %>% 
  mutate(prediction = mu + b_i +b_u+b_g+b_m) %>% 
  pull(prediction)

reg_model_7_rmse <- RMSE(preds, edx_test$rating)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized with Month Effect Model",  
                                     RMSE = reg_model_7_rmse ))



##### Regularized Week Effects



regularization <- function(lambda,edx_train_l,edx_test_l){
  mu <- mean(edx_train_l$rating)
  b_i <- edx_train_l %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx_train_l %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <-edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))
  b_w <- edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = week(timestamp)) %>% 
    group_by(timestamp) %>% 
    summarize(b_w = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))
  
  predicted_ratings <- 
    edx_test_l %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(timestamp = week(timestamp)) %>% 
    left_join(b_w, by = "timestamp") %>% 
    filter(!is.na(b_i), !is.na(b_u),!is.na(b_g),!is.na(b_w)) %>% 
    mutate(pred = mu + b_i + b_u +b_g+b_w) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
}


lambdas <- seq(0,10,0.25)

lambdas_rmse <- sapply(lambdas,
                       regularization,
                       edx_train_l = edx_train_l,
                       edx_test_l = edx_test_l)

lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)

lambdas_tibble



lambda <- lambdas[which.min(lambdas_rmse)]


mu <- mean(edx_train$rating)

b_i_reg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_g_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))

b_w_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  left_join(b_g_reg, by ="genres") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = week(timestamp)) %>% 
  group_by(timestamp) %>% 
  summarize(b_w = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))


preds <- edx_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_g_reg, by = "genres") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = week(timestamp)) %>% 
  left_join(b_w_reg, by = "timestamp") %>% 
  mutate(prediction = mu + b_i +b_u+b_g+b_w) %>% 
  pull(prediction)

reg_model_8_rmse <- RMSE(preds, edx_test$rating)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized with Weeks Effect Model",  
                                     RMSE = reg_model_8_rmse ))




####### Regularized Genre + Year + Hour + Month + week




regularization <- function(lambda,edx_train_l,edx_test_l){
  mu <- mean(edx_train_l$rating)
  b_i <- edx_train_l %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx_train_l %>% 
    left_join(b_i, by="movieId") %>%
    filter(!is.na(b_i)) %>% 
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  b_g <-edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    filter(!is.na(b_i), !is.na(b_u)) %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))
  b_y <- edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(year = year(timestamp)) %>% 
    group_by(year) %>% 
    summarize(b_y = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))
  b_m <- edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(year = year(timestamp)) %>% 
    left_join(b_y, by = "year") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(month = month(timestamp)) %>%
    group_by(month) %>% 
    summarize(b_m = sum(rating -b_i-b_u-b_g-b_y-mu)/(n()+lambda))
  b_w <-edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(year = year(timestamp)) %>% 
    left_join(b_y, by = "year") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(month = month(timestamp)) %>%
    left_join(b_m, by = "month") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(week = week(timestamp)) %>% 
    group_by(week) %>% 
    summarize(b_w = sum(rating -b_i-b_u-b_g-b_y-b_m-mu)/(n()+lambda))
  b_h <- edx_train_l %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(year = year(timestamp)) %>% 
    left_join(b_y, by = "year") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(month = month(timestamp)) %>%
    left_join(b_m, by = "month") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(week = week(timestamp)) %>% 
    left_join(b_w, by = "week") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(hour = hour(timestamp)) %>%
    group_by(hour) %>% 
    summarize(b_h = sum(rating -b_i-b_u-b_g-b_y-b_m-b_w-mu)/(n()+lambda))
  
  predicted_ratings <- 
    edx_test_l %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(year = year(timestamp)) %>% 
    left_join(b_y, by = "year") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(month = month(timestamp)) %>% 
    left_join(b_m, by = "month") %>% 
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(week = week(timestamp)) %>% 
    left_join(b_w, by = "week") %>%
    mutate(timestamp = as_datetime(timestamp)) %>%
    mutate(hour= hour(timestamp)) %>% 
    left_join(b_h, by = "hour") %>% 
    filter(!is.na(b_i), !is.na(b_u),!is.na(b_g),!is.na(b_y),!is.na(b_m),
           !is.na(b_w),!is.na(b_h)) %>% 
    mutate(pred = mu + b_i + b_u +b_g+b_y+b_m+b_w+b_h) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx_test_l$rating))
}


lambdas <- seq(0,10,0.25)

lambdas_rmse <- sapply(lambdas,
                       regularization,
                       edx_train_l = edx_train_l,
                       edx_test_l = edx_test_l)

lambdas_tibble <- tibble(Lambda = lambdas, RMSE = lambdas_rmse)

lambdas_tibble



lambda <- lambdas[which.min(lambdas_rmse)]


library(gghighlight)
ggplot(data =lambdas_tibble,aes(x = Lambda, y = RMSE))+
  geom_point(col ="coral")+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  xlab("Lambdas")+
  ylab("Rmse")+
  gghighlight(RMSE == min(RMSE))





mu <- mean(edx_train$rating)

b_i_reg <- edx_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_g_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating -b_i-b_u-mu)/(n()+lambda))

b_y_reg <- edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  left_join(b_g_reg, by ="genres") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(year = year(timestamp)) %>% 
  group_by(year) %>% 
  summarize(b_y = sum(rating -b_i-b_u-b_g-mu)/(n()+lambda))

b_m_reg <-edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  left_join(b_g_reg, by ="genres") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(year = year(timestamp)) %>% 
  left_join(b_y_reg, by = "year") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(month = month(timestamp)) %>% 
  group_by(month) %>% 
  summarize(b_m = sum(rating -b_i-b_u-b_g-b_y-mu)/(n()+lambda))
  
b_w_reg <-edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  left_join(b_g_reg, by ="genres") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(year = year(timestamp)) %>% 
  left_join(b_y_reg, by = "year") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(month = month(timestamp)) %>% 
  left_join(b_m_reg, by = "month") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(week = week(timestamp)) %>% 
  group_by(week) %>% 
  summarize(b_w = sum(rating -b_i-b_u-b_g-b_y-b_m-mu)/(n()+lambda))

b_h_reg <-edx_train %>% 
  left_join(b_i_reg, by="movieId") %>% 
  left_join(b_u_reg, by="userId") %>% 
  left_join(b_g_reg, by ="genres") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(year = year(timestamp)) %>% 
  left_join(b_y_reg, by = "year") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(month = month(timestamp)) %>% 
  left_join(b_m_reg, by = "month") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(week = week(timestamp)) %>% 
  left_join(b_w_reg, by = "week") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(hour = hour(timestamp)) %>% 
  group_by(hour) %>% 
  summarize(b_h = sum(rating -b_i-b_u-b_g-b_y-b_m-b_w-mu)/(n()+lambda))




preds <- edx_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>% 
  left_join(b_g_reg, by = "genres") %>%
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(year = year(timestamp)) %>% 
  left_join(b_y_reg, by = "year") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(month = month(timestamp)) %>% 
  left_join(b_m_reg, by = "month") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(week = week(timestamp)) %>% 
  left_join(b_w_reg, by = "week") %>% 
  mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(hour = hour(timestamp)) %>% 
  left_join(b_h_reg, by = "hour") %>% 
  mutate(prediction = mu + b_i +b_u+b_g+b_y+b_m+b_w+b_h) %>% 
  pull(prediction)

final_reg_model_rmse <- RMSE(preds, edx_test$rating)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Final Regularized Model",  
                                     RMSE = final_reg_model_rmse ))











############### Matrix Factorization
################### using recosystem

library(recosystem)
set.seed(123)

# Converting into recosystem input format:

train_reco <- with(edx_train,data_memory(user_index = userId,
                                         item_index = movieId,
                                         rating = rating))

test_reco <- with(edx_test, data_memory(user_index = userId,
                                        item_index = movieId,
                                        rating = rating))


# Creating model object:

mf_model <- Reco()

# Tuning model:

tuning <- mf_model$tune(train_reco, opts = list(dim = c(20,30),
                                              lrate = c(0.01,0.1),
                                              nthread = 4,
                                              niter = 10))
# Training model:

mf_model$train(train_reco, opts = c(tuning$min,
                                  nthread = 4,
                                  niter = 40))

# Prediction:
mf_preds <- mf_model$predict(test_reco, out_memory())


mf_model_rmse <- RMSE(mf_preds,edx_test$rating)


rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Matrix Factorization Model",  
                                     RMSE = mf_model_rmse ))





### Matrix Factorization with Final holding test set

set.seed(123)

# Converting into recosystem input format:

train_reco <- with(edx_dat,data_memory(user_index = userId,
                                         item_index = movieId,
                                         rating = rating))

test_reco <- with(final_holdout_test, data_memory(user_index = userId,
                                        item_index = movieId)) # rating omitted since we would not have it 


# Creating model object:

mf_model <- Reco()

# Tuning model:

tuning <- mf_model$tune(train_reco, opts = list(dim = c(20,30),
                                              lrate = c(0.01,0.1),
                                              nthread = 4,
                                              niter = 10))
# Training model:

mf_model$train(train_reco, opts = c(tuning$min,
                                  nthread = 4,
                                  niter = 40))

# Prediction:
mf_preds <- mf_model$predict(test_reco, out_memory())


final_mf_model_rmse <- RMSE(mf_preds,final_holdout_test$rating)



rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Final Matrix Factorization Model",  
                                     RMSE = final_mf_model_rmse ))


library(knitr)
library(kableExtra)
kable(rmse_results)

rmse_results %>% 
  kbl() %>% 
  kable_material_dark(full_width = F) %>% 
  column_spec(2, color = "white", 
              background = spec_color(rmse_results$RMSE[1:17], end = 0.7,
                                      direction = -1,option="E"))



########## Visualization
######## All Models stacked against each other 


# Higher is worse
rmse_results %>% 
  ggplot(aes(method,RMSE, fill = RMSE))+
  geom_bar(stat = "identity")+
  theme(axis.text.x = element_text(angle=70,hjust = 1,size = 5))+
  coord_cartesian(ylim = c(0.5,1.25))+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  xlab("Method")+
  ggtitle("Model Evaluation")+
  scale_fill_gradient2(low="coral",high = "azure4", mid = "azure3",
                       midpoint = 0.9)



# Performance based
rmse_results %>% 
  mutate(RMSE = 2- RMSE) %>% 
  arrange(desc(RMSE)) %>% 
  ggplot(aes(RMSE,reorder(method,RMSE), fill = RMSE))+
  geom_bar(stat = "identity")+
  theme(axis.title.y = element_blank())+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  coord_cartesian(xlim = c(0.4,1.25))+
  scale_x_continuous(breaks= c(0.4,0.6,0.8,1.0,1.2), labels= c(1.6,1.4,1.2,1,0.8))+
  scale_fill_gradient2(low="azure4",high = "coral", mid = "azure3",
                       midpoint = 1.1,breaks= c(0.6,0.8,1.0,1.2), 
                       labels= c(1.4,1.2,1,"below target"))






