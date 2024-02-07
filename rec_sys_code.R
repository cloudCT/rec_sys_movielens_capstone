
## NOTE:
# This script serves as an all-in-one inclusion of the scripts. The code was
# originally written in separate scripts.
# They can be seen in their original format on github.
# https://github.com/cloudCT/rec_sys_movielens_capstone.git


##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



############################################################################
############################################################################
######### 
#############
################### Model - Preparation


####### Packages used
## This can also be found in the README file

### Libraries to download:
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(lubridate)) install.packages("lubridate")
if(!require(rmarkdown)) install.packages("rmarkdown")
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(stringr)) install.packages("stringr")
if(!require(scales)) install.packages("scales")

if(!require(widyr)) install.package("widyr")
if(!require(knitr)) install.package("knitr")
if(!require(gghighlight)) install.package("gghighlight")
if(!require(RColorBrewer)) install.packages("RColorBrewer")
if(!require(kableExtra)) install.packages("kableExtra")

# Used in data exploration:
if(!require(wordcloud)) install.package("wordcloud") 

# Uncomment to download parallel computing package
# install.package("doParallel") 


# Later used in Matrix Factorization:
if(!require(recosystem)) install.package("recosystem")


## Later use in recommenderlab models
if(!require(recommenderlab)) install.packages("recommenderlab")
if(!require(reshape2)) install.packages("reshape2")


### Loading Libraries:

library(tidyverse)
library(caret)
# library(doParallel) # If wanted
library(tidyr)
library(scales)
library(widyr)
library(knitr)
library(wordcloud)
library(recosystem)
library(kableExtra)



###########
############# Preparing Dataset and creating further test and train sets for 
############# Model testing



## Putting years in separate column

edx_dat <- edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))"),
                          title = str_replace(title,"(\\(\\d{4}\\))$","")) %>%
  mutate(year = as.numeric(str_extract(year,"\\d{4}")))

edx_dat



# Note: The difference between using the actual sets, using only one test/train
#       set and the approach used here with two test/train sets to find the 
#       lambda in Regularization is negligible and they all yield the same
#       result. The decision to split the dataset twice, was made, in order to
#       accurately simulate a situation where unknown ratings need to be
#       predicted. The first set is our primary dataset for testing our models,
#       while the second set was created from the first to calculate the lambdas
#       without introducing bias.


## Test and train (Used for overall testing and to find final models lambda)

test_index <- createDataPartition(y = edx_dat$rating, times = 1, p = 0.1, list = FALSE)

edx_train <- edx_dat[-test_index,]
temp <- edx_dat[test_index,]

edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

removed <- anti_join(temp, edx_test)
edx_train <- rbind(edx_train, removed)

rm(test_index, temp, removed)


## Test and train set to find proper testing lambda


test_index <- createDataPartition(y = edx_train$rating, times = 1, p = 0.1, list = FALSE)

edx_train_l <- edx_train[-test_index,]
temp <- edx_train[test_index,]

edx_test_l <- temp %>% 
  semi_join(edx_train_l, by = "movieId") %>%
  semi_join(edx_train_l, by = "userId")

removed <- anti_join(temp, edx_test_l)
edx_train_l <- rbind(edx_train_l, removed)

rm(test_index, temp, removed)








############################################################################
############################################################################
######### 


#############
################### Initial - Data - Exploration


# To run code quicker with the use of parallel computing:
# Important Note: Make sure you choose number of cores correctly!

library(doParallel)
# detectCores() # - to find out number of cores available to you
registerDoParallel(cores = 8) # Device with 10 cores 
# (only incremental change the more cores are used)


# Lets first take a quick look at our dataset:

head(edx)


####
## How many columns and rows?
nrow(edx)
ncol(edx)


####
## How many zero ratings and how many average ratings of 3?
sum(edx$rating == 0)
sum(edx$rating == 3)



####
## How many movies?
n_distinct(edx$movieId)
####
## How many users?
n_distinct(edx$userId)


#### What are the most rated movies?

## What movies have the greatest number of ratings?

edx %>% group_by(movieId) %>% mutate(count = n()) %>% arrange(desc(count)) %>% 
  distinct(movieId,title,count)


edx %>% 
  group_by(title) %>% 
  summarize(count = n()) %>% 
  arrange(desc(count)) %>% 
  top_n(20,count) %>% 
  ggplot(aes(count,reorder(title,count),fill = count)) +
  geom_bar(stat = "identity")+
  scale_fill_gradient2(low="darkseagreen",high = "azure3", mid = "antiquewhite4", midpoint = 25000)+
  xlab("Count")+
  ylab(NULL) +
  theme_minimal()+
  ggtitle("Most Rated Movies")+
  theme( plot.title = element_text(size = (10)),
         panel.background = element_rect(fill = "cornsilk"))+
  scale_x_continuous(labels = unit_format(unit = "K", scale = 1e-3))

ggsave("figs/most_rated_movies.png")


#### Most common Ratings
## What are the most common ratings?

edx %>% group_by(rating) %>% mutate(count = n()) %>% arrange(desc(count)) %>% 
  distinct(rating,count)



edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line(col = "coral")+
  xlab("Rating")+
  ylab("Count")+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  scale_y_continuous(labels = unit_format(unit = "M", scale = 1e-6))

ggsave("figs/most_common_ratings.png")

#### User activity distribution

edx %>% 
  group_by(userId) %>%
  summarize(n = n()) %>% 
  ggplot(aes(userId,n))+
  geom_point(alpha = 0.1)+
  theme(axis.text.x = element_blank())+
  xlab("Users")+
  ylab("Rated Movies")+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  ggtitle("User Activity")

ggsave("figs/user_activity_distribution.png")


#### Number of ratings against rating mean by User ID

edx %>% 
  group_by(userId) %>% 
  summarize(n = n(), rating = mean(rating)) %>% 
  ggplot(aes(rating,n))+
  geom_point(alpha= 0.3, col = "cadetblue")+
  ggtitle("Rating Sum vs Rating Mean by UserID")+
  xlab("Average Rating")+
  ylab("Times rated")+
  theme( panel.background = element_rect(fill = "cornsilk"))

ggsave("figs/number_of_ratings_against_rating_mean_uid.png")

#### Number of ratings against rating mean by Movie ID

edx %>% 
  group_by(movieId) %>% 
  summarize(n = n(), rating = mean(rating)) %>% 
  ggplot(aes(rating,n))+ 
  geom_point(alpha= 0.3, col = "cadetblue")+
  ggtitle("Rating Sum vs Rating Mean by MovieID")+
  xlab("Average Rating")+
  ylab("Times rated")+
  theme( panel.background = element_rect(fill = "cornsilk"))

ggsave("figs/number_of_ratings_against_rating_mean_mid.png")


######
## Genre Exploration and splitting genres to find number of ratings of selected
## few genres

# Note: Many movies have more than one genre split by a "|" delimiter

library(tidyr)

# edx %>% filter(str_detect(.$genres,"Comedy|Drama|Thriller|Romance")) %>% 
#   separate_longer_delim(genres, delim = "\\|")


edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  filter(genres %in% c("Comedy","Drama","Thriller","Romance")) %>% 
  summarize(n = n())



## What are all the genres?
edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n))


### Visualization of unique Movies per Genre


edx %>% 
  select(movieId,genres) %>% 
  unique() %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  ggplot(aes(n,reorder(genres,n),fill = n))+
  geom_bar(stat = "identity")+
  scale_fill_gradient2(low="darkseagreen",high = "azure3", mid = "antiquewhite4", 
                       midpoint = 1500,
                       labels = unit_format(unit = "K", scale = 1e-3))+
  xlab("Count")+
  ylab(NULL) +
  theme_minimal()+
  ggtitle("Number of unique Movies by Genre")+
  theme( plot.title = element_text(size = (10)),
         panel.background = element_rect(fill = "cornsilk"))+
  scale_x_continuous(labels = unit_format(unit = "K", scale = 1e-3))+
  labs(fill = "Number of Movies")



ggsave("figs/unique_movieids.png")

### Genre Wordcloud
library(wordcloud)

wrd_cloud <- edx %>% mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n()) 
wordcloud(words = wrd_cloud$genres, freq = wrd_cloud$n, min.freq = 10, max.words = 10,
          random.order = F, rot.per = 0.35, scale = c(5,0.2), font = 4,
          random.color = F, colors = brewer.pal(8,"Spectral"),
          main = "Most Rated Genres")


ggsave("figs/wordcloud.png")

####### Genre Groups
## Percentage of multiple genres supplied for one movie

mean(str_detect(edx$genres, "\\|")) # Percentage of multiple genres

mean(str_detect(edx$genres, "^[\\-A-Za-z]+$")) # Only one genre
mean(str_detect(edx$genres, "^\\w+\\-*\\w+$"))



## Clean genre group sizes:
grp_sz <- edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres,userId) %>% 
  group_by(movieId,userId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  select(movieId,userId,n_genres) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres) %>% 
  group_by(n_genres) %>% 
  summarize(n = n()) %>% 
  distinct() %>% 
  arrange(desc(n))
grp_sz


# Visualization of group sizes:
library(scales)

grp_sz %>% ggplot(aes(factor(n_genres),n)) +
  geom_col(fill = "bisque4")+
  ylab("Count in Millions") +
  xlab("Group Size") +
  theme(axis.text.x = element_text(angle = 90))+
  scale_y_continuous(labels = unit_format(unit = "M", scale = 1e-6))+
  theme_light()+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))


ggsave("figs/group_sizes.png")

## Number of group size appearances by unique MovieID (no repeated movieId's)
edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(genres) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  distinct() %>% 
  arrange(desc(n)) 

# Visualization of group size appearance by unique MovieID

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(genres) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  distinct() %>% 
  arrange(desc(n)) %>% 
  ggplot(aes(factor(genres),n)) +
  geom_col(fill = "azure4")+
  ylab("Unique Movies") +
  xlab("Group Size") +
  theme(axis.text.x = element_text(angle = 90))+
  theme_light()+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  ggtitle("Group Size Appearance by Movie ID")


ggsave("figs/grp_sz_count_unique_movieid.png")

### Visualization of group size appearance by User ID


# edx %>% 
#   mutate(genres =str_split(genres, "\\|")) %>% 
#   unnest(cols = c(genres)) %>% 
#   select(userId,genres) %>% 
#   group_by(userId) %>% 
#   mutate(genres = n_distinct(genres)) %>% 
#   distinct() %>% 
#   ungroup() %>% 
#   select(genres) %>% 
#   group_by(genres) %>% 
#   summarize(n = n()) %>% 
#   distinct() %>% 
#   arrange(desc(n)) %>% 
#   ggplot(aes(factor(genres),n)) +
#   geom_col(fill = "azure4")+
#   ylab("Number of Ratings by User") +
#   xlab("Group Size") +
#   theme(axis.text.x = element_text(angle = 90))+
#   theme_light()+
#   theme(panel.background = element_rect(fill = "cornsilk"))+
#   ggtitle("Group Size Appearance by User ID")




# Looking at lone 8 genre movie

index <- edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres,movieId) %>% 
  group_by(n_genres) %>% 
  filter(n_genres == 8) %>% 
  distinct() %>% 
  select(movieId) %>% 
  as.list()

edx %>% filter(movieId %in% index$movieId) %>% select(movieId,genres)


# Looking at 7 genre movie:
index <- edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres,movieId) %>% 
  group_by(n_genres) %>% 
  filter(n_genres == 7) %>% 
  distinct() %>% 
  select(movieId) %>% 
  as.list()

edx %>% filter(movieId %in% index$movieId) %>% select(movieId,genres)# %>% head()

# Note: Looked at 7 and 8 genre movies to find out if the Genres are alphabetical
#       or if there might be order importance.
#       Found out its indeed alphabetical



## What genres do we find alone most often?

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres,movieId,genres) %>% 
  group_by(n_genres) %>% 
  filter(n_genres == 1) %>% 
  ungroup() %>% 
  distinct() %>% 
  select(genres) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n))


# Visualization of lone genres

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres,movieId,genres) %>% 
  group_by(n_genres) %>% 
  filter(n_genres == 1) %>% 
  ungroup() %>% 
  distinct() %>% 
  select(genres) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>%
  ggplot(aes(n,reorder(genres,n)))+
  geom_col() +
  theme(panel.background = element_rect(fill = "cornsilk"))+
  xlab("Count")+
  ylab("Genre")+
  ggtitle("Lone Appearance Count")


ggsave("figs/lone_genres.png")

## What genres appear together most often?
library(widyr)
edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  pairwise_count(.,genres,movieId) %>% 
  arrange(desc(n))


# Visualization most common genre pairs

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  pairwise_count(.,genres,movieId) %>% 
  arrange(desc(n)) %>% 
  top_n(10) %>% 
  unite(col = genres,"item1","item2", sep = "+") %>% 
  ggplot(aes(n,reorder(genres,n)))+
  geom_col() +
  theme(panel.background = element_rect(fill = "cornsilk"))+
  ggtitle("Most Common Genre Pairs")+
  xlab("Appearances Together")+
  ylab("Genre Pairs")


ggsave("figs/most_common_genre_pairs.png")

## Are some genres correlated with each other?

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  pairwise_cor(.,genres,movieId) %>% 
  arrange(desc(abs(correlation)))

# Note: Only somewhat noteable correlation was Children and Animation



####### NA's?
### Are there any NA's in the dataset?

sum(is.na(edx$userId))
sum(is.na(edx$movieId))
sum(is.na(edx$timestamp))
sum(is.na(edx$title))

## Ratings that are 0:
sum(edx$rating == 0)

## No genres provided:

sum(edx$genres == "(no genres listed)")

edx %>% filter(genres == "(no genres listed)")

edx %>% filter(movieId == 8606)



####### Genre Performance
### Do certain genres do better than others?

g_p <- edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n(), rating = mean(rating)) %>% 
  arrange(desc(rating)) %>% 
  mutate(ranking = rank(-n))

cor(g_p$rating,g_p$ranking) # Do genres that are being rated more simply do better
# because of the sample size or vice versa?



### Genre Ratings Average

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(rating = mean(rating)) %>% 
  ggplot(aes(reorder(genres,rating),rating,fill = genres))+
  geom_col()+
  coord_cartesian(ylim = c(3,5))+
  theme(axis.text.x = element_text(angle=70,hjust = 1,size = 5))+
  theme(panel.background = element_rect(fill = "cornsilk")) +
  theme(axis.title.x = element_blank())+
  ylab("Rating")


ggsave("figs/genre_ratings_avg.png")

## Visualization of Ratings per Genre


edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  ggplot(aes(n,reorder(genres,n),fill = n))+
  geom_bar(stat = "identity")+
  scale_fill_gradient2(low="darkseagreen",high = "azure3", mid = "antiquewhite4", 
                       midpoint = 1500000,
                       labels = unit_format(unit = "M", scale = 1e-6))+
  xlab("Count")+
  ylab(NULL) +
  theme_minimal()+
  ggtitle("Most Rated Genres")+
  theme( plot.title = element_text(size = (10)),
         panel.background = element_rect(fill = "cornsilk"))+
  scale_x_continuous(labels = unit_format(unit = "M", scale = 1e-6))+
  labs(fill = "Times rated")



ggsave("figs/ratings_per_genre.png")


# Errorbar plot of genre performance
edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  mutate(n = n(),avg = mean(rating),se = sd(rating)/sqrt(n))  %>%
  select(genres,rating,n,se,avg) %>% 
  distinct() %>% 
  ggplot(aes(genres,avg, ymin = avg - 2*se ,ymax =avg +2*se)) +
  geom_errorbar()+
  theme(axis.text.x = element_text(angle = 90, size = 8)) +
  theme(panel.background = element_rect(fill = "cornsilk"))+
  ggtitle("Errorbar Plot of Genre Performance")+
  ylab("Average Rating")+
  xlab("Genres")


ggsave("figs/errorbar_genres.png")





####### Ratings distribution
### Ratings distribution by userId

edx %>% group_by(userId) %>% 
  mutate(rating = mean(rating)) %>% 
  ungroup() %>% 
  select(userId,rating) %>% 
  distinct() %>% 
  group_by(rating) %>%
  mutate_at("rating",round,1) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(rating,n))+
  xlab("Rating")+
  ylab("Number of Ratings")+
  geom_point(color = "coral")


ggsave("figs/ratings_dis.png")

### Do Titles differ?
## Do movie names with the same id have different names


edx %>% select(movieId,title) %>% 
  group_by(movieId) %>% 
  distinct() %>% 
  mutate(n = n()) %>% 
  arrange(desc(n))



####### Movie years exploration
## Extracting movie release years

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year) %>% 
  distinct() %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  slice_max(n,n = 10)



## Visualization of movie years

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year) %>% 
  distinct() %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(year,n)) +
  geom_col() +
  theme(axis.text.x = element_text(angle=70,hjust = 1,size = 5)) +
  ylab("Movie Count") +
  xlab("Year") +
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))


ggsave("figs/movie_yrs.png")

## Number of ratings by movie release year
library(scales)

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year,userId) %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(year,n)) +
  geom_col() +
  theme(axis.text.x = element_text(angle=70,hjust = 1,size = 5)) +
  ylab("Ratings Count (in thousands)")+
  xlab("Movie Release Year") +
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))+
  scale_y_continuous(labels = unit_format(unit = "K", scale = 1e-3))



ggsave("figs/movie_yrs_rating_count.png")

## Distribution of movie release years

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year) %>% 
  distinct() %>% 
  mutate(year= as.numeric(year)) %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  summarize(past_80 = mean(year >= 1980),
            between_80_95 = mean(year >=1980 & year <= 1995),
            past_95 = mean(year >= 1995))


# Distribution of ratings by release year

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year,userId) %>% 
  mutate(year= as.numeric(year)) %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  summarize(between_90_03 = mean(year >=1990 & year <= 2003),
            past_90 = mean(year >= 1990))


## Average rating by year

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year,userId,rating) %>% 
  mutate(year = as.numeric(year)) %>% 
  group_by(year) %>% 
  summarize(rating = mean(rating)) %>%
  select(year,rating) %>% 
  arrange(desc(rating))


## Correlation of movie release year and average rating

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year,userId,rating) %>% 
  mutate(year = as.numeric(year)) %>% 
  group_by(year) %>% 
  summarize(rating = mean(rating)) %>%
  select(year,rating) %>% 
  arrange(desc(rating)) %>% 
  ggplot(aes(year,rating))+
  geom_smooth(col = "coral")+
  ylab("Rating") +
  xlab("Release Year") +
  theme(panel.background = element_rect(fill = "cornsilk"))



ggsave("figs/cor_movie_re_yr_avg_rat.png")


####### Timestamps
## Exploration of ratings by time they were submitted



# Visualizing years
edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  select(movieId,userId,timestamp) %>%
  group_by(timestamp) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(timestamp,n)) +
  geom_point()+
  geom_smooth(se = F) +
  ggtitle("Number of Ratings by Submission Year")+
  ylab("Ratings Count (in thousands)")+
  xlab("Year")+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))+
  scale_y_continuous(labels = unit_format(unit = "K", scale = 1e-3))



ggsave("figs/years_plot.png")  

# Visualizing months
edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = month(timestamp)) %>% 
  select(movieId,userId,timestamp) %>%
  group_by(timestamp) %>% 
  summarize(n = n()) %>%
  ggplot(aes(timestamp,n)) +
  geom_point()+
  geom_smooth(method = "lm", se = F)+
  ggtitle("Number of Ratings by Submission Month")+
  ylab("Ratings Count (in thousands)")+
  xlab("Month")+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))+
  scale_y_continuous(labels = unit_format(unit = "K", scale = 1e-3))+
  scale_x_discrete(limits = 1:12,labels = c("Jan","Feb","Mar","Apr","May","Jun","Jul",
                                            "Aug","Oct","Nov","Sep","Dec"))



ggsave("figs/month_plot.png")

# Visualizing hours
edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = hour(timestamp)) %>% 
  select(movieId,userId,timestamp) %>%
  group_by(timestamp) %>% 
  summarize(n = n()) %>%
  ggplot(aes(timestamp,n)) +
  geom_smooth(color = "coral")+
  ggtitle("Number of Ratings by Submission Hour")+
  ylab("Ratings Count (in thousands)")+
  xlab("Hour")+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))


ggsave("figs/hours_plot.png")

# Avg year rating
edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  arrange(desc(rating))

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  ggplot(aes(timestamp,rating)) +
  geom_point()+
  xlab("Year")+
  ylab("Average Rating")+
  ggtitle("Average Rating by Year")+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))+
  annotate(geom = "text", y = 4.0,x = 1998, label = "< small sample size")

ggsave("figs/avg_yr_rating.png")

# Avg month rating

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = month(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  arrange(desc(rating))

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = month(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  ggplot(aes(timestamp,rating)) +
  geom_point()+
  xlab("Month")+
  ylab("Average Rating")+
  ggtitle("Average Rating by Month")+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  scale_x_discrete(limits = 1:12,labels = c("Jan","Feb","Mar","Apr","May","Jun","Jul",
                                            "Aug","Oct","Nov","Sep","Dec"))

ggsave("figs/avg_month_rating.png")

# Avg hour rating

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = hour(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  arrange(desc(rating))

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = hour(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  ggplot(aes(timestamp,rating)) +
  geom_point()+
  xlab("Hour")+
  ylab("Average Rating")+
  ggtitle("Average Rating by Hour")+
  theme(panel.background = element_rect(fill = "cornsilk"))


ggsave("figs/avg_hour_rating.png")


##### Sample of movies to show sparsity

users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>% 
  dplyr::select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  pivot_wider(names_from = movieId, values_from = rating) %>% 
  (\(mat) mat[, sample(ncol(mat), 100)])()%>%
  as.matrix() %>% 
  t() %>%
  image(1:100, 1:100,. , xlab="Movie ID", ylab="User ID", title = "Sparsity")
abline(h=0:100+0.5, v=0:100+0.5, col = "cornsilk")


ggsave("figs/sparsity_sample.png")



############################################################################
############################################################################
######### 


#############
################### Initial - Data - Exploration


# To run code quicker with the use of parallel computing:
# Important Note: Make sure you choose number of cores correctly!

library(doParallel)
# detectCores() # - to find out number of cores available to you
registerDoParallel(cores = 8) # Device with 10 cores 
# (only incremental change the more cores are used)


# Lets first take a quick look at our dataset:

head(edx)


####
## How many columns and rows?
nrow(edx)
ncol(edx)


####
## How many zero ratings and how many average ratings of 3?
sum(edx$rating == 0)
sum(edx$rating == 3)



####
## How many movies?
n_distinct(edx$movieId)
####
## How many users?
n_distinct(edx$userId)


#### What are the most rated movies?

## What movies have the greatest number of ratings?

edx %>% group_by(movieId) %>% mutate(count = n()) %>% arrange(desc(count)) %>% 
  distinct(movieId,title,count)


edx %>% 
  group_by(title) %>% 
  summarize(count = n()) %>% 
  arrange(desc(count)) %>% 
  top_n(20,count) %>% 
  ggplot(aes(count,reorder(title,count),fill = count)) +
  geom_bar(stat = "identity")+
  scale_fill_gradient2(low="darkseagreen",high = "azure3", mid = "antiquewhite4", midpoint = 25000)+
  xlab("Count")+
  ylab(NULL) +
  theme_minimal()+
  ggtitle("Most Rated Movies")+
  theme( plot.title = element_text(size = (10)),
         panel.background = element_rect(fill = "cornsilk"))+
  scale_x_continuous(labels = unit_format(unit = "K", scale = 1e-3))

ggsave("figs/most_rated_movies.png")


#### Most common Ratings
## What are the most common ratings?

edx %>% group_by(rating) %>% mutate(count = n()) %>% arrange(desc(count)) %>% 
  distinct(rating,count)



edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_line(col = "coral")+
  xlab("Rating")+
  ylab("Count")+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  scale_y_continuous(labels = unit_format(unit = "M", scale = 1e-6))

ggsave("figs/most_common_ratings.png")

#### User activity distribution

edx %>% 
  group_by(userId) %>%
  summarize(n = n()) %>% 
  ggplot(aes(userId,n))+
  geom_point(alpha = 0.1)+
  theme(axis.text.x = element_blank())+
  xlab("Users")+
  ylab("Rated Movies")+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  ggtitle("User Activity")

ggsave("figs/user_activity_distribution.png")


#### Number of ratings against rating mean by User ID

edx %>% 
  group_by(userId) %>% 
  summarize(n = n(), rating = mean(rating)) %>% 
  ggplot(aes(rating,n))+
  geom_point(alpha= 0.3, col = "cadetblue")+
  ggtitle("Rating Sum vs Rating Mean by UserID")+
  xlab("Average Rating")+
  ylab("Times rated")+
  theme( panel.background = element_rect(fill = "cornsilk"))

ggsave("figs/number_of_ratings_against_rating_mean_uid.png")

#### Number of ratings against rating mean by Movie ID

edx %>% 
  group_by(movieId) %>% 
  summarize(n = n(), rating = mean(rating)) %>% 
  ggplot(aes(rating,n))+ 
  geom_point(alpha= 0.3, col = "cadetblue")+
  ggtitle("Rating Sum vs Rating Mean by MovieID")+
  xlab("Average Rating")+
  ylab("Times rated")+
  theme( panel.background = element_rect(fill = "cornsilk"))

ggsave("figs/number_of_ratings_against_rating_mean_mid.png")


######
## Genre Exploration and splitting genres to find number of ratings of selected
## few genres

# Note: Many movies have more than one genre split by a "|" delimiter

library(tidyr)

# edx %>% filter(str_detect(.$genres,"Comedy|Drama|Thriller|Romance")) %>% 
#   separate_longer_delim(genres, delim = "\\|")


edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  filter(genres %in% c("Comedy","Drama","Thriller","Romance")) %>% 
  summarize(n = n())



## What are all the genres?
edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n))


### Visualization of unique Movies per Genre


edx %>% 
  select(movieId,genres) %>% 
  unique() %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  ggplot(aes(n,reorder(genres,n),fill = n))+
  geom_bar(stat = "identity")+
  scale_fill_gradient2(low="darkseagreen",high = "azure3", mid = "antiquewhite4", 
                       midpoint = 1500,
                       labels = unit_format(unit = "K", scale = 1e-3))+
  xlab("Count")+
  ylab(NULL) +
  theme_minimal()+
  ggtitle("Number of unique Movies by Genre")+
  theme( plot.title = element_text(size = (10)),
         panel.background = element_rect(fill = "cornsilk"))+
  scale_x_continuous(labels = unit_format(unit = "K", scale = 1e-3))+
  labs(fill = "Number of Movies")



ggsave("figs/unique_movieids.png")

### Genre Wordcloud
library(wordcloud)

wrd_cloud <- edx %>% mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n()) 
wordcloud(words = wrd_cloud$genres, freq = wrd_cloud$n, min.freq = 10, max.words = 10,
          random.order = F, rot.per = 0.35, scale = c(5,0.2), font = 4,
          random.color = F, colors = brewer.pal(8,"Spectral"),
          main = "Most Rated Genres")


ggsave("figs/wordcloud.png")

####### Genre Groups
## Percentage of multiple genres supplied for one movie

mean(str_detect(edx$genres, "\\|")) # Percentage of multiple genres

mean(str_detect(edx$genres, "^[\\-A-Za-z]+$")) # Only one genre
mean(str_detect(edx$genres, "^\\w+\\-*\\w+$"))



## Clean genre group sizes:
grp_sz <- edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres,userId) %>% 
  group_by(movieId,userId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  select(movieId,userId,n_genres) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres) %>% 
  group_by(n_genres) %>% 
  summarize(n = n()) %>% 
  distinct() %>% 
  arrange(desc(n))
grp_sz


# Visualization of group sizes:
library(scales)

grp_sz %>% ggplot(aes(factor(n_genres),n)) +
  geom_col(fill = "bisque4")+
  ylab("Count in Millions") +
  xlab("Group Size") +
  theme(axis.text.x = element_text(angle = 90))+
  scale_y_continuous(labels = unit_format(unit = "M", scale = 1e-6))+
  theme_light()+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))


ggsave("figs/group_sizes.png")

## Number of group size appearances by unique MovieID (no repeated movieId's)
edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(genres) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  distinct() %>% 
  arrange(desc(n)) 

# Visualization of group size appearance by unique MovieID

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(genres) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  distinct() %>% 
  arrange(desc(n)) %>% 
  ggplot(aes(factor(genres),n)) +
  geom_col(fill = "azure4")+
  ylab("Unique Movies") +
  xlab("Group Size") +
  theme(axis.text.x = element_text(angle = 90))+
  theme_light()+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  ggtitle("Group Size Appearance by Movie ID")


ggsave("figs/grp_sz_count_unique_movieid.png")

### Visualization of group size appearance by User ID


# edx %>% 
#   mutate(genres =str_split(genres, "\\|")) %>% 
#   unnest(cols = c(genres)) %>% 
#   select(userId,genres) %>% 
#   group_by(userId) %>% 
#   mutate(genres = n_distinct(genres)) %>% 
#   distinct() %>% 
#   ungroup() %>% 
#   select(genres) %>% 
#   group_by(genres) %>% 
#   summarize(n = n()) %>% 
#   distinct() %>% 
#   arrange(desc(n)) %>% 
#   ggplot(aes(factor(genres),n)) +
#   geom_col(fill = "azure4")+
#   ylab("Number of Ratings by User") +
#   xlab("Group Size") +
#   theme(axis.text.x = element_text(angle = 90))+
#   theme_light()+
#   theme(panel.background = element_rect(fill = "cornsilk"))+
#   ggtitle("Group Size Appearance by User ID")




# Looking at lone 8 genre movie

index <- edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres,movieId) %>% 
  group_by(n_genres) %>% 
  filter(n_genres == 8) %>% 
  distinct() %>% 
  select(movieId) %>% 
  as.list()

edx %>% filter(movieId %in% index$movieId) %>% select(movieId,genres)


# Looking at 7 genre movie:
index <- edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres,movieId) %>% 
  group_by(n_genres) %>% 
  filter(n_genres == 7) %>% 
  distinct() %>% 
  select(movieId) %>% 
  as.list()

edx %>% filter(movieId %in% index$movieId) %>% select(movieId,genres)# %>% head()

# Note: Looked at 7 and 8 genre movies to find out if the Genres are alphabetical
#       or if there might be order importance.
#       Found out its indeed alphabetical



## What genres do we find alone most often?

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres,movieId,genres) %>% 
  group_by(n_genres) %>% 
  filter(n_genres == 1) %>% 
  ungroup() %>% 
  distinct() %>% 
  select(genres) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n))


# Visualization of lone genres

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  group_by(movieId) %>% 
  mutate(n_genres = n_distinct(genres)) %>% 
  distinct() %>% 
  ungroup() %>% 
  select(n_genres,movieId,genres) %>% 
  group_by(n_genres) %>% 
  filter(n_genres == 1) %>% 
  ungroup() %>% 
  distinct() %>% 
  select(genres) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>%
  ggplot(aes(n,reorder(genres,n)))+
  geom_col() +
  theme(panel.background = element_rect(fill = "cornsilk"))+
  xlab("Count")+
  ylab("Genre")+
  ggtitle("Lone Appearance Count")


ggsave("figs/lone_genres.png")

## What genres appear together most often?
library(widyr)
edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  pairwise_count(.,genres,movieId) %>% 
  arrange(desc(n))


# Visualization most common genre pairs

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  pairwise_count(.,genres,movieId) %>% 
  arrange(desc(n)) %>% 
  top_n(10) %>% 
  unite(col = genres,"item1","item2", sep = "+") %>% 
  ggplot(aes(n,reorder(genres,n)))+
  geom_col() +
  theme(panel.background = element_rect(fill = "cornsilk"))+
  ggtitle("Most Common Genre Pairs")+
  xlab("Appearances Together")+
  ylab("Genre Pairs")


ggsave("figs/most_common_genre_pairs.png")

## Are some genres correlated with each other?

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  select(movieId,genres) %>% 
  pairwise_cor(.,genres,movieId) %>% 
  arrange(desc(abs(correlation)))

# Note: Only somewhat noteable correlation was Children and Animation



####### NA's?
### Are there any NA's in the dataset?

sum(is.na(edx$userId))
sum(is.na(edx$movieId))
sum(is.na(edx$timestamp))
sum(is.na(edx$title))

## Ratings that are 0:
sum(edx$rating == 0)

## No genres provided:

sum(edx$genres == "(no genres listed)")

edx %>% filter(genres == "(no genres listed)")

edx %>% filter(movieId == 8606)



####### Genre Performance
### Do certain genres do better than others?

g_p <- edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n(), rating = mean(rating)) %>% 
  arrange(desc(rating)) %>% 
  mutate(ranking = rank(-n))

cor(g_p$rating,g_p$ranking) # Do genres that are being rated more simply do better
# because of the sample size or vice versa?



### Genre Ratings Average

edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(rating = mean(rating)) %>% 
  ggplot(aes(reorder(genres,rating),rating,fill = genres))+
  geom_col()+
  coord_cartesian(ylim = c(3,5))+
  theme(axis.text.x = element_text(angle=70,hjust = 1,size = 5))+
  theme(panel.background = element_rect(fill = "cornsilk")) +
  theme(axis.title.x = element_blank())+
  ylab("Rating")


ggsave("figs/genre_ratings_avg.png")

## Visualization of Ratings per Genre


edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  ggplot(aes(n,reorder(genres,n),fill = n))+
  geom_bar(stat = "identity")+
  scale_fill_gradient2(low="darkseagreen",high = "azure3", mid = "antiquewhite4", 
                       midpoint = 1500000,
                       labels = unit_format(unit = "M", scale = 1e-6))+
  xlab("Count")+
  ylab(NULL) +
  theme_minimal()+
  ggtitle("Most Rated Genres")+
  theme( plot.title = element_text(size = (10)),
         panel.background = element_rect(fill = "cornsilk"))+
  scale_x_continuous(labels = unit_format(unit = "M", scale = 1e-6))+
  labs(fill = "Times rated")



ggsave("figs/ratings_per_genre.png")


# Errorbar plot of genre performance
edx %>% 
  mutate(genres =str_split(genres, "\\|")) %>% 
  unnest(cols = c(genres)) %>% 
  group_by(genres) %>% 
  mutate(n = n(),avg = mean(rating),se = sd(rating)/sqrt(n))  %>%
  select(genres,rating,n,se,avg) %>% 
  distinct() %>% 
  ggplot(aes(genres,avg, ymin = avg - 2*se ,ymax =avg +2*se)) +
  geom_errorbar()+
  theme(axis.text.x = element_text(angle = 90, size = 8)) +
  theme(panel.background = element_rect(fill = "cornsilk"))+
  ggtitle("Errorbar Plot of Genre Performance")+
  ylab("Average Rating")+
  xlab("Genres")


ggsave("figs/errorbar_genres.png")





####### Ratings distribution
### Ratings distribution by userId

edx %>% group_by(userId) %>% 
  mutate(rating = mean(rating)) %>% 
  ungroup() %>% 
  select(userId,rating) %>% 
  distinct() %>% 
  group_by(rating) %>%
  mutate_at("rating",round,1) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(rating,n))+
  xlab("Rating")+
  ylab("Number of Ratings")+
  geom_point(color = "coral")


ggsave("figs/ratings_dis.png")

### Do Titles differ?
## Do movie names with the same id have different names


edx %>% select(movieId,title) %>% 
  group_by(movieId) %>% 
  distinct() %>% 
  mutate(n = n()) %>% 
  arrange(desc(n))



####### Movie years exploration
## Extracting movie release years

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year) %>% 
  distinct() %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  slice_max(n,n = 10)



## Visualization of movie years

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year) %>% 
  distinct() %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(year,n)) +
  geom_col() +
  theme(axis.text.x = element_text(angle=70,hjust = 1,size = 5)) +
  ylab("Movie Count") +
  xlab("Year") +
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))


ggsave("figs/movie_yrs.png")

## Number of ratings by movie release year
library(scales)

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year,userId) %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(year,n)) +
  geom_col() +
  theme(axis.text.x = element_text(angle=70,hjust = 1,size = 5)) +
  ylab("Ratings Count (in thousands)")+
  xlab("Movie Release Year") +
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))+
  scale_y_continuous(labels = unit_format(unit = "K", scale = 1e-3))



ggsave("figs/movie_yrs_rating_count.png")

## Distribution of movie release years

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year) %>% 
  distinct() %>% 
  mutate(year= as.numeric(year)) %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  summarize(past_80 = mean(year >= 1980),
            between_80_95 = mean(year >=1980 & year <= 1995),
            past_95 = mean(year >= 1995))


# Distribution of ratings by release year

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year,userId) %>% 
  mutate(year= as.numeric(year)) %>% 
  group_by(year) %>% 
  summarize(n = n()) %>% 
  summarize(between_90_03 = mean(year >=1990 & year <= 2003),
            past_90 = mean(year >= 1990))


## Average rating by year

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year,userId,rating) %>% 
  mutate(year = as.numeric(year)) %>% 
  group_by(year) %>% 
  summarize(rating = mean(rating)) %>%
  select(year,rating) %>% 
  arrange(desc(rating))


## Correlation of movie release year and average rating

edx %>% mutate(year = str_extract(title, "(\\(\\d{4}\\))")) %>%
  mutate(year = str_extract(year,"\\d{4}")) %>% 
  select(movieId,year,userId,rating) %>% 
  mutate(year = as.numeric(year)) %>% 
  group_by(year) %>% 
  summarize(rating = mean(rating)) %>%
  select(year,rating) %>% 
  arrange(desc(rating)) %>% 
  ggplot(aes(year,rating))+
  geom_smooth(col = "coral")+
  ylab("Rating") +
  xlab("Release Year") +
  theme(panel.background = element_rect(fill = "cornsilk"))



ggsave("figs/cor_movie_re_yr_avg_rat.png")


####### Timestamps
## Exploration of ratings by time they were submitted



# Visualizing years
edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  select(movieId,userId,timestamp) %>%
  group_by(timestamp) %>% 
  summarize(n = n()) %>% 
  ggplot(aes(timestamp,n)) +
  geom_point()+
  geom_smooth(se = F) +
  ggtitle("Number of Ratings by Submission Year")+
  ylab("Ratings Count (in thousands)")+
  xlab("Year")+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))+
  scale_y_continuous(labels = unit_format(unit = "K", scale = 1e-3))



ggsave("figs/years_plot.png")  

# Visualizing months
edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = month(timestamp)) %>% 
  select(movieId,userId,timestamp) %>%
  group_by(timestamp) %>% 
  summarize(n = n()) %>%
  ggplot(aes(timestamp,n)) +
  geom_point()+
  geom_smooth(method = "lm", se = F)+
  ggtitle("Number of Ratings by Submission Month")+
  ylab("Ratings Count (in thousands)")+
  xlab("Month")+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))+
  scale_y_continuous(labels = unit_format(unit = "K", scale = 1e-3))+
  scale_x_discrete(limits = 1:12,labels = c("Jan","Feb","Mar","Apr","May","Jun","Jul",
                                            "Aug","Oct","Nov","Sep","Dec"))



ggsave("figs/month_plot.png")

# Visualizing hours
edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = hour(timestamp)) %>% 
  select(movieId,userId,timestamp) %>%
  group_by(timestamp) %>% 
  summarize(n = n()) %>%
  ggplot(aes(timestamp,n)) +
  geom_smooth(color = "coral")+
  ggtitle("Number of Ratings by Submission Hour")+
  ylab("Ratings Count (in thousands)")+
  xlab("Hour")+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))


ggsave("figs/hours_plot.png")

# Avg year rating
edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  arrange(desc(rating))

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = year(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  ggplot(aes(timestamp,rating)) +
  geom_point()+
  xlab("Year")+
  ylab("Average Rating")+
  ggtitle("Average Rating by Year")+
  theme(panel.background = element_rect(fill = "cornsilk"),
        text = element_text(family = "sans"))+
  annotate(geom = "text", y = 4.0,x = 1998, label = "< small sample size")

ggsave("figs/avg_yr_rating.png")

# Avg month rating

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = month(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  arrange(desc(rating))

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = month(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  ggplot(aes(timestamp,rating)) +
  geom_point()+
  xlab("Month")+
  ylab("Average Rating")+
  ggtitle("Average Rating by Month")+
  theme(panel.background = element_rect(fill = "cornsilk"))+
  scale_x_discrete(limits = 1:12,labels = c("Jan","Feb","Mar","Apr","May","Jun","Jul",
                                            "Aug","Oct","Nov","Sep","Dec"))

ggsave("figs/avg_month_rating.png")

# Avg hour rating

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = hour(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  arrange(desc(rating))

edx %>% mutate(timestamp = as_datetime(timestamp)) %>%
  mutate(timestamp = hour(timestamp)) %>% 
  select(movieId,userId,timestamp,rating) %>%
  group_by(timestamp) %>% 
  summarize(rating=mean(rating), n = n()) %>% 
  ggplot(aes(timestamp,rating)) +
  geom_point()+
  xlab("Hour")+
  ylab("Average Rating")+
  ggtitle("Average Rating by Hour")+
  theme(panel.background = element_rect(fill = "cornsilk"))


ggsave("figs/avg_hour_rating.png")


##### Sample of movies to show sparsity

users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>% 
  dplyr::select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  pivot_wider(names_from = movieId, values_from = rating) %>% 
  (\(mat) mat[, sample(ncol(mat), 100)])()%>%
  as.matrix() %>% 
  t() %>%
  image(1:100, 1:100,. , xlab="Movie ID", ylab="User ID", title = "Sparsity")
abline(h=0:100+0.5, v=0:100+0.5, col = "cornsilk")


ggsave("figs/sparsity_sample.png")




############################################################################
############################################################################
######### 
#############
################### Models



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









############################################################################
############################################################################
######### 
#############
################### Recommenderlab - Models




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











