
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



ggsave("movie_yrs_rating_count.png")

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








