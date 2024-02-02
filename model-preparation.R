

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
set.seed(1, sample.kind = "Rounding")
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

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx_train$rating, times = 1, p = 0.1, list = FALSE)

edx_train_l <- edx_train[-test_index,]
temp <- edx_train[test_index,]

edx_test_l <- temp %>% 
  semi_join(edx_train_l, by = "movieId") %>%
  semi_join(edx_train_l, by = "userId")

removed <- anti_join(temp, edx_test_l)
edx_train_l <- rbind(edx_train_l, removed)

rm(test_index, temp, removed)









