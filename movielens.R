
#####################################################################
#
# HarvardX PH125.9x - Capstone - Movielens Project
# Author: Kwaku Owusu-Tieku
# Submission Date: 3/9/2022
#
#####################################################################

# Loading libs
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer")
if(!require(colorspace)) install.packages("colorspace")


# the almighty tidyverse, you can't do without
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

# the beautiful colorbrewer, bring life to your charts
if (!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")

# for LaTeX (pdf) compilation
if(!require(tinytex)) {
  install.packages("tinytex", repos = "http://cran.us.r-project.org")
  tinytex::install_tinytex()  # install TinyTeX
}

# for the amazing tables package by Hao Zhu
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

# for arranging objects like charts horizontally and vertically using arrangeGrob(), etc
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(dplyr)
library(ggplot2)
library(broom)
library(scales)
library(RColorBrewer)
library(tinytex)
library(kableExtra)
library(gridExtra)
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(matrixStats)
library(RColorBrewer)
library(colorspace)

# set seed to ensure consistent 
set.seed(1, sample.kind="Rounding") 


# min number of rating per user needed to be included in the training set
MIN_USER_RATING_COUNT  <- 50

# min number of rating per movie needed to be included in the training set
MIN_MOVIE_RATING_COUNT <- 50


###############################################################
#
# supporting functions
#
###############################################################


###############################################################
# split genres and create column for each genre
###############################################################
fnProcessGenres <- function(dataset){
  
  dataset <- dataset %>% mutate(genre = str_split(genres, "\\|" )) %>%
    unnest(cols = genre) %>% mutate(is_used=1)
  dataset <- dataset %>%
    pivot_wider(names_from = genre, values_from=is_used, values_fill = 0)
  
  (genre_names <- names(dataset))
  
  ## rename the following genres
  
  if(which(genre_names == "Sci-Fi") > 0)
    names(dataset)[which(genre_names == "Sci-Fi")] <- "SciFi"
  
  if (which(genre_names == "Film-Noir") > 0)
    names(dataset)[which(genre_names == "Film-Noir")] <- "FilmNoir"
  
  if(which(genre_names == "(no genres listed)") > 0)
    names(dataset)[which(genre_names == "(no genres listed)")] <- "NoGenre"
  
  dataset %>% select(-c("genres") )
  
}

fnCreateIndexes <- function(dataset){
  createDataPartition(y = dataset$rating, times = 1, p = 0.1, list = FALSE)
}

###################################################################
##
## calculate the rmse used for measuring the model performance
## This measures how the predicted ratings deviate from the actual ratings
##
###################################################################
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


###################################################################
# fit a model for one specific lambda and return predictions
###################################################################
fnFitModelAndPredict <- function(lambda, train_set, test_set) {
  
  # 1. calculate the over mean for the training set
  mu <- mean(train_set$rating)
  #print (paste(Sys.time(), " - overall movie rating mean in current training set: ", round(mu, 2)))
  
  ## 2. regularize the movie bias
  movie_effects <- train_set %>% group_by(movieId) %>%
    summarise(b_i = sum( rating - mu) / (n()+lambda))
  
  ## 3. regularize the user bias
  user_effects <- train_set %>%
    left_join(movie_effects, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i) / (n()+lambda))
  
  # 4. make a predictions on the test based on the model
  preds <- test_set %>%
    left_join(movie_effects, by="movieId") %>%
    left_join(user_effects, by="userId") %>%
    mutate(predicted_rating  = mu + b_i + b_u) 
  
  RMSE(preds$rating, preds$predicted_rating) 
}


##################################################################
# this function allows us to perform cross validation.
# since the edx dataset is so large we can't use the standard cross validation,
# it won't fit in memory, so here I do it manually.
#
# we are given list of indexes from createDataPartition()
# we create train/test from that, and fit a model for it.
##################################################################
fnTrainAndTest <- function(train_test_indexes, dataset, lambdas){
  print(paste(Sys.time(), " - Starting function fnTrainAndTest()..."))
  
  # step 1. create train and test sets
  cv_train_set <- dataset[-train_test_indexes,]
  temp <- dataset[train_test_indexes,]
  
  # Make sure userId and movieId in test set are also in training set
  cv_test_set <- temp %>% 
    semi_join(cv_train_set, by = "movieId") %>%
    semi_join(cv_train_set, by = "userId")
  
  # Add rows removed from test set back into training set
  removed <- anti_join(temp, cv_test_set)
  
  cv_train <- rbind(cv_train_set, removed)
  
  rm(temp, removed)
  
  print(paste(Sys.time(), " - calling fnFitModelAndPredict()..."))
  # fit models for all the different lambdas for this train/test, and return predictions
  # return list of rmses for each lambda
  rmses <- sapply(lambdas, fnFitModelAndPredict, train_set=cv_train_set, test_set=cv_test_set)
  print(paste(Sys.time(), " - done calling fnFitModelAndPredict()."))

  print(paste(Sys.time(), " - Done with function fnTrainAndTest()..."))
  rmses
  
}

####################################################################
# create cross validation samples from dataset
####################################################################
fnCreateSamples <- function(num_samples, dataset, sample_size=SAMPLE_SIZE.SVD_MODELS)
{
  sample(nrow(dataset), sample_size) 
}




###############################################################
#
# End supporting functions
#
###############################################################



##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 3.6 or earlier:
##movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
##                                           title = as.character(title),
##                                           genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(ratings, movies, test_index, temp, movielens, removed)

saveRDS(edx, "rda/edx.rda")
saveRDS(validation, "rda/validation.rda")



###############################################################################
#
# checkpoint #1
# at this point main data loaded, split into edx and validation completed
#
# next : checkpoint #2 - data wrangling
###############################################################################

head(edx)
head(validation)

###############################################################################
#
# begin data wrangling
# requires checkpoint #1 to be completed
###############################################################################

print(paste(Sys.time(), " - start data wrangling..."))

# 1. split genres into columns and add to the dataset
print(paste(Sys.time(), " - processing genres..."))
edx <- fnProcessGenres(edx)
print(paste(Sys.time(), " - done processing genres."))

# 2. extract movie release year from titles
print(paste(Sys.time(), " - extracting movie release year..."))
edx <-  edx %>%
  mutate(release_year = str_extract_all(title, pattern = "\\(\\d{4}\\)")) 

edx$release_year <- edx$release_year %>% str_replace(pattern= "\\(", "")
edx$release_year <- edx$release_year %>% str_replace(pattern= "\\)", "")
edx <- edx %>% mutate(release_year = as.integer(release_year))

# convert review timestamp to datetime 
edx <- edx %>% mutate(timestamp = as_datetime(timestamp))

edx <- edx %>% mutate(movieId = as.integer(movieId))
edx <- edx %>% select(userId, movieId, rating, timestamp, release_year, title, everything())
edx <- edx %>% mutate(across(Comedy:NoGenre, as.integer))
glimpse(edx)

saveRDS(edx, "rda/edx_v1.rda")
print(paste(Sys.time(), " - data wrangling complete."))

###############################################################################
#
# checkpoint # 2. all data wrangling  should be completed at this point.
#
# next :  data visualization
###############################################################################



###############################################################################
#
# data visualization and exploration begins
# requires checkpoint # 2. data wrangling to be completed
###############################################################################


# illustrating variation in movie ratings.
# case 1: user bias -  a case where two users rating the same movie,
# one user always gives a higher rating than the other.
#
# pick some random movies and users to illustrate the point
# note these users and movies have been carefully selected.
# both  users have rated the selected movies


#1. pick two random users
some_two_random_users   <- c(8,18)

#2. pick random movies
some_four_random_movies <- edx %>% filter(title %in% c("Jumanji (1995)", " Babe (1995)", 
                                                       "Seven (a.k.a. Se7en) (1995)", 
                                                       "Beverly Hills Cop III (1994)", 
                                                       "Conan the Barbarian (1982)")) %>% 
  select(movieId, title) %>% distinct() %>% arrange(movieId)


# select movieIds
some_four_random_movie_ids <- some_four_random_movies %>% pull(movieId)

# create a list of rating for these users for the selected movies
movie_ratings_by_two_users <- edx %>% filter(movieId %in% some_four_random_movie_ids ) %>% 
  inner_join(edx %>% filter(userId %in%  some_two_random_users), 
             by = c("movieId", "userId")) %>% 
  select(userId, title=title.x, rating=rating.x) %>% 
  mutate(title=factor(title)) 

# create a visualization of it
user_bias_chart1 <- movie_ratings_by_two_users %>%  mutate(title = fct_reorder(title, rating)) %>% 
  ggplot(aes(x=factor(title), y=rating, fill=factor(userId))) + 
  geom_col(width=0.5,  position = position_dodge(0.6)) + 
  labs(title = 'Two users rating same movies (user bias)' , y="Rating", x="Movie", fill="User") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top") 

movie_ratings_by_two_users <- movie_ratings_by_two_users %>%  select(userId, title, rating)  %>% spread(title, rating) %>% as.matrix() 

## add row names
y_row_names <- movie_ratings_by_two_users[,1]
y_col_names <- colnames(movie_ratings_by_two_users)
colnames(movie_ratings_by_two_users) <- y_col_names

saveRDS(movie_ratings_by_two_users, "rda/user_bias_table1.rda")
saveRDS(user_bias_chart1, "rda/user_bias_chart1.rda")

user_bias_chart1
dev.new()
movie_ratings_by_two_users

rm(y_col_names, y_row_names)


# illustrating variations in movie ratings.
#
# case 2: user bias -  some users tend to give the same rating for 
#                      every movie regardless

# we start by calculating standard deviations for each user's ratings
user_rating_sds <- edx %>% select(userId, movieId, rating) %>% group_by(userId) %>% 
  summarise(n=n(), sd=sd(rating)) %>% arrange(desc(sd))

# see users and the standard deviation in their ratings
user_rating_sds %>% select(userId, sd) %>% arrange((sd)) %>% slice(1:1000) %>% view()

# now lets plot a sample of the standard deviations
Sys.time()
set.seed(1, sample.kind = "Rounding") 
sd_sample_index <- sample(nrow(user_rating_sds), 50000)
sd_sample <- user_rating_sds[sd_sample_index, ]

sd_sample <- sd_sample %>% 
  mutate(level_of_deviation= if_else(sd==0, "no_variation", 
                                     if_else(sd<=0.5, "low_variation", 
                                             if_else(sd<=1.5, "medium_variation", "high_variation"))))


ratings_sd_chart <- sd_sample %>% ggplot(aes(x=factor(userId), y=sd)) + 
  geom_point(aes(color=level_of_deviation)) + 
  labs(title = "Levels of deviation (standard deviations for user ratings)") +
  ylab("standard deviation \n (in user ratings)") + xlab("users") +
  theme(axis.text.x=element_blank())
Sys.time()

ratings_sd_chart

saveRDS(ratings_sd_chart, "rda/ratings_sd_chart.rda")
rm(sd_sample, ratings_sd_chart, sd_sample_index)

# some selected sample user and movies for illustration
user_1 <- 1
user_24176 <- 24176
user_18965 <- 18965

#selected_users <- c(1, 1686)
#edx %>% filter(userId %in% selected_users) %>% select(rating) %>% view()

# all these users have a standard deviation of 0, meaning they have no 
# variation in their rating (same rating for every movie)
zero_sds <- user_rating_sds %>% filter(sd==0) %>% pull(userId)

# find all users with zero standard deviations and how many movies they have rated
zero_sds <- edx %>% filter(userId %in% zero_sds) %>% group_by(userId) %>% 
  summarise(favorite_rating=unique(rating), num_ratings=n())

# save the result for the report
saveRDS(zero_sds, "rda/users_with_zero_sds.rda")

# now let's see what rating they gave: user 1 rates everything a 5 star
edx %>% filter(userId==user_1) %>% select(userId, movieId, rating)

# user 24176 hates every movie; rates everything 1 star
edx %>% filter(userId==user_24176) %>% distinct(rating)

# again for this user (18965), every movie is a 5 star
edx %>% filter(userId==user_18965) %>% distinct(rating)

rm(zero_sds, user_rating_sds)

###############################################################################
#
# checkpoint #: checkpoint #3 - data visualization and exploration completed
#
###############################################################################


###############################################################################
#
# modeling - approach
#
# 1. from the edx dataset, set aside 10% for testing during the training phase
# 2. the remaining 90% will be used in cross validation
# 3. on the remaining 90%, use ten-fold cross validation
#    to create train/test sets 
# 4. on each cv train/test, and calculate the biases, calculate rmse across 
#    the given set of lambda values
# 5.  find the best lambda from each cv from step 4
# 6. average all the lambdas to best final best_lambda
# 7. apply this best_lambda to our test sample from step 1
#
# fitting phase:
# 8. finally compute all the biases on the entire edx dataset
# 9. use the best lambda from step 6, ad apply to final edx validation set
#
###############################################################################



###############################################################################
#
# data partitioning
#
# from the edx dataset, set aside 10% for testing during the training phase
#
# the remaining 90% will be used is cross validation
#
###############################################################################

#glimpse(edx)
# partition edx into train and test

# test set will be 10% of our data
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in training set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into training set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)


# remove edx and validation for now
rm(test_index, temp, removed)

print(paste(Sys.time(), " - number of rows in train and test set" ))
nrow(train_set)
nrow(test_set)

saveRDS(train_set, "rda/train_set.rda")
saveRDS(test_set, "rda/test_set.rda")


###############################################################################
# 
# checkpoint# 3 - modeling - data partitioning
#
# next checkpoint : checkpoint#4 - build base model
###############################################################################


# remove the main edx and validation sets for now
rm(edx, validation)


train_set <- train_set %>% select(userId, movieId, rating)
test_set <- test_set  %>% select(userId, movieId, rating)

#############################################################################
# create the base model
#############################################################################
#
# My model will be based on the linear model described in the coursework 
#
# Irizarry, Rafael A., (2021), Introduction to Data Science Data Analysis and 
# Prediction Algorithms with R, Chapter 34, Section 34.7 - Recommendation 
# systems.
#
#  - https://rafalab.github.io/dsbook/large-datasets.html#recommendation-systems/. 
#                                                                                                                                                                                                                                      This model trains an algorithm using user and movie effects (biases)
# the model is expressed as follows:
#
# y = mu + bu + bi + e
#
#  where: mu is the overall mean
#         bu is the user effect (user bias)
#         bi is the movie effect (bias introduced by the movie)
#         e is the irreducible error
#
################################################################

paste("training started: ", Sys.time())

# perform cross validation.
#
# since the edx dataset is so large we can't use the standard cross validation,
# it won't fit in memory, so here I do it manually.

# 1. create 10 sets of training/testing - manual 10-fold cross validation sets
CV.FOLDS   <- 10
indexes <- lapply(1:CV.FOLDS, FUN=function(x){fnCreateIndexes(train_set)})

# 2. create a list of lambdas to find best lambda from each cross validation train/test
lambdas <- seq(1, 10, 0.1)

# 3. use sapply to execute a function that performs the entire train/test scenario
(cv_rmse_list <- lapply(indexes, fnTrainAndTest, dataset=train_set, lambdas=lambdas))
saveRDS(cv_rmse_list, "rda/cv_rmse_list_10_fold.rda")

# we should have best lambda from each of the ten folds, average them
# this is our best lambda
print(paste(Sys.time(), " - best 10 lambdas..."))
(best_10_lambdas <- sapply(cv_rmse_list, which.min) %>% 
    sapply(FUN = function(x){lambdas[x]}))

# save this info for reporting
saveRDS(best_10_lambdas, "rda/best_10_lambdas.rda")

print(paste(Sys.time(), " - avg of best 10 lambdas..."))
(average_of_best_lambdas <- best_10_lambdas %>% mean())

# now let's visualize the rmses from the 10 cross validation runs
cv_rmse_list <- lapply(1:CV.FOLDS, FUN=function(x)
{
  data.frame(lambda=lambdas,
             rmse=cv_rmse_list[[x]],
             iteration=factor(rep(x, length(cv_rmse_list[[x]]))))
}) %>% bind_rows()

(cv_rmse_list_chart <- cv_rmse_list %>%  ggplot() +
    geom_point(aes(x=lambda, y=rmse, colour=iteration), size=2) +
    scale_color_brewer(palette = "Paired") + theme_light())

# save the object for use later in report
saveRDS(cv_rmse_list_chart, "rda/cv_rmse_list_chart.rda")

# lets see how the model performs on the test set
print (paste(Sys.time(), " - started fitting model on entire training set..."))

# 1. calculate the overall mean for the training set
mu <- mean(train_set$rating)

# 2. regularize the movie bias
movie_effects <- train_set %>% group_by(movieId) %>%
  summarise(b_i = sum( rating - mu) / (n()+average_of_best_lambdas))

# 3. regularize the user bias
user_effects <- train_set %>%
  left_join(movie_effects, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i) / (n()+average_of_best_lambdas))

test_set <- test_set %>%
  left_join(movie_effects, by="movieId") %>%
  left_join(user_effects, by="userId") %>%
  mutate(predicted  = mu + b_i + b_u)

print(paste(Sys.time(), " - model prediction on test " ))

model_rmses <- data.frame(description="Best lambda", 
                           value=average_of_best_lambdas)

(model_rmses <- rbind(model_rmses, 
                      data.frame(description="RMSE on test set", 
                                 value=RMSE(test_set$rating, test_set$predicted))))

saveRDS(model_rmses, "rda/model_rmses.rda")
print (paste(Sys.time(), " - done fitting model on entire training set."))


saveRDS(model_rmses, "rda/model_rmses.rda")

##############################################################
# final model:
# fitting phase - fit a model on the entire edx data
# this computes all the biases using the optimized lambda
# which will be applied to the final validation set
##############################################################

# load the edx dataset, load the clean one 
edx <- readRDS("rda/edx_v1.rda") %>% select(userId, movieId, rating)

print (paste(Sys.time(), " - started fitting phase..."))
# 1. calculate the over mean for the training set
mu <- mean(edx$rating)

# 2. regularize the movie bias
movie_effects <- edx %>% group_by(movieId) %>%
  summarise(b_i = sum( rating - mu) / (n()+average_of_best_lambdas))

# 3. regularize the user bias
user_effects <- edx %>%
  left_join(movie_effects, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - mu - b_i) / (n()+average_of_best_lambdas))

print (paste(Sys.time(), " - done final fitting phase."))
rm(edx)

##############################################################
# final predictions
##############################################################

print (paste(Sys.time(), " - start final predictions..."))
validation <- readRDS("rda/validation.rda") %>% select(userId, movieId, rating)

# predictions on validation
validation <- validation %>%
  left_join(movie_effects, by="movieId") %>%
  left_join(user_effects, by="userId") %>%
  mutate(predicted  = mu + b_i + b_u)


(model_rmses <- rbind(model_rmses, 
                      data.frame(description="RMSE on validation set", 
                                 value=RMSE(validation$rating, validation$predicted))))


saveRDS(model_rmses, "rda/model_rmses.rda")

saveRDS(user_effects, "rda/final_user_effects.rda")
saveRDS(movie_effects, "rda/final_movie_effects.rda")

print (paste(Sys.time(), " - done final predictions."))

