
## Movie Recommendation with R  - a Machine Learning Task
## Predicting Movie Ratings using Biases

### By Kwaku Owusu-Tieku
Please submit comments to engineer.zkot2@gmail.com

  
[[view entire report...]](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/report.pdf)
  

## Introduction

Recommendation systems are used by companies such as Netflix, Amazon, and other large online retailers to suggest items for users based on some parameters. For instance, items can be suggested to users based on popularity.  

In the case of movie recommendation, we could assume that a user might like a movie simply because many people like that movie.  In other cases we could analyze user profiles for attributes such as gender, age group, etc. to find similarities among users and recommend items based on their similarities. For instance, we might recommend Horror movies to males if our analysis reveal that males tend to watch more Horror movies than females. 

In our dataset, we have a selected number of users and a set of movies. Not all users have watched or rated every movie. In fact, no single user has rated every movie. The challenge here is to assign (predict) ratings to movies that a user has not rated. The assumption here is that a higher rating means the user liked the movie. Therefore, if our prediction assigns a high rating to a movie for a user, we take that to mean the user will most likely like that movie, hence, we can recommend that movie to the user.

This project presents a movie recommendation system that can be used for predicting movie ratings.

## The Dataset
The dataset used in this project is the ml-10m.zip containing 10million records. This dataset can be found at http://files.grouplens.org/datasets/movielens/ml-10m.zip.
As per standard data science practice, the data has been partitioned into training (the edx set), and hold-out (validation set). All training and tuning will be peformed on the edx set.

## Quick Glance at the Dataset
To begin the analysis, the movielens data is downloaded and unzipped. There are two important files in this download - movies.dat and ratings.dat. These files can be viewed using any standard text viewer.

**movies.dat**   
  A quick peek at the __movies.dat__ shows that the file is a delimeted with double colon (::) as the separator. Each line is unique movie having the movieId, the title, and the list of genres for that movie. Notice that the genre is also a delimeted string with | as the separator.   

![movies dataset structure](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/raw_data_movies_record_structure.PNG?raw=true)

**ratings.dat**   
  The next file in the downloaded zip is the __ratings.dat__. This file contains each rating a user has given a movie. Again this is a delimited file with :: as the separator. Each line contains userId, the movieId, the rating that was provided, and the timestamp.   
\newpage

![data dataset structure](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/raw_data_ratings_record_structure.PNG?raw=true)

## Data Wrangling
In the dataset, each rating is an observation, therefore the data in the _ratings.dat_ is joined with the dataset in the _movies.dat_. After some data wrangling, the resulting data looks as below. All features are converted to the appropriate types. For instance, the movieId and userId must be converted from character to numeric. The movie title and genres by default might be loaded as factors; they are converted to characters. The following table shows a sample of the data after initial wrangling.      

![sample movie data](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/sample_data.PNG?raw=true)


![sample movie data](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/sample_data-table.PNG?raw=true)


## Data Exploratory
This section presents some of the data exploratory performed in the project to gain understanding of the general properties of the data.

Before exploring, I ask a few questions that I attempt to answer using the data. 

* For instance, what are the patterns in the user rating? 
* What are the top most rated movies?
* Who are the most active users in terms of number of rating?
* Do some people just give 5-star rating to every movie and others just rating every movie poorly?
* What is the overall rating average for all movies? What are the average rating per movie and per user?

The following table shows the total number of records, the total number of users, number of movies in both the training and validation datasets.  
![sample movie data](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/data-exploratory-1.PNG?raw=true)


### Null Values
A simple query to the training dataframe shows that there are no null (NA) values in either the edx or validation datasets. This is good news since we don't need to exclude any invalid data.
![sample movie data](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/data-exploratory-null-values.PNG?raw=true)

### Overall Mean, Median
Using the unique() and the summary() functions, we see that overall there are ten unique ratings, given by users with a minimum of ```r min(edx$rating) ``` and max of ```r max(edx$rating) ```. No rating of zero (0) is given. The overall mean for movie rating is ```r round(mean(edx$rating),2) ```, with a median of ```r median(edx$rating) ```. This means most users seem to be generous and give pretty high ratings with __4.0__ being the most predominant rating.

![sample movie data](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/data-exploratory-mean-median.PNG?raw=true)
