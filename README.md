
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

<br/>  
  
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/raw_data_movies_record_structure.PNG?raw=true)
  
<br/>  

![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/raw_data_movies.PNG?raw=true)

**ratings.dat**   
  The next file in the downloaded zip is the __ratings.dat__. This file contains each rating a user has given a movie. Again this is a delimited file with :: as the separator. Each line contains userId, the movieId, the rating that was provided, and the timestamp.   


![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/raw_data_ratings_record_structure.PNG?raw=true)
  
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/raw_data_ratings.PNG?raw=true)

## Data Wrangling
In the dataset, each rating is an observation, therefore the data in the _ratings.dat_ is joined with the dataset in the _movies.dat_. After some data wrangling, the resulting data looks as below. All features are converted to the appropriate types. For instance, the movieId and userId must be converted from character to numeric. The movie title and genres by default might be loaded as factors; they are converted to characters. The following table shows a sample of the data after initial wrangling.      

![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/sample_data.PNG?raw=true)


![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/sample_data-table.PNG?raw=true)


## Data Exploratory
This section presents some of the data exploratory performed in the project to gain understanding of the general properties of the data.

Before exploring, I ask a few questions that I attempt to answer using the data. 

* For instance, what are the patterns in the user rating? 
* What are the top most rated movies?
* Who are the most active users in terms of number of rating?
* Do some people just give 5-star rating to every movie and others just rating every movie poorly?
* What is the overall rating average for all movies? What are the average rating per movie and per user?

The following table shows the total number of records, the total number of users, number of movies in both the training and validation datasets.  
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/data-exploratory-1.PNG?raw=true)

### Null Values
A simple query to the training dataframe shows that there are no null (NA) values in either the edx or validation datasets. This is good news since we don't need to exclude any invalid data.
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/data-exploratory-null-values.PNG?raw=true)

### Overall Mean, Median
Using the unique() and the summary() functions, we see that overall there are ten unique ratings, given by users with a minimum of ```r min(edx$rating) ``` and max of ```r max(edx$rating) ```. No rating of zero (0) is given. The overall mean for movie rating is ```r round(mean(edx$rating),2) ```, with a median of ```r median(edx$rating) ```. This means most users seem to be generous and give pretty high ratings with __4.0__ being the most predominant rating.

![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/data-exploratory-mean-median.PNG?raw=true)


### User ratings distribution
The movie rating distribution can be visualized using a simple histogram.   
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/rating-distribution-code.PNG?raw=true)
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/rating-distribution.PNG?raw=true)



### Variations in users' ratings and behavior 
We know from experience that users exhibit different behaviors when it comes to rating items, whether it's movies, items purchased or whatever. For some users, ratings movies may be habitual, meaning they rate almost every movie they watch, others rate only those movies they like, while others only rate movies they really hated just to warn other people. Of course, there are some users who don't rate at all. 

Among users who make the effort to rate movies, some are very generous and give every movie a 5 star rating, others may give a poor or average rating to every movie. These differences are called biases; the exact reasons for these biases are not known, buts it's important to be aware of them when making predictions about how users may rate future movies. 


### Most Active Users vs Least Active Users
I have selected top ten most active users and ten least active users by number of rating and shown in the table. From the table, we see a huge difference in the number of movies rated between the two groups with most active users in thousands while least active users in tens. There is also a difference in average ratings with some of the least active users having even higher averages than active users. This is important to note because users with few ratings may be outliers and can affect our predictions. 
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/most-active-least-active-users.PNG?raw=true)

### Most rated movies
To get an idea of most rated movies based on average rating, I exclude movies that have received less than 50 ratings. This provides a better view of what users think are the best movies. The table below lists the top 10 movies by average ratings. 
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/most-rated-movies.PNG?raw=true)

### Some users give higher ratings than others 
As an illustration, I selected four random movies, two users who have watched all four movies, and observed their rating. See the table and table and chart below. From the figures, user __18__ gives higher ratings to all four movies than user __8__.
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/some-users-rate-higher-than-others.PNG?raw=true)

### Movie Genres
Movie genres can also influence movie ratings. Some people like Action movies while others like Comedy. As mentioned in the Data Wrangling section, the genres have been broken down into columns representing the individual genres. The following table shows the frequency of each genre in the edx dataset. 
  
  
From the table and the graph below, it can be seen that Drama and Comedy are the most frequent genres in the dataset than any other genre. Genres such as Western, Film Noir, and Documentary are less common.   
<br/>
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/genre-distribution.PNG?raw=true)


## Machine Learning - Model, Approach, and Analysis
he model used in this analysis will be based on the linear model described in the course work^[Irizarry, Rafael A., (2021), Introduction to Data Science Data Analysis and Prediction Algorithms with R, Chapter 34, Section 34.7 - Recommendation systems. https://rafalab.github.io/dsbook/large-datasets.html#recommendation-systems/]. This model trains an algorithm using user and movie effects (biases). 


The main steps are outlined below and also shown the subsequent figure following the steps.

0. the data is first partitioned into edx and validation sets
1. from the edx dataset, set aside 10% for testing during the training phase
2. the remaining 90% will be used in cross validation; on the remaining 90%, use 10-fold cross validation to create train/test sets (cross validation train/test pairs)
3. on each cv train/test, calculate the biases, calculate rmse across the given set of lambda values
4. find the _"best"_ lambda from each cv from step 3
5. average all the _"best"_ lambdas to obtain best final _best_lambda_
6. fit a final model by computing all the biases on the entire edx dataset using _best_lambda_ from step 5
7. use the final model from step 6, and apply it to final validation set to make final predictions
![](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/images/modeling_approach-orig.PNG?raw=true)

[[view entire report...]](https://github.com/kowusu01/KOwusu.Tieku.HarvardX.Capstone.Movielens/blob/main/report.pdf)
