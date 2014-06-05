# Practical Machine Learning, June 2014 - Peer Assessed Assignment
==================================================================

## Introduction
The goal of this assessment is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the way they performed barbell lifts - correctly and incorrectly in 5 different ways.

## The data
The data file is downloaded from the Coursera website and loaded into R:





```r
train = read.csv("pml-training.csv")
```


It is seen that this data set contains 19622 observations of 160 variables. The variable to be predicted, classse, is a qualitative variable having the values A, B, C, D or E.

A lot of the other variables are the above-mentioned readings from the accelerometers and these are kept for the classification.

There are many variables that ought to contain calculated quantitatives from these readings, but as a rule these are empty, NA's and occasionally a number. These variables, typically having a name starting with 'max', 'min', 'kutrosis', 'stdev' etc, are deleted.

Furthermore, the variables related to the timestamp and the user are deleted, at they cannot provide any information about the movement, and since some of these are unique, we risk overfitting by using them. The same is true of the variable X, that is a primary key for the observations.

After deleting these variable, and keeping only the actual readings of the accelerometers and the classe, we see that none of these variables contains NA's, so imputation is not necessary.






```r
train$X = NULL
train$raw_timestamp_part_1 = NULL
train$raw_timestamp_part_2 = NULL
train$cvtd_timestamp = NULL
train$new_window = NULL
train$num_window = NULL
train$kurtosis_roll_belt = NULL
train$kurtosis_picth_belt = NULL
train$kurtosis_yaw_belt = NULL
train$skewness_roll_belt = NULL
train$skewness_roll_belt.1 = NULL
train$skewness_yaw_belt = NULL
train$max_roll_belt = NULL
train$max_picth_belt = NULL
train$max_yaw_belt = NULL
train$min_roll_belt = NULL
train$min_pitch_belt = NULL
train$min_yaw_belt = NULL
train$amplitude_roll_belt = NULL
train$amplitude_pitch_belt = NULL
train$amplitude_yaw_belt = NULL
train$var_total_accel_belt = NULL
train$avg_roll_belt = NULL
train$stddev_roll_belt = NULL
train$var_roll_belt = NULL
train$avg_pitch_belt = NULL
train$stddev_pitch_belt = NULL
train$var_pitch_belt = NULL
train$avg_yaw_belt = NULL
train$stddev_yaw_belt = NULL
train$var_yaw_belt = NULL
train$var_accel_arm = NULL
train$avg_roll_arm = NULL
train$stddev_roll_arm = NULL
train$var_roll_arm = NULL
train$avg_pitch_arm = NULL
train$stddev_pitch_arm = NULL
train$var_pitch_arm = NULL
train$avg_yaw_arm = NULL
train$stddev_yaw_arm = NULL
train$var_yaw_arm = NULL
train$kurtosis_roll_arm = NULL
train$kurtosis_picth_arm = NULL
train$kurtosis_yaw_arm = NULL
train$skewness_roll_arm = NULL
train$skewness_pitch_arm = NULL
train$skewness_yaw_arm = NULL
train$max_roll_arm = NULL
train$max_picth_arm = NULL
train$max_yaw_arm = NULL
train$min_roll_arm = NULL
train$min_pitch_arm = NULL
train$min_yaw_arm = NULL
train$amplitude_roll_arm = NULL
train$amplitude_pitch_arm = NULL
train$amplitude_yaw_arm = NULL
train$kurtosis_roll_dumbbell = NULL
train$kurtosis_picth_dumbbell = NULL
train$kurtosis_roll_dumbbell = NULL
train$kurtosis_picth_dumbbell = NULL
train$kurtosis_yaw_dumbbell = NULL
train$skewness_roll_dumbbell = NULL
train$skewness_pitch_dumbbell = NULL
train$skewness_yaw_dumbbell = NULL
train$max_roll_dumbbell = NULL
train$max_picth_dumbbell = NULL
train$max_yaw_dumbbell = NULL
train$min_roll_dumbbell = NULL
train$min_pitch_dumbbell = NULL
train$min_yaw_dumbbell = NULL
train$amplitude_roll_dumbbell = NULL
train$amplitude_pitch_dumbbell = NULL
train$amplitude_yaw_dumbbell = NULL
train$total_accel_dumbbell = NULL
train$var_accel_dumbbell = NULL
train$avg_roll_dumbbell = NULL
train$stddev_roll_dumbbell = NULL
train$var_roll_dumbbell = NULL
train$avg_pitch_dumbbell = NULL
train$stddev_pitch_dumbbell = NULL
train$var_pitch_dumbbell = NULL
train$avg_yaw_dumbbell = NULL
train$stddev_yaw_dumbbell = NULL
train$var_yaw_dumbbell = NULL
train$kurtosis_roll_forearm = NULL
train$kurtosis_picth_forearm = NULL
train$kurtosis_yaw_forearm = NULL
train$skewness_roll_forearm = NULL
train$skewness_pitch_forearm = NULL
train$skewness_yaw_forearm = NULL
train$max_roll_forearm = NULL
train$max_picth_forearm = NULL
train$max_yaw_forearm = NULL
train$min_roll_forearm = NULL
train$min_pitch_forearm = NULL
train$min_yaw_forearm = NULL
train$amplitude_roll_forearm = NULL
train$amplitude_pitch_forearm = NULL
train$amplitude_yaw_forearm = NULL
train$var_accel_forearm = NULL
train$total_accel_forearm = NULL
train$var_accel_forearm = NULL
train$avg_roll_forearm = NULL
train$stddev_roll_forearm = NULL
train$var_roll_forearm = NULL
train$avg_pitch_forearm = NULL
train$stddev_pitch_forearm = NULL
train$var_pitch_forearm = NULL
train$avg_yaw_forearm = NULL
train$stddev_yaw_forearm = NULL
train$var_yaw_forearm = NULL
train$user_name = NULL
```


## The prediction algorithm

After some experimentation we find that the best algorithm to use is a random forest. The algorithms tried are logistic regression, simple regression trees, random forests, and the gbm-boosting algorithm. For each of these algorithm we conduct a simple procedure consisting of making a model using the entire training set and observing the accuracy of the model on this training set. The algorithm with the highest accuracy is random forest, which therefore is used in the following - and the accuracy is around 99%.

In order to optimize the random forest algorithm and the sole parameter in this one, mtry, which is the number of randomly chosen variables considered for each split, and to get better estimates of the accuracy using 10-fold cross validation we use the following command in R:


```r
rfModel = train(classe ~ ., data = train, method = "rf", prox = FALSE, trControl = trainControl(method = "cv", 
    number = 10), tuneGrid = expand.grid(mtry = c(2, 5, 10, 15, 20)))
```


The result is


```r
rfModel
```

```
## Random Forest 
## 
## 19622 samples
##    50 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 17661, 17659, 17659, 17659, 17660, 17660, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.002        0.002   
##   5     1         1      0.001        0.002   
##   10    1         1      0.001        0.002   
##   20    1         1      9e-04        0.001   
##   20    1         1      0.001        0.002   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 10.
```


Firstly, the result above only shows what happens for certain values of mtry. Further experimentation (which is time- and space-consuming) shows that the value of mtry is ot that critical, in that we get accuracies of around 99% for a whole range of values of mtry.

The algorithm chooses mtry = 10.

We can also see, that the accuracy is estimated to be 99.7%  with a standard deviation of 0.122%. As the out of sample error is 1 minus the accuracy, we see that the out of sample error is 0.3% with the same standard deviation  0.122%.

## The test set
For the record we have that the model gave all predictions correct on the provided test set. 




