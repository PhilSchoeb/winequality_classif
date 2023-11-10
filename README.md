Classification of wine quality using a two different parzen_window models:
hard_parzen (harsh neighborhood) and a soft_parzen with gaussian kernel.

Philippe Schoeb

October 12th 2023

The data is generated from a file named winequality.txt (data for this problem can be found online).
I separate the data in three sets: training, validation and test set.

Then, I optimise the two hyperparameters h for hard_parzen and sigma for the gaussian kernel by using the validation set.
After that, I test the best two parameters with the test set. 
This program does not output anything. The best hyperparameters are printed with their respective error rate.

Details about hard_parzen as i do not find any information about it besides the class I am taking:
It is a parzen_window model with kernel k: 
k(x_i, x) = IndicatorFunction_[0, h] (distance(x_i, x) where x is the point we want to classify. 
So basically, k(x_i, x) = 1 if distance(x_i, x) <= h and k(x_i, x) = 0 if distance(x_i, x) > h.




 
