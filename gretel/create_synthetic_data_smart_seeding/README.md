# Create Synthetic Data From Partial Rows

This blueprint shows you how to utilize partial values from your training data to create
a new synthetic dataset. The method in this blueprint achieves the following:

- creates a dataset that is the same shape as the training data
- utilizes partial row values from the training data

When building the model, you specifcy one or more "seed columns."  When generating data,
for each row in the data, the original values for those columns are extracted and 
used as input to create a new record.

Essentially, you specify which columns to preserve, and Gretel synthesizes the rest
of the row data for you.