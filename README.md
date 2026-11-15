# house_predictor 🏠

How It Works (Step by Step)

1. Input: The user gives 4 simple details: City, BHK, Bathrooms and Super Area (sqft).

2. Preprocessing (inside the pipeline):

One-Hot Encoding converts the city name into numbers. For example, “Pune” becomes a set of 0/1 columns, because a model can’t work with text.
Standard Scaling brings all values to a similar range, so a large number like area (1000+) doesn’t overpower a small one like BHK (1 to 4).

3. Prediction: The Ridge Regression model has learned from 12,000 houses how each feature affects price. It combines them into a weighted formula (roughly: price = city effect + BHK effect + bathroom effect + area effect) and outputs a predicted price in INR.

4. Why Ridge? Ridge adds a penalty on large coefficients. This keeps the model stable and stops it from over-relying on any single feature, especially when features like BHK, bathrooms and area are correlated (bigger houses usually have more rooms).
