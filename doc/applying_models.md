# Applying models



The utboost model can be exported as python code, and you can apply the model.

## Python

Apply the model in Python format. The method is available within the output Python file with the model description.

1. Export the trained model as python code.

   ```python
   # fit the model first
   model.fit(...)
   # export model to python code
   model.to_python('model.py')
   ```

2. Using this python file to apply the model.

   ```python
   # define feature vector
   features = [...]
   # predict
   predict_outcomes = apply_model(features)
   ```



