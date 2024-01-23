# UTBoost

UTBoost is a powerful uplift modeling library based on boosting framework over decision trees.
It can handle large-scale RCT (randomized controlled trial) datasets and demonstrates superior predictive performance.

## Documentations

- [Installation Instructions](./doc/install.md)
- [Applying Model](./doc/applying_models.md)

## Quick Start

See the **[tutorial notebook](./examples/tutorial.ipynb)** for details.

```python
# import approaches
from utboost import UTBClassifier, UTBRegressor

# define model (CausalGBM algorithm)
model = UTBClassifier(
    ensemble_type='boosting',
    criterion='gbm',
    iterations=20,
    max_depth=4
)

# fit model
model.fit(X=X_train, ti=ti_train, y=y_train)

# predict outcomes
preds = model.predict(X_test)
# predict uplift
uplift_preds = preds[:, 1] - preds[:, 0]

```

## File Locations

 * `src/*` — C++ code that ultimately compiles into a library
 * `include/` — C++ header files
 * `python-package/` — python package
 
## License

This project is open-sourced under the MIT license. You can find the terms of the license [here](http://opensource.org/licenses/MIT).

## Reference Paper

Junjie Gao, Xiangyu Zheng, DongDong Wang, Zhixiang Huang, Bangqi Zheng, Kai Yang. "[UTBoost: A Tree-boosting based System for Uplift Modeling](https://arxiv.org/abs/2312.02573)".