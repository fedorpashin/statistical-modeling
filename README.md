# statistical-modeling

**statistical-modeling** is a Python package that provides two main features:

- Statistical analysis of samples
- Working with distributions

## Example

```python
import statistical_modeling as sm

from matplotlib import pyplot as plt

s = sm.BinomialDistribution(10, 0.5).sample(100, sm.BinomialDistribution.Algorithm.cumulative)

print(f"Mean = {s.Mean}, Variance = {s.Variance}")

fig, ax = plt.subplots()
s.plot_pdf(ax)
fig.show()
```

