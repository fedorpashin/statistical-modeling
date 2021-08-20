# statistical-modeling


[![EO principles respected here](https://www.elegantobjects.org/badge.svg)](https://www.elegantobjects.org)
[![Managed By Self XDSD](https://self-xdsd.com/b/mbself.svg)](https://self-xdsd.com/p/fedorpashin/statistical-modeling?provider=github)


**statistical-modeling** is a Python package that provides two main features:

- Statistical analysis of samples
- Working with distributions

## Example

```python
import statistical_modeling as sm

from matplotlib import pyplot as plt

print(sm.RandomInt(sm.geometric.Distribution(0.1)))

s = sm.RandomSample(
    100,
    sm.binomial.Distribution(10, 0.5),
    sm.binomial.CumulativeAlgorithm()
)

print(
    f"Mean = {sm.Mean(s)},\n"
    f"Variance = {sm.Variance(s)}"
)

fig, ax = plt.subplots()
sm.plot_pdf(s, ax)
fig.show()
```
