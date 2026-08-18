import src.main as main
import matplotlib.pyplot as plt

# print(main.housing_full.head())
# print(main.housing_full["ocean_proximity"].value_counts())
# print(main.housing_full.describe())

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

main.housing_full.hist(bins=50, figsize=(12, 8))

plt.show()