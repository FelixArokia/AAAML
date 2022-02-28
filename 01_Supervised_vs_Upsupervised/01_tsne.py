import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn import manifold

data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, cache=True)

pixel_values, targets = data
targets = targets.astype(int)

pixel_values = np.array(pixel_values)

print(f'pixel_values shape : {pixel_values.shape}')
print(f'pixel_values type : {type(pixel_values)}')
single_image = pixel_values[1, :]
print(f'single_image shape: {single_image.shape}')
print(f'single image type: {type(single_image)}')

single_image = single_image.reshape(28, 28)
print(f'single image after reshape: {single_image.shape}')
plt.imshow(single_image)
plt.show()

tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_values[:3000, ])
print(f'type of transformed data : {type(transformed_data)}')

transformed_df = pd.DataFrame(transformed_data)
# OR
transformed_df = pd.DataFrame(np.column_stack((transformed_data, targets[:3000])))

transformed_df.columns = ['Dim1', 'Dim2', 'Target']

print(transformed_df.head())

grid = sns.FacetGrid(transformed_df, hue="Target", size=8)
grid.map(plt.scatter, "Dim1", "Dim2").add_legend()

plt.show()