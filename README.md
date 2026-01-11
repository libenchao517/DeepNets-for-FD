# DeepNets-for-FD
DeepNets-for-FD软件包中有一些故障诊断任务中常用的CNN网络。下面是这些方法的使用案例和参考文献。

## 使用案例
- CNN-2560-768 [1]
```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from DeepNet import CNN_2560_768
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
images, _, target, _ = train_test_split(images, target, train_size=1000)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = CNN_2560_768(sample_height=28, sample_weight=28, num_classes=10, epoch=5)
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

- TDCNN gcForest [2]
```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from DeepNet import TDCNN_GCFOREST
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
images, _, target, _ = train_test_split(images, target, train_size=1000)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = TDCNN_GCFOREST( sample_height=28, sample_weight=28, num_classes=10, epoch=5)
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

- LeNet 5
```
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from DeepNet import LeNet_5
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
images, _, target, _ = train_test_split(images, target, train_size=1000)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = LeNet_5(sample_height=28, sample_weight=28, num_classes=10, epoch=5)
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

- LiNet [3]
```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from DeepNet import LiNet
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
images, _, target, _ = train_test_split(images, target, train_size=1000)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = LiNet(sample_height=28, sample_weight=28, num_classes=10, epoch=5)
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

- 1DCNN Softmax [3]
```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from DeepNet import One_Dcnn_Softmax
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
images, _, target, _ = train_test_split(images, target, train_size=1000)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = One_Dcnn_Softmax( sample_height=1, sample_weight=784, num_classes=10, epoch=5)
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

- TICNN [4]
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from DeepNet import TICNN
images, target = make_classification(n_samples=2000, n_features=1024,  n_informative=512, n_redundant=256, n_classes=10, n_clusters_per_class=1, random_state=42)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = TICNN(sample_height=1, sample_weight=1024, num_classes=10, epoch=5)
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

- WDCNN [5]
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from DeepNet import WDCNN
images, target = make_classification(n_samples=2000, n_features=1024,  n_informative=512, n_redundant=256, n_classes=10, n_clusters_per_class=1, random_state=42)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = WDCNN(sample_height=1, sample_weight=1024, num_classes=10, epoch=5)
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

- gcForest
```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from DeepNet import GC_Forest
images, target = make_classification(n_samples=2000, n_features=1024,  n_informative=512, n_redundant=256, n_classes=10, n_clusters_per_class=1, random_state=42)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = GC_Forest(sample_height=32, sample_weight=32, num_classes=10, epoch=5)
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

- MA 1DCNN [6]
```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from DeepNet import MA_1DCNN
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
images, _, target, _ = train_test_split(images, target, train_size=1000)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = MA_1DCNN(sample_height=1, sample_weight=784, num_classes=10, epoch=5, device="cpu")
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

- MIX CNN [7]
```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from DeepNet import MIX_CNN
mnist = fetch_openml('mnist_784', parser='auto', version=1)
images, target = mnist["data"].to_numpy(), mnist["target"].to_numpy()
images, _, target, _ = train_test_split(images, target, train_size=1000)
x_train, x_test, t_train, t_test = train_test_split(images, target, train_size=0.5)
model = MIX_CNN(sample_height=1, sample_weight=784, num_classes=10, epoch=5, device="cpu")
y_pred = model.fit_transform(x_train, x_test, t_train, t_test)
```

## 参考文献
```
[1] Wen L, Li X, Gao L, et al. A new convolutional neural network-based data-driven fault diagnosis method[J]. IEEE transactions on industrial electronics, 2017, 65(7): 5990-5998.
[2] Xu Y, Li Z, Wang S, et al. A hybrid deep-learning model for fault diagnosis of rolling bearings[J]. Measurement, 2021, 169: 108502. 
[3] Jin T, Yan C, Chen C, et al. Light neural network with fewer parameters based on CNN for fault diagnosis of rotating machinery[J]. Measurement, 2021, 181: 109639.
[4] Zhang W, Li C, Peng G, et al. A deep convolutional neural network with new training methods for bearing fault diagnosis under noisy environment and different working load[J]. Mechanical systems and signal processing, 2018, 100: 439-453.
[5] Zhang W, Peng G, Li C, et al. A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals[J]. Sensors, 2017, 17(2): 425.
[6] Wang H, Liu Z, Peng D, et al. Understanding and learning discriminant features based on multiattention 1DCNN for wheelset bearing fault diagnosis[J]. IEEE Transactions on Industrial Informatics, 2019, 16(9): 5735-5745.
[7] Zhao Z, Jiao Y. A fault diagnosis method for rotating machinery based on CNN with mixed information [J]. IEEE Transactions on Industrial Informatics, 2022, 19(8): 9091-9101.
```
