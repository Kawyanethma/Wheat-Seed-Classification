# Introduction
## _Brief Description of the Dataset_
Dataset contains 210 soft X-ray images of wheat kernels (varieties: Kama, Rosa,Canadian) from experimental fields in Lublin, Poland.<br/>
- Dataset can be found at: [Seeds - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/236/seeds)<br/>

To construct the data, seven geometric parameters of wheat kernels were measured in this dataset:
1. Area ( A )
2. Perimeter ( P )
3. Compactness ( $C = 4πA/P^2$ )
4. Length of kernel
5. Width of kernel
6. Asymmetry coefficient
7. Length of kernel groove

# Exploratory Data Analysis
## Data Understanding
- Reporting the data types, number of instances, data quality issues and any missing values.
- Features and target variable(s) in the dataset

## Data Preprocessing
- Checking for the duplicates, missing values calculation, scaling, encoding categorical variables
- Data Standardization: This process centers the feature around zero and scales it to have a unit variance.
- Data Normalization: Data normalization is a similar technique to standardization, but instead of using the mean and standard deviation, it scales the features to a specific range, usually [0, 1] or [-1, 1].

## Data Visualization
- Visualize the target variable(s) and their relationship with the features.

# Preprocessing and Data Engineering
## Initial overview of the dataset
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 210 entries, 0 to 209
Data columns (total 8 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   area                     210 non-null    float64
 1   perimeter                210 non-null    float64
 2   compactness              210 non-null    float64
 3   length of kernel         210 non-null    float64
 4   width of kernel          210 non-null    float64
 5   asymmetry coefficient    210 non-null    float64
 6   length of kernel groove  210 non-null    float64
 7   type                     210 non-null    int64  
dtypes: float64(7), int64(1)
memory usage: 13.3 KB
None
```
## Handling Missing values
- There are no missing values in the dataset.
```
area                       0
perimeter                  0
compactness                0
length of kernel           0
width of kernel            0
asymmetry coefficient      0
length of kernel groove    0
type                       0
dtype: int64
```
## Handling duplicate values
- There are no duplicate values in the dataset.
```
Empty DataFrame
Columns: [area , perimeter , compactness, length of kernel, width of kernel, asymmetry coefficient, length of kernel groove, type]
Index: []
```

## Descriptive Statistics
![Table_1](https://github.com/Kawyanethma/Wheat-Seed-Classification/assets/92635894/43d5f53e-90b8-4a8d-8cd0-22b30cecf3ca)

## Data Standardization/Normalization
This uses to create a level playing field for all features. For example, In the dataset, some might have higher values compared to others. To avoid that this technique used.
![graph_1](https://github.com/Kawyanethma/Wheat-Seed-Classification/assets/92635894/4c8d573e-1dac-4b60-8a4a-5e325c148596)


## Univariate Analysis
Univariate analysis is an exploratory data analysis technique that focuses on understanding the distribution and characteristics of a single variable (feature) in a dataset.
![image](https://github.com/Kawyanethma/Wheat-Seed-Classification/assets/92635894/84df4978-ee62-4cfd-a333-a9929249429e)

## Bivariate analysis
Bivariate analysis is an exploratory data analysis technique that examines the relationship between two variables (features) in a dataset.
![image](https://github.com/Kawyanethma/Wheat-Seed-Classification/assets/92635894/ef48bdca-a1e6-42d8-929b-89b35631887d)

## Evaluating skewness
According to this dataset is relatively symmetrical.
```
0    0.399889
1    0.386573
2   -0.537954
3    0.525482
4    0.134378
5    0.401667
6    0.561897
dtype: float64
```
# Deciding The Model Architecture
There are 4 models, in each model,
- No: of epochs: 100
- Initial learning rate at 0.001
- For each model optimizers were different.

## Model 1
### _Optimizer – Adam_
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 20)                160       
                                                                 
 dense_1 (Dense)             (None, 20)                420       
                                                                 
 dense_2 (Dense)             (None, 3)                 63        
                                                                 
=================================================================
Total params: 643 (2.51 KB)
Trainable params: 643 (2.51 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
![image](https://github.com/Kawyanethma/Wheat-Seed-Classification/assets/92635894/d47b4e18-2314-4e5b-88b3-28de67731570)
```
1/1 [==============================] - 0s 28ms/step - loss: 0.2981 - accuracy: 0.9048
Test Accuracy: 0.9048
```

## Model 2
### _Optimizer – Adagrad_
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_3 (Dense)             (None, 30)                240       
                                                                 
 dense_4 (Dense)             (None, 20)                620       
                                                                 
 dense_5 (Dense)             (None, 3)                 63        
                                                                 
=================================================================
Total params: 923 (3.61 KB)
Trainable params: 923 (3.61 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
![image](https://github.com/Kawyanethma/Wheat-Seed-Classification/assets/92635894/172c2642-4eb8-4fd3-9028-308c7441be92)
```
1/1 [==============================] - 0s 36ms/step - loss: 0.8617 - accuracy: 0.4762
Test Accuracy: 0.4762
```
## Model 3
### _Optimizer – SGD_
```
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_6 (Dense)             (None, 10)                80        
                                                                 
 dense_7 (Dense)             (None, 10)                110       
                                                                 
 dense_8 (Dense)             (None, 3)                 33        
                                                                 
=================================================================
Total params: 223 (892.00 Byte)
Trainable params: 223 (892.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
![image](https://github.com/Kawyanethma/Wheat-Seed-Classification/assets/92635894/8f334984-e3c6-49d3-83c8-3ac899faf19b)
```
1/1 [==============================] - 0s 29ms/step - loss: 0.7248 - accuracy: 0.8095
Test Accuracy: 0.8095
```
## Model 4
### _Optimizer – RMSprop_
```
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_9 (Dense)             (None, 6)                 48        
                                                                 
 dense_10 (Dense)            (None, 10)                70        
                                                                 
 dense_11 (Dense)            (None, 3)                 33        
                                                                 
=================================================================
Total params: 151 (604.00 Byte)
Trainable params: 151 (604.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
![image](https://github.com/Kawyanethma/Wheat-Seed-Classification/assets/92635894/9752d210-61e9-46a9-9ecc-ea9104c6f05c)
```
1/1 [==============================] - 0s 25ms/step - loss: 1.1106 - accuracy: 0.1905
Test Accuracy: 0.1905
```

# Conclusion
The results demonstrate the impact of both model architecture and optimizer selection on model performance. Through a comprehensive exploratory data analysis and model development process.
For these Artificial Nural networks (ANN) model was conducted with various neuron sizes in the hidden layers. As well as model were used different optimizers such as Adam, SGD, Adagrad
and RmsProp. In conclusion, For ANNs , both model architecture and optimizer play a crucial role in how an ANN learns and performs.
