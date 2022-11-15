# [Radius Neighbors Classifier in SciKitLearn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier) for Multi-class Classification
## Base problem category as per Ready Tensor specifications.

### Tags
- radius neighbors
- multi-class classification
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- fastAPI
- nginx
- uvicorn
- docker

### Introduction

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution.

During the model development process, the algorithm was trained and evaluated on a variety of publicly available datasets such as `dna_splice_junction`, `gesture_phase`, `ipums_census_small`, `landsat_satellite`, `page_blocks`, `primary_tumor`, `soybean_disease`, `spotify_genre`, `steel_plate_fault`, `vehicle_silhouettes`.

This Multi-class Classifier is written using Python as its programming language. Scikitlearn is used to implement the main algorithm, create the data preprocessing pipeline, and evaluate the model. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. FastAPI + Nginx + uvicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.

### Setup

- create a virtualenv (_optional_)

		```bash
		mkvirtualenv radius-neighbors
		workon radius-neighbors
		```
- install dependencies

		```bash
		pip install -r requirements.txt
		```

- Run locally to generate `ml_vol`

		```bash
		cd local_test
		./run_local.py
		```

- Ensure appropriate dir structure

```bash
$PROJECT_DIR
├── datasets
│   ├── car
│   ├── dna_splice_junction
│   ├── gesture_phase
│   └── ...
├── repository
│   ├── Dockerfile
│   ├── README.md
│   ├── app
│   ├── local_test
│   ├── ml_vol
│   └── requirements.txt
└── ml_vol # (same as repository/ml_vol)
    ├── hpt
    ├── inputs
    ├── model
    └── outputs
```

- Build docker container

		```bash
		cd $PROJECT_DIR/repository
		docker build -t ready-tensor/radius-neighbors .
		```
- **Additionally**, if any requirements are changed (in `requirements.in`):

		```bash
		pip install pip-tools
		pip-compile requirements.in > requirements.txt
		pip install -r requirements.txt
		```

## Running

```bash
cd $PROJECT_DIR
docker run -it -v $PROJECT_DIR/ml_vol:/opt/ml_vol -v $PROJECT_DIR/repository/app:/opt/app -p 8080:8080 ready-tensor/radius-neighbors train
# replace train with test|serve once training is done
```

