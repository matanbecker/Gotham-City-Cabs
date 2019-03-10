import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import ast
import pandas as pd
import numpy as np
from collections import Counter
import itertools as it
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats

# Loads data from a given path
def load_data(path,head):
	data = pd.read_csv(path)
	data = data.head(head)
	data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
	return data

# Feature generation function
def add_features(data):
	kmeans = KMeans(n_clusters=16, random_state=0)
	data['cluster'] = kmeans.fit_predict(data[['pickup_x','pickup_y','dropoff_x','dropoff_y']])
	data['euc_dist'] = data.apply(lambda p: np.linalg.norm(np.array((p.pickup_x,p.pickup_y))-np.array((p.dropoff_x,p.dropoff_y))) ,axis=1)
	data['manh_dist'] = data.apply(lambda p: abs(p.dropoff_x-p.pickup_x)+abs(p.dropoff_y-p.pickup_y) ,axis=1)
	data['direction_x'] = data.apply(lambda p: (p.dropoff_x-p.pickup_x)/p.manh_dist,axis=1)
	data['direction_y'] = data.apply(lambda p: (p.dropoff_y-p.pickup_y)/p.manh_dist,axis=1)
	data = get_density(data,2,[5])
	data = get_density(data,5,[15])
	data['dow'] = data['pickup_datetime'].dt.dayofweek
	data['hour'] = data['pickup_datetime'].dt.hour
	data['month'] = data['pickup_datetime'].dt.month
	exclude = ['pickup_datetime']
	data = data[[col for col in data.columns if col not in exclude]]
	return data

# Calculates point density and radial density for a given point size and set of radius.
def get_density(data,point_size,rs):
	train = pd.read_csv('Train.csv')
	points = zip(train['pickup_x'],train['pickup_y']) + zip(train['dropoff_x'],train['dropoff_y'])
	points = map(lambda p: (p[0]-p[0]%point_size+ float(point_size)/2.0,p[1]-p[1]%point_size+ float(point_size)/2.0),points)
	points = Counter(points)
	p_name = 'p'+str(point_size)+'_density'
	grid = pd.DataFrame(pd.Series(points),columns=[p_name])
	grid['x'] = map(lambda p: p[0],grid.index)
	grid['y'] = map(lambda p: p[1],grid.index)
	for r in rs:
		r_name = 'p'+str(point_size)+'r'+str(r)+'_density'
		grid[r_name] = grid.apply(lambda p: grid[grid['x'].between(p.x-r,p.x+r) & grid['y'].between(p.y-r,p.y+r)][p_name].sum(),axis=1)
	
	temp_grid = grid.copy()
	temp_grid.columns = ['pickup_'+col if col not in ['x','y'] else col for col in grid.columns]
	data['x'] = data['pickup_x'].apply(lambda x: x-x%point_size+ float(point_size)/2.0)
	data['y'] = data['pickup_y'].apply(lambda x: x-x%point_size+ float(point_size)/2.0)
	data = data.merge(temp_grid,on=['x','y'],how='left')

	temp_grid = grid.copy()
	temp_grid.columns = ['dropoff_'+col if col not in ['x','y'] else col for col in grid.columns]
	data['x'] = data['dropoff_x'].apply(lambda x: x-x%point_size+ float(point_size)/2.0)
	data['y'] = data['dropoff_y'].apply(lambda x: x-x%point_size+ float(point_size)/2.0)
	data = data.merge(temp_grid,on=['x','y'],how='left')

	data = data[[col for col in data.columns if col not in ['x','y']]]
	return data

# General purpose preprocessing functions.  Feed it a data set and specify which features to treat categorically 
# and it returns scaled scalars and sparse matric representations of categorical variables.
def preprocess(data,categorical_vars):
	scalar_vars = [col for col in data.columns if col not in categorical_vars + ['duration']]
	data = scale_data(data,scalar_vars)
	data = encode_categorical(data,categorical_vars)
	return data

def scale_data(data,scalar_vars):
	scaler = preprocessing.StandardScaler()
	for feat in scalar_vars:
		data[feat] = scaler.fit_transform(data[[feat]])
	return data

def encode_categorical(data,categorical_vars):
	enc = preprocessing.OneHotEncoder()
	for var in categorical_vars:
		enc.fit(data[[var]])
		data = data.join(pd.DataFrame(enc.transform(data[[var]]).toarray(),columns=[var+str(x) for x in xrange(0,len(data[var].unique()))]))
	data = data[[col for col in data.columns if col not in categorical_vars]]
	return data


# Cross validates a dataset using shufflesplit given some model
def train_model(data,features,target,model):
	X = data[features]
	y = data[target] 
	cv = ShuffleSplit(n_splits=5, random_state=0)
	mse = cross_val_score(model,X,y,cv=cv,scoring='neg_mean_squared_error').mean() * -1
	return mse

# Generates the new features and outputs the data to csv
def regenerate_data(head):
	# Load data, add features, remove outliers
	data = load_data('Train.csv',head)
	data = data[(np.abs(stats.zscore(data[[col for col in data.columns if col!='pickup_datetime']])) < 3).all(axis=1)]
	data = add_features(data)
	data = data.dropna()
	data.columns = [col if not any(x in col for x in ['dropoff_x','pickup_x','dropoff_y','pickup_y']) else 'loc_'+col for col in data.columns]
	data.to_csv('updated_data.csv')

# Fit power set of feature categories to linear models and record results
def run_linear_models(data):
	count = 0
	results = pd.DataFrame()
	total_feats = ['pickup_p2','dropoff_p2','pickup_p5','dropoff_p5','hour','month','loc','direction','manh_dist','euc_dist','dow','cluster']
	feature_space = reduce(lambda x,y: x + list(it.combinations(total_feats,y)),xrange(1,6),[])
	for feats in feature_space:
		features = [col for col in data.columns if any(f in col for f in feats)]
		model = LinearRegression()
		mse = train_model(data,features,'duration',model)
		results = results.append({'mse':mse,'f_count':len(features),'features':feats},ignore_index=True)
		count += 1
		if count%1==0:
			os.system('clear')
			print results.sort_values('mse').head(10)

	os.system('clear')
	print 'Conducted %i tests.  Best results:' % (count)
	print results.sort_values('mse').head(10)[['features','mse']]
	results.to_csv('lm_results.csv')


# Generate data with new features
regenerate_data(100000000)
# Load and preprocess the updated data
data = pd.read_csv('updated_data.csv')
data = preprocess(data,['hour','month','dow','cluster'])
# Test possible linear models and save results
run_linear_models(data)

# Load the results and perform random forest on best ten
results = pd.read_csv('lm_results.csv')
print '\nBest feature combinations used in Random Forest:'
data = pd.read_csv('updated_data.csv')
data = preprocess(data,['hour','month','dow','cluster'])

results = results.sort_values('mse').head(10)
for index,row in results.iterrows():
	feats = ast.literal_eval(row['features'])
	features = [col for col in data.columns if any(f in col for f in feats)]
	model = RandomForestRegressor()
	mse = train_model(data,features,'duration',model)
	print feats,'mse:',mse





