import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, AveragePooling1D, MaxPooling1D, Conv1D, Flatten
from keras import optimizers
from keras import backend as K
from keras.callbacks import Callback,ModelCheckpoint
K.set_image_dim_ordering('th')


# Custom callback class to plot progress in real time
class PlotMSE(Callback):
    def on_epoch_end(self, epoch, logs=None):
    	global log
        log = log.append(logs, ignore_index=True)
        if len(log)>1:
        	log = log[(log['mean_squared_error']<=120000) & (log['val_mean_squared_error']<=120000)]
        	#log = log.tail(200)
        if epoch%1==0:
        	plt.clf()
	        plt.plot(log.index, log['mean_squared_error'],c='r')
	        plt.plot(log.index, log['val_mean_squared_error'], c='b')
	        plt.pause(0.00001)
	        fig.canvas.draw()

# General purpose preprocessing functions
def preprocess(data,categorical_vars):
	scalar_vars = [col for col in data.columns if not any(x in col for x in categorical_vars+['duration'])]
	data = scale_mean_var(data,scalar_vars)
	data = encode_categorical(data,categorical_vars)
	data = data.dropna()
	return data

def scale_mean_var(data,scalar_vars):
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

# Split data to target and features
def target_loc(data):
	y = data[['duration']]
	X = data[[col for col in data.columns if col not in ['duration']]]
	return X,y

# Forward select best features for using in neural network
# NOTE we ultimately did not find this successful, so we do not report on it
def fw_select_features(data):
	y_cols = ['duration']
	y = data[y_cols]
	x_count = len([x for x in data.columns if x not in y_cols])

	X = pd.DataFrame()
	feats = []

	bests = pd.DataFrame()
	for mid in xrange(1,100):
		results = pd.DataFrame()
		for i in xrange(1,x_count): 
			feat = 'Var'+str(i)
			if not any(feat in x for x in X.columns):
				for op in ['X']:
					name = op + feat
					temp_feats = X.columns+[name]
					if op=='X':
						temp_X = data[feat]
					elif op=='D':
						temp_X = data[feat].apply(lambda x: 1/x)
					elif op=='S':
						temp_X = data[feat].apply(lambda x: x**2)
					temp_X.name = name
					if len(X)>0:
						temp_X = X.join(temp_X)
					else:
						temp_X = pd.DataFrame(temp_X)
					X_train, X_test, y_train, y_test = train_test_split(temp_X.values, y.values, test_size=0.2, random_state=71)
					lm = LinearRegression()
					lm.fit(X_train,y_train)
					results = results.append({\
						'mid':str(mid)+name,\
						'name':name,\
						'feat':feat,\
						'feats':temp_feats,\
						'mse':mean_squared_error(y_test,lm.predict(X_test)),\
						'R^2':lm.score(X_test,y_test)}, ignore_index=True)

		best = results.iloc[results['R^2'].argmax]
		bests = bests.append(best,ignore_index=True)
		op = best['name'][0]
		feat = best['feat']
		if op=='X':
			temp_X = data[feat]
		elif op=='D':
			temp_X = data[feat].apply(lambda x: 1/x)
		elif op=='S':
			temp_X = data[feat].apply(lambda x: x**2)
		temp_X.name = best['name']
		if len(X)>0:
			if temp_X.name not in X.columns:
				X = X.join(temp_X)
		else:
			X = pd.DataFrame(temp_X)
		plt.clf()
		plt.plot(xrange(1,len(bests)+1),bests['R^2'])
		plt.pause(0.000001)
		print X.columns
		print best['R^2']

	feats = bests.iloc[bests['R^2'].argmax]['feats']
	return feats,X


# Built the neural network model
def build_model(activation,optimizer):
    model = Sequential()
    #model.add(Conv1D(nb_filter=2,\
    #	filter_length=8,border_mode="valid",activation=activation,\
   	#	subsample_length=1,input_shape=(num_feats,1,)))
    #model.add(MaxPooling1D(pool_size=4, strides=None,input_shape=(num_feats,1,)))
    #model.add(AveragePooling1D(pool_size=16, strides=None,input_shape=(num_feats,1,)))
    model.add(Dropout(0.05,input_shape=(num_feats,1,)))
    model.add(Flatten(input_shape=(num_feats,1,)))
    model.add(Dense(128,input_dim=num_feats ,activation=activation))
    model.add(Dropout(0.05,input_shape=(num_feats,1,)))
    model.add(Dense(128,input_dim=num_feats ,activation=activation))
    model.add(Dropout(0.05,input_shape=(num_feats,1,)))
    model.add(Dense(512,input_dim=num_feats ,activation=activation))
    model.add(Dropout(0.1,input_shape=(num_feats,1,)))
    model.add(Dense(num_classes, activation='softplus'))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    return model


np.random.seed(7)
# Extract and preprocess the data
data = pd.read_csv('updated_data.csv')
data = data.dropna()
data = preprocess(data,['hour','month','dow','cluster'])

# Split into training and test data
X,y = target_loc(data)
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.15, random_state=19)

# Prepare plot for validation/training mse
fig = plt.gcf()
fig.show()
fig.canvas.draw()
log = pd.DataFrame()

# Reshape data (only needed for convoluton and pooling)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
y_train = y_train.reshape(y_train.shape[0],y_train.shape[1])
y_test = y_test.reshape(y_test.shape[0],y_test.shape[1])

num_classes = len(y.columns)
num_feats = len(X.columns)

# Prepare the model
optimizer = optimizers.Adam()
batch_size = 30
model = build_model('relu',optimizer)

# Load model
#model.load_weights('saved/weights.21-65807.77.hdf5')

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000000, batch_size=batch_size,callbacks=[PlotMSE(),ModelCheckpoint('saved/weights.{epoch:02d}-{val_loss:.2f}.hdf5')])

# Show the best result
print log['val_mean_squared_error'].min()



# For making predictions on test set
"""
test = pd.read_csv('test_data.csv')
test = preprocess(test,['hour','month','dow','cluster'])
test = np.array(test.values)
test = test.reshape(test.shape[0],test.shape[1],1)
pred = pd.DataFrame(model.predict(test),columns=['duration'])
pred.to_csv('PREDICTION.csv')
"""



