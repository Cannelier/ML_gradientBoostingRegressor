# ML_gradientBoostingRegressor
Kaggle-like competition, predict benefits based on user features

Libraries:

pip install scikit-learn
pip install numpy
pip install matplotlib
pip install pandas

Trains GBRT algorithm on labeled_dataset_axaggpdsc.csv.
Targets column = 'Benefice net annuel'

function encoding converts categorical values into number:
	col_str = Array of categorical fields
	col_str_dicts = Nested dictionnary
		{'Field1' : {'ValueA': 0, 'ValueB':1, 'ValueC':2...},
		'Field2': {'ValueA': 0,...},
		...}
	Then replace in df dataframe for each field : ValueA => 0, ValueB => 1...


function dataprep converts mainly replaces missing values by average values.

Graphs of Deviation and Variable Importance were based on Scikit-learn doc:
https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py 