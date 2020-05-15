# topTenCreditNeuralNet_Model
 This is the working Repository for Westend Financial Bank final project. It showcases different methods for Machine Learning.

# Overview

	Our Team collaborated on a Neural Network Model with the objective to train a model to predict an outcome of Credit Risk for potential applicants. The NN model is based on the previous customer Default patterns from the Credit dataset. The Credit_Data_Original.csv is comprised of individuals’ prior history of 30 various features including categories of loan detail, purpose of loan, financial information, and personal information such as employment and years of residency.  In order to produce the Default model and analyze outcomes, we employed Jupiter Notebook to import key libraries: Pandas, Matplotlib.pyplot, Numpy, SkLearn, and Tensorflow.keras. 

	Utilizing Random Forest and sorting Feature Importance on our Credit_Data_Original.csv on another notebook, we were able to narrow our final Credit_Data_Revised.csv to hold the top ten features of importance to optimize results when training the neural network model. 

	Ten Features of Importance: Loan Amount, Checking Acct Bal, Applicant Age, Loan Duration, Credit History, Years Employed, Savings Acct Bal, Install Rate, Years of Residency, and Job Type. The Top Ten Features are the same 10 questions on the Application Homepage.  The Default Model will run an applicant’s input and post a result status of approval.

# Read Data, Preprocess, and Build a Model

	 Pandas operates by reading in the revised csv data and additional data clean up. 
	This is followed by Preprocessing the dataset with multiple functions of SkLearn.   
	 SkLearn train_test_split function is essential to parse out the data into two subsets. Each subset includes 75% for training data and 25% for testing data.
	Sklean MinMaxScaler scales and standardize the data for better processing to put into Neural Network.
	SkLearn LabelEncoder converts all data to normalize labels. Also, the function transforms any non-numerical to numerical. This coincides with the next function by Tensorflow Keras.
	Once all labels are numerical, Tensorflow Keras to_categorical coverts those integers to a binary matrix for computation.
	The data is ready!!!
	Tensorflow Keras will build the model
	Functions of tensorflow.keras.models 
	model = Sequential()      
•	 *creates the initial model
	Functions of tensorflow.keras.layers import Dense 
	model.add(Dense(units=30, activation='relu', input_dim=10))
	model.add(Dense(units=30, activation='relu'))
	model.add(Dense(units=2, activation='softmax'))
•	Adds input layer, hidden layers, and output(classifier layer)
	Functions to compile model (algorithm)
	model.compile(optimizer='adam', (basic learning rate .001 with beta1 float close to 1)
	loss='categorical_crossentropy', (opt binary_crossentropy )
	metrics=['accuracy'])
	Functions to fit model (training the model)
	model.fit(
	    X_train_scaled,
	    y_train_categorical,
	    epochs=50,
	    shuffle=True,
	    verbose=2
	Function to Evaluate for accuracy 
	model_loss, model_accuracy = model.evaluate(
	    X_test_scaled, y_test_categorical, verbose=2)
	Save that model!
	model.save("DEFAULT_model_trained_top_10.h5")


# Model Evaluation
	Model accuracy is at 76%, which is a practical ratio without over fitting the model.

Time to Predict and Compare 

Our Team compared relationally 250 rows of that data of Actual and Predicted DEFAULT results. 
	First, utilize the function model.predict_classes() on the test data .
	encoded_predictions = default_model.predict_classes(X_test_scaled[:])
	prediction_labels = label_encoder.inverse_transform(encoded_predictions)
	Then, print Actual Outcomes and Variable Outcomes to compare.
	print(f"DEFAULT OUTCOME KEY: (DEFAULT YES: 1 , DEFAULt NO: 0 )")
	print(f"------------------------------------------------------")
	print(f"Predicted Outcome: {prediction_labels}")
	print(f"Actual Outcome: {list(y_test[:])}")
 

# For Loop Fun! 
	Our Team created a new Data Frame that held these values to easily visualize and compare.
	Additionally, we created a STATUS column. We applied string values to that column that contains a For Loop function to state the approval status of each account .
	The loop result is applied under 3 parameters: if predictive and actual outcomes both return default, apply always Denied, if both outcomes did not return default, always apply Approved, mixed results, apply Pending Further Analysis.
	This visualization showcases the first 20 results.



# Analyzing the Results of Predictive Outcomes
	We gathered the total count of each valued result.
	statusValueCount= approval_df["Approval_Status"].value_counts() 
	totalStatusOutcome = len(approval_df["Approval_Status"])

	We calculated the percentages.
	statusOutcomeRatio = statusValueCount/ totalStatusOutcome

	The results of the Default model indicate a high Approved rating of nearly 64%. 
	Each other these bar charts visualize the higher Approved rating opposed to Pending Further Analysis and Denied ratings combined. Both Total Predictive Outcomes and Percentages Predictive Outcomes  are plotted on the Double Y Axis line and bar chart.
   


# Random Forest and Features of Importance
	Definition: 
	A random forest is a meta estimator that fits numerous decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
(scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

	 The Random Forest is ensemble method from SkLearn library, it shows how appropriated columns values to feature importance ratios.  So, each individual test has variable of conditions is based on a ratio importance that help determine the predicted result.  It is the based logic that outlines the mathematical process in the final decision and/or prediction.  The CliffsNotes of Machine Learning’s little black box.

	This Random Forest read in 31 Columns from the Credit DataOriginal.csv to classify.   The RandomForestClassifier from SkLearn fits the default.data (less the DEFAULT column) with  default.target ( just the DEFAULT column). 

	Automatically, the RandomForestClassifier will calculate the feature importance rating. The rating is weighted value of each Feature (column) in the decision process. By sorting the feature importance in descending order, the highest  feature rating will list 1 – 30.

	In order to analyze the list, we created a Data Frame of the list Feature Importance and their ratings. Then, we utilized iloc of the top ten while we sort by descending order . This returned the Top Ten Importance Features we used to revise specific columns in a new data frame.  The 10 columns along with the DEFAULT column became the Credit_Data_Revised.cvs used in the Neural Network Model.
