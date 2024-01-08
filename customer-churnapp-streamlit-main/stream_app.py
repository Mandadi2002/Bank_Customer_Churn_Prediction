import pickle
import streamlit as st
import pandas as pd
from PIL import Image
model_file = 'C:\\Users\\LENOVO\\Downloads\\customer-churnapp-streamlit-main\\customer-churnapp-streamlit-main\\model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


def main():

	image = Image.open('C:\\Users\\LENOVO\\Downloads\\customer-churnapp-streamlit-main\\customer-churnapp-streamlit-main\\images\\icone.png')
	image2 = Image.open('C:\\Users\\LENOVO\\Downloads\\customer-churnapp-streamlit-main\\customer-churnapp-streamlit-main\\images\\image.png')
	st.image(image,use_column_width=False)
	add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?",
	("Online", "Batch"))
	st.sidebar.info('This app is created to predict Customer Churn')
	st.sidebar.image(image2)
	st.title("Predicting Customer Churn")
	if add_selectbox == 'Online':
		gender = st.selectbox('Gender:', ['male', 'female'])
		NumberofProducts = st.selectbox(' Number of Products:', [1,2,3,4])
		HasCrCard = st.selectbox('Do the Customer have Credit Card (1=Yes,0=No) ?:', [0,1])
		IsActiveMember = st.selectbox('Is the Customer Active Member(1=Yes,0=No):',[0,1])
		Geography = st.selectbox('Country:', ['Germany', 'France', 'Spain'])
		Age = st.number_input('Age:', min_value=18, max_value=90, value=18)
		EstimatedSalary = st.number_input('Enter the Estimated Salary',min_value=5000, max_value=90000, value=5000)
		CreditScore = st.number_input('Credit Score', min_value=376, max_value=850, value=376)
		tenure = st.number_input('Number of months the customer has been with the current telco provider(Tenure) :', min_value=0, max_value=240, value=0)
		monthlycharges= st.number_input('Monthly charges :', min_value=0, max_value=240, value=0)
		totalcharges = tenure*monthlycharges
		output= ""
		output_prob = ""
		input_dict={
				"gender":gender,
				"NumberofProducts": NumberofProducts,
				"Age":Age,
				"HasCrCard":HasCrCard,
				"Geography":Geography,
				"EstimatedSalary":EstimatedSalary,
				"CreditScore": CreditScore,
				"IsActiveMember":IsActiveMember,
				"tenure": tenure,
				"monthlycharges": monthlycharges,
				"totalcharges": totalcharges
			}

		if st.button("Predict"):
			X = dv.transform([input_dict])
			y_pred = model.predict_proba(X)[0, 1]
			churn = y_pred >= 0.5
			output_prob = float(y_pred)
			output = bool(churn)
		st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))
	if add_selectbox == 'Batch':
		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
		if file_upload is not None:
			data = pd.read_csv(file_upload)
			X = dv.transform([data])
			y_pred = model.predict_proba(X)[0, 1]
			churn = y_pred >= 0.5
			churn = bool(churn)
			st.write(churn)

if __name__ == '__main__':
	main()