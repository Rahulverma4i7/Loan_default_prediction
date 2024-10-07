from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = 'model_debt.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Expected features (same as the model training)
expected_features = names = [
    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Debt', 
     'Gender_encoded', 'HasMortgage_encoded', 'HasDependents_encoded', 
    'HasCoSigner_encoded', 'MaritalStatus_Divorced_encoded', 
    'MaritalStatus_Married_encoded', 'MaritalStatus_Single_encoded', 
    "Education_Bachelor's_encoded", 'Education_High School_encoded', 
    "Education_Master's_encoded", 'Education_PhD_encoded', 
    'LoanPurpose_Auto_encoded', 'LoanPurpose_Business_encoded', 
    'LoanPurpose_Education_encoded', 'LoanPurpose_Home_encoded', 
    'LoanPurpose_Other_encoded', 'EmploymentType_Full-time_encoded', 
    'EmploymentType_Part-time_encoded', 'EmploymentType_Self-employed_encoded', 
    'EmploymentType_Unemployed_encoded'
]

print(len(expected_features))
@app.route('/predict', methods=['POST'])


def predict():
    try:
        # Extract data from the request JSON
        data = request.get_json()
        
        # Create a DataFrame from the request data
        input_data = pd.DataFrame(data, columns=expected_features)
        
        # Make prediction using the model
        prediction = model.predict(input_data)
        
        # Convert model output to a meaningful result
        results = ['Default' if pred == 1 else 'Not Default' for pred in prediction]
        
        # Return the result as JSON
        return jsonify({'prediction': results})
        
    except Exception as e:
        # If there's an error, return it as JSON
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

    # app.run(debug=True)
    
    
    
