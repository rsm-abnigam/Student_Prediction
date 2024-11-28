import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

# Load the dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# Ensure there are no missing values in the dataset
df = df.dropna()
df.info()



# Splitting features and target also known as indepedent and dependent variables
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

# Identify categorical and numerical features
categorical_features = ["Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities", 
                        "Motivation_Level", "Internet_Access"]
numerical_features = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores"]



# Preprocessing: Scaling numerical features and encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ],
    remainder="drop"  # Drop any columns not explicitly specified
)

# Transform the data
X_preprocessed = preprocessor.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


coefficients = model.coef_
intercept = model.intercept_

feature_names = preprocessor.get_feature_names_out()
equation = f"Exam_Score = {intercept:.2f} + " + " + ".join(
    [f"{coeff:.2f} * {name}" for coeff, name in zip(coefficients, feature_names)]
)
print("Regression Equation:")
print(equation)


import joblib
import pandas as pd

# Save the preprocessor and model
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(model, "regression_model.pkl")


# Load the preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("regression_model.pkl")


def clean_feature_names(feature_names):
    """
    Clean feature names by removing 'num__' and 'cat__' prefixes.
    :param feature_names: List of feature names with prefixes.
    :return: List of cleaned feature names.
    """
    return [name.replace("num__", "").replace("cat__", "") for name in feature_names]

def predict_exam_score(input_data):
    """
    Predict Exam_Score based on input features.
    :param input_data: Dictionary containing input features and values.
    :return: Predicted Exam_Score.
    """
    # Convert input data to a DataFrame for processing
    input_df = pd.DataFrame([input_data])
    
    # Transform input data using the preprocessor
    input_transformed = preprocessor.transform(input_df)
    
    # Predict using the trained model
    predicted_score = model.predict(input_transformed)
    
    return predicted_score[0]

# Example for retrieving and cleaning feature names for the regression equation
feature_names = preprocessor.get_feature_names_out()
cleaned_feature_names = clean_feature_names(feature_names)

# Display the regression equation without prefixes
coefficients = model.coef_
intercept = model.intercept_
regression_equation = (
    f"Exam_Score = {intercept:.2f} + " + 
    " + ".join(
        [f"{coeff:.2f} * {name}" for coeff, name in zip(coefficients, cleaned_feature_names)]
    )
)

print("Regression Equation:")
print(regression_equation)



new_student = {
    "Hours_Studied": 25,
    "Attendance": 85,
    "Parental_Involvement": "Medium",
    "Access_to_Resources": "High",
    "Extracurricular_Activities": "Yes",
    "Sleep_Hours": 7,
    "Previous_Scores": 78,
    "Motivation_Level": "Medium",
    "Internet_Access": "Yes"
}
predicted_score = predict_exam_score(new_student)
print(f"Predicted Exam Score: {predicted_score:.2f}")


import matplotlib.pyplot as plt
import numpy as np

# Extract coefficients and feature names
coefficients = model.coef_
feature_names = preprocessor.get_feature_names_out()

# Clean feature names by removing prefixes
cleaned_feature_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]

# Sort features by importance
sorted_indices = np.argsort(np.abs(coefficients))[::-1]
sorted_coefficients = coefficients[sorted_indices]
sorted_features = np.array(cleaned_feature_names)[sorted_indices]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_coefficients)
plt.title("Feature Importance (Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Features")
plt.gca().invert_yaxis()
plt.show()


from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Predict on the test set
y_pred = model.predict(X_test)

# Scatter plot: Actual vs Predicted Exam Scores
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, label="Predicted vs Actual")
plt.title("Actual vs. Predicted Exam Scores")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Perfect Prediction")
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


import seaborn as sns
import matplotlib.pyplot as plt

# Ensure y_test and y_pred are numpy arrays for compatibility with seaborn
y_test_array = np.array(y_test)
y_pred_array = np.array(y_pred)

# Plot distributions using KDE
plt.figure(figsize=(8, 6))
sns.kdeplot(y_test_array, label="Actual Exam Scores", shade=True, bw_adjust=1.0, color="blue")
sns.kdeplot(y_pred_array, label="Predicted Exam Scores", shade=True, bw_adjust=1.0, color="orange")
plt.title("Distribution of Actual vs. Predicted Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Density")
plt.legend()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Convert categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Compute correlation matrix
correlation_matrix = df_encoded.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Correlation Heatmap")
plt.show()


# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--", label="Zero Residual")
plt.title("Residual Plot")
plt.xlabel("Predicted Exam Score")
plt.ylabel("Residual")
plt.legend()
plt.show()


# Plot feature vs. target
plt.figure(figsize=(8, 6))
plt.scatter(df["Hours_Studied"], df["Exam_Score"], alpha=0.7)
plt.title("Impact of Hours Studied on Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()


# Select key numerical features
key_features = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Exam_Score"]

# Pairplot
sns.pairplot(df[key_features], diag_kind="kde")
plt.show()



from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")





import streamlit as st
import pandas as pd
import plotly.express as px
import pdfkit

# Load Data
@st.cache_data
def load_data():
    # Example dataset
    data = {
        "Student": ["Alice", "Bob", "Charlie", "David"],
        "Hours_Studied": [5, 8, 3, 6],
        "Attendance": [90, 85, 78, 88],
        "Previous_Scores": [75, 82, 60, 70],
        "Exam_Score": [78, 88, 68, 75],
    }
    return pd.DataFrame(data)

df = load_data()

# Dashboard Layout
st.title("Interactive Dashboard: Predicting Student Performance")
st.write("Explore and analyze student performance data.")

# Filters
st.sidebar.header("Filters")
min_score = st.sidebar.slider("Minimum Exam Score", 0, 100, 60)
filtered_data = df[df["Exam_Score"] >= min_score]

# Visualizations
st.subheader("Filtered Data")
st.write(filtered_data)

# Scatter Plot: Hours Studied vs Exam Score
st.subheader("Hours Studied vs. Exam Score")
fig = px.scatter(filtered_data, x="Hours_Studied", y="Exam_Score", color="Student", size="Attendance")
st.plotly_chart(fig)

# Exclude non-numeric columns
numeric_df = df.select_dtypes(include=["number"])

# Compute correlation matrix
corr = numeric_df.corr()

# Plot heatmap
st.subheader("Correlation Heatmap")
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="viridis", title="Correlation Heatmap")
st.plotly_chart(fig_corr)


# Generate Report
st.subheader("Generate Report")

def generate_html_report(filtered_data):
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Student Performance Report</h1>
        <h2>Summary</h2>
        <p>Total Students: {len(filtered_data)}</p>
        <h2>Filtered Data</h2>
        {filtered_data.to_html(index=False)}
    </body>
    </html>
    """
    return html_content

if st.button("Generate Report"):
    html_report = generate_html_report(filtered_data)
    with open("report.html", "w") as f:
        f.write(html_report)

    pdfkit.from_file("report.html", "Student_Performance_Report.pdf")
    st.success("Report generated successfully!")
    with open("Student_Performance_Report.pdf", "rb") as f:
        st.download_button("Download Report", f, file_name="Student_Performance_Report.pdf")

