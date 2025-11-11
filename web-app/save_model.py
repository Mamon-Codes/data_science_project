# save_model.py
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# 1️⃣ Load dataset
df = pd.read_csv("cleaned.csv")

# 2️⃣ Clean column names
df.columns = df.columns.str.strip()  # remove leading/trailing spaces

# 3️⃣ Rename columns to match your desired feature names
df.rename(columns={
    'Your Academic Stage': 'acadmic_stage',
    'What coping strategy you use as a student?': 'strategy_used',
    'Do you have any bad habits like smoking, drinking on a daily basis?': 'bad_habbits',
    'What would you rate the academic  competition in your student life': 'academic_competation',
    'Rate your academic stress index': 'stress_level'
}, inplace=True)

# 4️⃣ Define features and target
categorical_cols = ['acadmic_stage', 'Study Environment', 'strategy_used', 'bad_habbits']
numerical_cols = ['Peer pressure', 'Academic pressure from your home', 'academic_competation']

X = df[categorical_cols + numerical_cols]
y = df['stress_level']

# 5️⃣ Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs', C=10))
])

# 6️⃣ Train the model
pipe.fit(X, y)

# 7️⃣ Save the trained pipeline
joblib.dump(pipe, "stress_model.pkl")
print("Saved model to stress_model.pkl")


