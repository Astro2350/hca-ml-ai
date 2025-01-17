import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load(r"C:\Users\iyk5988\OneDrive - HCA Healthcare\Desktop\ML Stuff\trained_model.pkl")
memo_vectorizer = joblib.load(r"C:\Users\iyk5988\OneDrive - HCA Healthcare\Desktop\ML Stuff\memo_vectorizer.pkl")

test_csv_path = r"C:\Users\iyk5988\OneDrive - HCA Healthcare\Desktop\ML Stuff\tested_data-20241206.csv"

print("Loading test data...")
test_data = pd.read_csv(test_csv_path)

print("Encoding test data...")
test_data['memo'] = test_data['memo'].fillna('')

print("Transforming 'memo' column...")
memo_features = memo_vectorizer.transform(test_data['memo']).toarray()

X_test = test_data[['matched_vendor_id', 'gl_account', 'primary_category_id']].copy()
memo_feature_df = pd.DataFrame(memo_features, columns=memo_vectorizer.get_feature_names_out())
X_test = pd.concat([X_test, memo_feature_df], axis=1)

label_encoder = LabelEncoder()
X_test['matched_vendor_id_encoded'] = label_encoder.fit_transform(X_test['matched_vendor_id'])
X_test['gl_account_encoded'] = label_encoder.fit_transform(X_test['gl_account'])
X_test['primary_category_id_encoded'] = label_encoder.fit_transform(X_test['primary_category_id'])

X_test = X_test[['matched_vendor_id_encoded', 'gl_account_encoded', 'primary_category_id_encoded'] + list(memo_feature_df.columns)]

print("Making predictions...")
predictions = model.predict(X_test)

output_path = r"C:\Users\iyk5988\OneDrive - HCA Healthcare\Desktop\ML Stuff\predicted_test_results.csv"
test_data['predicted_matched_category_id'] = predictions
test_data.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")
