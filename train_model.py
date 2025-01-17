import pandas as pd
import dask.dataframe as dd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

csv_file_path = r"C:\Users\iyk5988\OneDrive - HCA Healthcare\Desktop\ML Stuff\trained_data-20241206.csv"

df = dd.read_csv(csv_file_path)

chunksize = 100000
label_encoder = LabelEncoder()

X_list = []
y_list = []

memo_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

for chunk in df.to_delayed():
    chunk = chunk.compute()
    chunk = chunk.astype({
        'matched_vendor_id': 'int32',
        'gl_account': 'object',
        'primary_category_id': 'int32',
        'matched_category_id': 'int32',
        'memo': 'object'
    })

    chunk['memo'] = chunk['memo'].fillna('')

    chunk['matched_vendor_id_encoded'] = label_encoder.fit_transform(chunk['matched_vendor_id'])
    chunk['gl_account_encoded'] = label_encoder.fit_transform(chunk['gl_account'])
    chunk['primary_category_id_encoded'] = label_encoder.fit_transform(chunk['primary_category_id'])

    memo_features = memo_vectorizer.fit_transform(chunk['memo']).toarray()
    chunk = pd.concat([chunk, pd.DataFrame(memo_features, columns=memo_vectorizer.get_feature_names_out())], axis=1)

    X_list.append(chunk[['matched_vendor_id_encoded', 'gl_account_encoded', 'primary_category_id_encoded'] + list(memo_vectorizer.get_feature_names_out())])
    y_list.append(chunk['matched_category_id'])

X = pd.concat(X_list, axis=0)
y = pd.concat(y_list, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, r"C:\Users\iyk5988\OneDrive - HCA Healthcare\Desktop\ML Stuff\trained_model.pkl")
joblib.dump(memo_vectorizer, r"C:\Users\iyk5988\OneDrive - HCA Healthcare\Desktop\ML Stuff\memo_vectorizer.pkl")

print("Model and vectorizer saved successfully.")
