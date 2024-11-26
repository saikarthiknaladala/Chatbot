import firebase_admin
from firebase_admin import credentials, db
import csv

# Initialize the Firebase Admin SDK
cred = credentials.Certificate("/Users/saikarthiknaladala/Documents/UNIBUDDY/Store/log.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://login-info-134a1-default-rtdb.firebaseio.com"
})

# Reference to the database root
ref = db.reference('/')

# Retrieve data from a specific location in the database
data = ref.get()

# Define CSV file path
csv_file_path = "/Users/saikarthiknaladala/Documents/csvfiles/firebase_data.csv"

# Write data to CSV file
with open(csv_file_path, "w", newline="") as csvfile:
    fieldnames = ['Id', 'email', 'full_name', 'last_login', 'phonenumber', 'university']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for user_id, user_data in data['users'].items():
        writer.writerow({
            'Id': user_id,
            'email': user_data['email'],
            'full_name': user_data['full_name'],
            'last_login': user_data['last_login'],
            'phonenumber': user_data['phonenumber'],
            'university': user_data['university']
        })

print(f"Data successfully saved to '{csv_file_path}'")
