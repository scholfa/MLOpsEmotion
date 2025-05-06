#!/bin/bash

echo "🔧 Configuring DVC GDrive remote..."

if [[ -z "$GDRIVE_FOLDER_ID" ]]; then
  echo "❌ GDRIVE_FOLDER_ID is not set!"
  exit 1
fi

KEY=/app/secret/gdrive-sa.json
if [[ ! -f "$KEY" ]]; then
  echo "❌ Service account key file not found at $KEY!"
  exit 1
fi

# Read the client_email from the JSON
CLIENT_EMAIL=$(python3 - <<PYCODE
import json
print(json.load(open("$KEY"))["client_email"])
PYCODE
)

dvc remote remove mygdrive   2>/dev/null || true
dvc remote add -d mygdrive gdrive://$GDRIVE_FOLDER_ID

# Configure service-account usage
dvc remote modify mygdrive \
    gdrive_use_service_account true

dvc remote modify mygdrive \
    gdrive_service_account_json_file_path $KEY

dvc remote modify mygdrive \
    gdrive_service_account_user_email $CLIENT_EMAIL

echo "✅ DVC remote 'mygdrive' configured with service account $CLIENT_EMAIL."
