import gspread
from google.oauth2.service_account import Credentials

# Define the scope
scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Authenticate
creds = Credentials.from_service_account_file("cred.json", scopes=scopes)
client = gspread.authorize(creds)

# === SHEET 1: Delivery Suggestion ===
sheet1_id = "1ZqRUK9oGRuVGEp1N1UjQYKoPT-6_8Dqgv2hN831Ch1k"
workbook1 = client.open_by_key(sheet1_id)
sheet1 = workbook1.get_worksheet(1) # or use get_worksheet(0) or get_worksheet(index)

data1 = sheet1.get_all_values()

with open("delivery_suggestion.txt", "w", encoding="utf-8") as f1:
    for row in data1:
        f1.write("\t".join(map(str, row)) + "\n")


# === SHEET 2: IPO Analysis ===
sheet2_id = "1W18qQXCsgM4e29eb6TXrmdidZlwjXyfrc_sCTgVT7pI"
workbook2 = client.open_by_key(sheet2_id)
sheet2 = workbook2.get_worksheet(6)  # or get_worksheet(index)

data2 = sheet2.get_all_values()

with open("ipo_analysis.txt", "w", encoding="utf-8") as f2:
    for row in data2:
        f2.write("\t".join(map(str, row)) + "\n")

print("Data saved successfully to delivery_suggestion.txt and ipo_analysis.txt")
