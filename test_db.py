from dotenv import load_dotenv
import psycopg2
import os

print("Script started")

load_dotenv(dotenv_path=".env")

db_url = os.getenv("DATABASE_URL")
print("DATABASE_URL present:", bool(db_url))

if not db_url:
    raise ValueError("DATABASE_URL not found in .env")

try:
    conn = psycopg2.connect(db_url)
    print("Connected successfully!")
finally:
    if "conn" in locals():
        conn.close()
