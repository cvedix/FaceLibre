import subprocess
import random
from datetime import datetime, timedelta

DB_USER = "root"
DB_PASS = "Admin123456"
DB_NAME = "usc"
DB_TABLE = "tc_face_similarity"

def get_all_ids():
    cmd = ["mysql", "-h", "127.0.0.1", "-P", "3306", f"-u{DB_USER}", f"-p{DB_PASS}", "-N", "-e", f"SELECT id FROM {DB_TABLE}", DB_NAME]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except subprocess.CalledProcessError as e:
        print(f"[-] MySQL Error: {e.stderr}")
        return []

def update_timestamp(record_id, timestamp):
    sql = f"UPDATE {DB_TABLE} SET timestamp = '{timestamp}' WHERE id = {record_id};"
    cmd = ["mysql", "-h", "127.0.0.1", "-P", "3306", f"-u{DB_USER}", f"-p{DB_PASS}", "-e", sql, DB_NAME]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] MySQL Error updating ID {record_id}: {e.stderr}")
        return False

def main():
    ids = get_all_ids()
    if not ids:
        print("[-] No records found to update.")
        return

    print(f"[*] Updating timestamps for {len(ids)} records...")
    
    count = 0
    for record_id in ids:
        random_days = random.randint(0, 30)
        random_hours = random.randint(0, 23)
        random_minutes = random.randint(0, 59)
        random_seconds = random.randint(0, 59)
        ts = datetime.now() - timedelta(days=random_days, hours=random_hours, minutes=random_minutes, seconds=random_seconds)
        timestamp_str = ts.strftime('%Y-%m-%d %H:%M:%S')

        if update_timestamp(record_id, timestamp_str):
            count += 1
            if count % 50 == 0:
                print(f"[+] Updated {count} records...")

    print(f"[+] Finished. Total records updated: {count}")

if __name__ == "__main__":
    main()
