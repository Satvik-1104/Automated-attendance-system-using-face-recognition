from celery import shared_task
from .database import SessionLocal
from .models import EntryExitTime
from .routers.attendance import derive_schedule_id
from .config import ENTRY_LOGS_DIR, EXIT_LOGS_DIR, PROCESSED_ENTRY_LOGS_DIR, PROCESSED_EXIT_LOGS_DIR
import os
from datetime import datetime


@shared_task
def process_log_files():
    db = SessionLocal()
    try:
        # Process entry logs
        for filename in os.listdir(ENTRY_LOGS_DIR):
            file_path = os.path.join(ENTRY_LOGS_DIR, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        fields = line.strip().split(',')
                        if len(fields) != 7:
                            continue  # Skip malformed lines
                        track_id, roll_number, _, _, timestamp_str, came_in_str, classroom_code = fields
                        if roll_number == "Unknown":
                            continue  # Skip unrecognized faces
                        try:
                            roll_number = int(roll_number)
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                            came_in = came_in_str.lower() == "true"
                        except ValueError:
                            continue  # Skip invalid data
                        if came_in:
                            schedule_id = derive_schedule_id(timestamp, classroom_code, db)
                            if schedule_id:
                                # Create new entry record
                                new_record = EntryExitTime(
                                    roll_number=roll_number,
                                    schedule_id=schedule_id,
                                    entry_time=timestamp,
                                    exit_time=None
                                )
                                db.add(new_record)
                                db.commit()
                # Move file to processed folder
                processed_path = os.path.join(PROCESSED_ENTRY_LOGS_DIR, filename)
                os.rename(file_path, processed_path)

        # Process exit logs
        for filename in os.listdir(EXIT_LOGS_DIR):
            file_path = os.path.join(EXIT_LOGS_DIR, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        fields = line.strip().split(',')
                        if len(fields) != 7:
                            continue
                        track_id, roll_number, _, _, timestamp_str, came_in_str, classroom_code = fields
                        if roll_number == "Unknown":
                            continue
                        try:
                            roll_number = int(roll_number)
                            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                            came_in = came_in_str.lower() == "true"
                        except ValueError:
                            continue
                        if came_in:
                            schedule_id = derive_schedule_id(timestamp, classroom_code, db)
                            if schedule_id:
                                # Find the latest unmatched entry
                                record = db.query(EntryExitTime).filter(
                                    EntryExitTime.roll_number == roll_number,
                                    EntryExitTime.schedule_id == schedule_id,
                                    EntryExitTime.exit_time.is_(None)
                                ).order_by(EntryExitTime.entry_time.desc()).first()
                                if record:
                                    record.exit_time = timestamp
                                    db.commit()
                                # If no matching entry, optionally log a warning (add logging if desired)
                # Move file to processed folder
                processed_path = os.path.join(PROCESSED_EXIT_LOGS_DIR, filename)
                os.rename(file_path, processed_path)
    finally:
        db.close()
        