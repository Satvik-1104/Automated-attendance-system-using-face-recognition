from celery import Celery
from celery.schedules import crontab

celery_app = Celery('attendance_system', broker='redis://localhost:6379/0')  # Adjust broker as per your setup

celery_app.conf.beat_schedule = {
    'process-log-files-every-5-minutes': {
        'task': 'attendance_system.tasks.process_log_files',
        'schedule': crontab(minute='*/5'),  # Runs every 5 minutes
    },
}
celery_app.conf.timezone = 'Asia/Kolkata'  # Ensure timezone matches IST
