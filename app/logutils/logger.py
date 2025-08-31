import csv
import os
import datetime
from fastapi import Request

LOGS_DIR = os.path.join(os.getcwd(), 'app', 'logutils')
LOG_FILE = os.path.join(LOGS_DIR, 'api_logs.csv')

LOG_FIELDS = [
    'timestamp',
    'request_method',
    'endpoint',
    'status_code',
    'latency_ms',
    'client_id',
    'success',
    'prediction',
    'confidence',
    'error_message'
]

def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()
            
def log_request(
    request: Request, 
    status_code: int, 
    latency_ms: int, 
    success: bool, 
    prediction: str = None, 
    confidence: float = None, 
    error_message: str = None
):
    init_log_file()
    
    client_id = request.headers.get('X-Client-ID')
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'request_method': request.method,
        'endpoint': str(request.url.path),
        'status_code': status_code,
        'latency_ms': latency_ms,
        'client_id': client_id,
        'success': success,
        'prediction': prediction or '',
        'confidence': confidence or '',
        'error_message': error_message or ''
    }
    
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writerow(log_entry)
    except Exception as e:
        print(f"Error logging request: {e}") # CHANGE THIS
        
    return log_entry

def get_logs(filters: dict = {}):
    init_log_file()
    
    logs = []
    try:
        with open(LOG_FILE, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'start_date' in filters and 'end_date' in filters:
                    try:
                        log_date = datetime.datetime.fromisoformat(row['timestamp'])
                        start_date = datetime.datetime.fromisoformat(filters['start_date'])
                        end_date = datetime.datetime.fromisoformat(filters['end_date'])
                        if not (start_date <= log_date <= end_date):
                            continue  # Skip to the next row if outside the date range
                    except (ValueError, TypeError):
                        continue # Skip rows with invalid date format
                        
                if 'status_code' in filters and row['status_code'] != str(filters['status_code']):
                    continue

                if 'client_id' in filters and row['client_id'] != filters['client_id']:
                    continue

                if 'success' in filters and row['success'].lower() != str(filters['success']).lower():
                    continue

                logs.append(row)
                    
    except (IOError, FileNotFoundError) as e:
        raise IOError(f"Could not read log file: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while retrieving logs: {e}")
        
    return logs