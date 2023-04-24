import requests

def get_queue_task():
    url = 'http://127.0.0.1:80/api/v1/queue/task'
    response = requests.get(url)

    if response.status_code == 200:
        json_response = response.json()
        return json_response
    else:
        print(f"Error: Request returned status code {response.status_code}")
        return None

if __name__ == "__main__":
    queue_task = get_queue_task()
    if queue_task:
        print(queue_task)