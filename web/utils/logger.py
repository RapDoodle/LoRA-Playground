from datetime import datetime

from models.task_log import TaskLog


def log_message(msg):
    # The output will be as the following example:
    #   [2020-05-15 17:08:53.508167] Unable to connect
    current_time = datetime.now()
    with open('./log.txt', 'a') as log:
        log.write("[{}] {}\n".format(str(current_time), str(msg)))


def write_task_log(task_id, msg, commit=False):
    task_log = TaskLog(task_id, msg)
    task_log.save(commit=commit)
    return task_log