from datetime import datetime
import subprocess as subproc
import time


if __name__ == '__main__':
    cmd = ['bash', 'run_minecraft']
    num_proc = 4

    print('Start pooling')
    # proc = [subproc.Popen(cmd, stdout = subproc.PIPE) for i in range(num_proc)]
    proc = [None for _ in range(num_proc)]

    while 1:
        time.sleep(10)
        for i in range(num_proc):
            if proc[i] == None or proc[i].poll() != None:
                print('[%s] Start new instance' % datetime.now().strftime("%H:%M:%S"))
                proc[i] = subproc.Popen(cmd, stdout = subproc.PIPE)
