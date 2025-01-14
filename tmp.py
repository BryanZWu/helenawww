# Based on RealPython Threading Example page at https://realpython.com/intro-to-python-threading/ and
#   Python.org _thread library documentation at
#   https://docs.python.org/3/library/_thread.html?highlight=_thread#module-_thread
import logging
import random
import sys
import time
from threading import Thread, Lock, Condition

from core import Core


def execute_ticketing_system_participation(ticket_number, part_id, shared_variable):
    output_file_name = "output-" + part_id + ".txt"
    # NOTE: Do not remove this print statement as it is used to grade assignment,
    # so it should be called by each thread
    print("Thread retrieved ticket number: {} started".format(ticket_number), file=open(output_file_name, "a"))
    time.sleep(random.randint(0, 10))
    # wait until your ticket number has been called
    cv = shared_variable[0]
    # output your ticket number and the current time

    cv.acquire()
    while shared_variable[1] != ticket_number:
        cv.wait()
    shared_variable[1] += 1
    cv.release()
    # cv.wait_for(shared_variable[1] == ticket_number)
    

    # while True:
    #     lock.wait() # Condition.wait
    #     lock.acquire() # does not wait for the shared variable to change
    #     if shared_variable[1] == ticket_number:
    #         # do stuff here
    #         shared_variable[1] += 1
    #         lock.notify_all()
    #         break
    #     lock.release()
    # lock.release()

    # while True:
    #     if shared_variable[1] == ticket_number:
    #         lock.acquire()
    #         shared_variable[1] += 1
    #         break
    # lock.release()


    # NOTE: Do not remove this print statement as it is used to grade assignment,
    # so it should be called by each thread

    print("Thread with ticket number: {} completed".format(ticket_number), file=open(output_file_name, "a"))
    return 0


class Assignment(Core):

    USERNAME = "USERNAME"
    active_threads = []

    def __init__(self, args):
        self.num_threads = 1
        self.args_conf_list = [['-n', 'num_threads', 1, 'number of concurrent threads to execute'],
                                ['-u', 'user', None, 'the user who is turning in the assignment, needs  to match the '
                                                    '.user file contents'],
                               ['-p', 'part_id', 'test', 'the id for the assignment, test by default']]
        super().__init__(self.args_conf_list)
        super().parse_args(args=args)
        _format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=_format, level=logging.INFO,
                            datefmt="%H:%M:%S")

    def run(self):
        output_file_name = "output-" + self.part_id + ".txt"
        open(output_file_name, 'w').close()
        if self.test_username_equality(self.USERNAME):
            sleeping_time = 0
            shared_var = [Condition(), 0]
            for index in range(self.num_threads):
                logging.info("Assignment run    : create and start thread %d.", index)
                # This is where you will start a thread that will participate in a ticketing system
                # have the thread run the execute_ticketing_system_participation function
                Thread(target=execute_ticketing_system_participation, args=(index, self.part_id, shared_var)).start()
                
                # Threads will be given a ticket number and will wait until a shared variable is set to that number

                # The code will also need to know when all threads have completed their work
                sleeping_time += 1
            shared_var[0].notify_all()
            time.sleep(sleeping_time)
            self.manage_ticketing_system()
            logging.info("Assignment completed all running threads.")
            return 0
        else:
            logging.error("Assignment had an error your usernames not matching. Please check code and .user file.")
            return 1

    def manage_ticketing_system(self):
        # increment a ticket number shared by a number of threads and check that no active threads are running

        return 0


if __name__ == "__main__":
    assignment = Assignment(args=sys.argv[1:])
    exit_code = assignment.run()
    sys.exit(exit_code)
