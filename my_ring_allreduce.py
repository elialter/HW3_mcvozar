import numpy as np


def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    needed_receiver = int((comm_rank - 1) % comm_size)
    needed_send = int((comm_rank + 1) % comm_size)
    curr_process_copy = np.copy(send)
    arr_len = len(curr_process_copy)
    for _ in range(0, len(send) % comm_size):
        np.append(curr_process_copy, np.array([curr_process_copy[0]]))
    for idx in range(comm_size - 1):
        receive_data = np.zeros_like(np.array(curr_process_copy[((comm_rank - idx) % comm_size):arr_len-arr_len % comm_size:comm_size]))
        send_request = comm.Isend(np.array(curr_process_copy[((comm_rank - idx) % comm_size):arr_len-arr_len % comm_size:comm_size]), dest=needed_send, tag=88)
        comm.Recv(receive_data, source=needed_receiver, tag=88)
        for place in range(0, (arr_len-arr_len % comm_size) // comm_size):
            curr_process_copy[place * comm_size + (needed_receiver - idx) % comm_size] = op(curr_process_copy[place * comm_size + (needed_receiver - idx) % comm_size], receive_data[place])
        send_request.wait()
    comm.barrier()
    for idx in range(comm_size - 1):
        receive_data = np.zeros_like(np.array(curr_process_copy[(comm_rank + 1 - idx) % comm_size:arr_len-arr_len % comm_size:comm_size]))
        send_request = comm.Isend(np.array(curr_process_copy[(comm_rank + 1 - idx) % comm_size:arr_len-arr_len % comm_size:comm_size]), dest=needed_send, tag=88)
        comm.Recv(receive_data, source=needed_receiver, tag=88)
        for place in range(0, (arr_len-arr_len % comm_size)//comm_size):
            curr_process_copy[place * comm_size + (needed_receiver + 1 - idx) % comm_size] = receive_data[place]
        send_request.wait()
    comm.barrier()
    for idx in range(len(recv)):
        recv[idx] = curr_process_copy[idx]
