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

    size = comm.Get_size()
    rank = comm.Get_rank()
    recive_f = int((rank - 1) % size)
    send_to = int((rank + 1) % size)
    arr = np.copy(send)
    for _ in range(0, len(send)%size):
        np.append(arr, np.array([arr[0]]))

    arr_len = len(arr)

    for i in range(size - 1):
        send_offset = (rank - i) % size
        recive_offset = (recive_f - i) % size
        send_data = np.array(arr[send_offset:arr_len-arr_len%size:size])
        recive_data = np.zeros_like(send_data)

        req = comm.Isend(send_data, dest=send_to, tag=88)
        comm.Recv(recive_data, source=recive_f, tag=88)

        for j in range(0, (arr_len-arr_len % size)//size):
            arr[j*size+recive_offset] = op(arr[j*size+recive_offset], recive_data[j])
        req.wait()

    comm.barrier()

    for i in range(size - 1):
        send_offset = (rank + 1 - i) % size
        recive_offset = (recive_f + 1 - i) % size
        send_data = np.array(arr[send_offset:arr_len-arr_len % size:size])
        recive_data = np.zeros_like(send_data)

        req = comm.Isend(send_data, dest=send_to, tag=88)
        comm.Recv(recive_data, source=recive_f, tag=88)

        for j in range(0, (arr_len-arr_len % size)//size):
            arr[j*size+recive_offset] = recive_data[j]
        req.wait()

    comm.barrier()

    for i in range(len(recv)):
        recv[i] = arr[i]


#    raise NotImplementedError("To be implemented")
