import numpy as np


def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    needed_data = np.empty_like(send)
    for i in range(len(send)):
        recv[i] = send[i]
    comm.barrier()

    for i in range(comm_size):
        if (i == comm_rank):
            continue
        comm.Isend(send, dest=i, tag=88)

    for i in range(comm_size):
        if (i == comm_rank):
            continue
        comm.Recv(needed_data, source=i, tag=88)
        for j in range(needed_data.shape[0]):
            recv[j] = op(recv[j], needed_data[j])

 #   raise NotImplementedError("To be implemented")
