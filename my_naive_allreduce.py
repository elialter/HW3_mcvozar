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
    rank = comm.Get_rank()
    size = comm.Get_size()
    reciveData = np.empty_like(send)
    for i in range(len(send)):
        recv[i] = send[i]
    comm.barrier()

    for i in range(size):
        if (i == rank):
            continue
        comm.Isend(send, dest=i, tag=88)

    for i in range(size):
        if (i == rank):
            continue
        comm.Recv(reciveData, source=i, tag=88)
        for j in range(reciveData.shape[0]):
            recv[j] = op(recv[j], reciveData[j])

 #   raise NotImplementedError("To be implemented")
