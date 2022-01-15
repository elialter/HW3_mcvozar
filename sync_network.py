from network import *
from my_ring_allreduce import *
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class SynchronicNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):

        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for epoch in range(self.epochs):

            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size // size)

            for x, y in mini_batches:
                # doing props
                self.forward_prop(x)
                ma_nabla_b, ma_nabla_w = self.back_prop(y)

                # summing all ma_nabla_b and ma_nabla_w to nabla_w and nabla_b
                nabla_w = []
                nabla_b = []
                # TODO: add your code
                nabla_w = [np.zeros_like(element) for element in ma_nabla_w]
                nabla_b = [np.zeros_like(element) for element in ma_nabla_b]
                for idx in range(self.num_layers):
                    ringallreduce(ma_nabla_w[idx], nabla_w[idx], comm, MPI.SUM)
                    ringallreduce(ma_nabla_b[idx], nabla_b[idx], comm, MPI.SUM)

                # End of my code
                # calculate work
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]

            self.print_progress(validation_data, epoch)

        MPI.Finalize()
