from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of workers and masters
        self.num_masters = number_of_masters

    def fit(self, training_data, validation_data=None):
        # MPI setup
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = self.size - self.num_masters

    self.layers_per_master = self.num_layers // self.num_masters

    # split up work
    if self.rank < self.num_masters:
        self.do_master(validation_data)
    else:
        self.do_worker(training_data)

    # when all is done
    self.comm.Barrier()
    MPI.Finalize()


def do_worker(self, training_data):
    """
    worker functionality
    :param training_data: a tuple of data and labels to train the NN with
    """
    # setting up the number of batches the worker should do every epoch
    # TODO: add your code
    ###
    self.number_of_batches = len(range(self.rank - self.num_masters, self.number_of_batches, self.num_workers))
    ###

    for epoch in range(self.epochs):
        # creating batches for epoch
        data = training_data[0]
        labels = training_data[1]
        mini_batches = self.create_batches(data, labels, self.mini_batch_size)
        for x, y in mini_batches:
            # do work - don't change this
            self.forward_prop(x)
            nabla_b, nabla_w = self.back_prop(y)

            # send nabla_b, nabla_w to masters
            # TODO: add your code
            ###
            send_buffer = []
            for dest in range(self.num_masters):
                for tag in range(dest, self.num_layers, self.num_masters):
                    send_buffer.append(self.comm.Isend(np.copy(nabla_b[tag]), dest=dest, tag=tag))
                    send_buffer.append(self.comm.Isend(np.copy(nabla_w[tag]), dest=dest, tag=(tag + self.num_layers)))
            ###

            # recieve new self.weight and self.biases values from masters
            # TODO: add your code
            ###
            receive_buffer = []
            for dest in range(self.num_masters):
                for tag in range(dest, self.num_layers, self.num_masters):
                    receive_buffer.append(
                        self.comm.Irecv(self.biases[tag], source=dest, tag=(5 * self.num_layers + tag)))
                    receive_buffer.append(self.comm.Irecv(self.weights[tag], source=dest,
                                                          tag=(5 * self.num_layers + tag + self.num_layers)))
            for req in send_buffer:
                req.wait()
            for req in receive_buffer:
                req.wait()
            ###


def do_master(self, validation_data):
    """
    master functionality
    :param validation_data: a tuple of data and labels to train the NN with
    """
    # setting up the layers this master does
    nabla_w = []
    nabla_b = []
    for i in range(self.rank, self.num_layers, self.num_masters):
        nabla_w.append(np.zeros_like(self.weights[i]))
        nabla_b.append(np.zeros_like(self.biases[i]))

    for epoch in range(self.epochs):
        for batch in range(self.number_of_batches):

            # wait for any worker to finish batch and
            # get the nabla_w, nabla_b for the master's layers
            # TODO: add your code
            ###
            nabla_b = []
            nabla_w = []
            MPI_status = MPI.Status()
            receive_buffer = []
            self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=MPI_status)
            source_worker = MPI_status.Get_source()
            for tag in range(self.rank, self.num_layers, self.num_masters):
                nabla_b.append(np.zeros_like(self.biases[tag]))
                receive_buffer.append(self.comm.Irecv(np.zeros_like(self.biases[tag]), source=source_worker, tag=tag))
                nabla_w.append(np.zeros_like(self.weights[tag]))
                receive_buffer.append(self.comm.Irecv(np.zeros_like(self.weights[tag]), source=source_worker,
                                                      tag=(tag + self.num_layers)))
            for req in receive_buffer:
                req.wait()
            ###

            # calculate new weights and biases (of layers in charge)
            for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                self.weights[i] = self.weights[i] - self.eta * dw
                self.biases[i] = self.biases[i] - self.eta * db

            # send new values (of layers in charge)
            # TODO: add your code
            ###
            send_buffer = []
            for idx in range(self.rank, self.num_layers, self.num_masters):
                send_buffer.append(
                    self.comm.Isend(np.copy(self.biases[idx]), dest=worker, tag=(5 * self.num_layers + idx)))
                send_buffer.append(self.comm.Isend(np.copy(self.weights[idx]), dest=worker,
                                                   tag=(5 * self.num_layers + idx + self.num_layers)))
            for req in send_buffer:
                req.wait()
            ###

        self.print_progress(validation_data, epoch)

    # gather relevant weight and biases to process 0
    # TODO: add your code
    ###
    if self.rank != 0:
        result_send_buffer = []
        for idx in range(self.rank, self.num_layers, self.num_masters):
            result_send_buffer.append(
                self.comm.Isend(np.copy(self.biases[idx]), dest=0, tag=(10 * self.num_layers + idx)))
            result_send_buffer.append(
                self.comm.Isend(np.copy(self.weights[idx]), dest=0, tag=(10 * self.num_layers + idx + self.num_layers)))
        for req in result_send_buffer:
            req.wait()

    else:
        result_receive_buffer = []
        for idx in range(0, self.num_layers):
            if j % self.num_masters != 0:
                result_receive_buffer.append(
                    self.comm.Irecv(self.biases[idx], source=MPI.ANY_SOURCE, tag=(10 * self.num_layers + idx)))
                result_receive_buffer.append(self.comm.Irecv(self.weights[idx], source=MPI.ANY_SOURCE,
                                                             tag=(10 * self.num_layers + idx + self.num_layers)))
        for req in result_receive_buffer:
            req.wait()
    ###


