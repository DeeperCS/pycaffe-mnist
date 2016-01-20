# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:40:21 2016

@author: joe
"""
import sys
sys.path.insert(0, '/home/joe/github/caffe/python')

import os,time
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import fileinput, re

from PIL import Image

import matplotlib.pyplot as plt
 
def loadData(filename):
    """
    Load data to list
 
    Parameters:
        filename: contain the file path and its label per line
    example:
    /home/joe/mnist/train/5/00000.png 5
    /home/joe/mnist/train/0/00001.png 0
    /home/joe/mnist/train/4/00002.png 4
    ...
    
    Return:
        X: file names 
        Y: labels 
      
    """
    X = []
    Y = []
    for line in fileinput.input(filename):
        entries = re.split(' ', line.strip())
        X.append(entries[0])
        Y.append(entries[1])
    return X, Y
    
    
def batch_generator(X, Y, batchSize):
    """
        Using yield style to generate batch size data
        
        Note:
            The last batch may fill a full batchsize
    """
    assert(len(X)==len(Y))
    for i in range(0, len(X), batchSize):
        nRet = min(batchSize, (len(X)-i))
        yield X[i:(i+nRet)], Y[i:(i+nRet)]


       
def loadSolver(fileName):
    """
        Load Solver file and net file in it
    """
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(solverFile).read(), solverParam)
    # net parameter
    netFile = solverParam.train_net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFile).read(), netParam)

    # model storage
    outDir = solverParam.snapshot_prefix
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
        
    return solverParam, netParam

        
def train_loop(net, XImageFileName, Labels, XImageFileNameTest, LabelsTest, doTest=True):
    
    solverParam, netParam = loadSolver(solverFile)
    # input dim
    inputDim = None
    if netParam.layer[0].type == 'MemoryData':
        batch_size = netParam.layer[0].memory_data_param.batch_size
        channels = netParam.layer[0].memory_data_param.channels
        height = netParam.layer[0].memory_data_param.height
        width = netParam.layer[0].memory_data_param.width
        inputDim = [batch_size, channels, height, width]
        
    X_batch = np.zeros(inputDim, dtype=np.float32)
    Y_batch = np.zeros(inputDim[0], dtype=np.float32)
    losses = np.zeros((solverParam.max_iter,), dtype=np.float32)
    accuries = np.zeros((solverParam.max_iter,), dtype=np.float32)
    
    for name, blobs in net.params.iteritems():
        for bIdx, b in enumerate(blobs):
            print('{}[{}] : {}'.format(name, bIdx, b.data.shape))
    
    # Iteration for a max_iter and record epoch
    currIter = 0
    currEpoch = 0  
    while currIter < solverParam.max_iter:
        #-------------------------------------------------------------------
        # iterate for a single epoch (All training data for one time)
        # The inner loop below is for a single epoch, which we may terminate
        # early if the max of iterations is reached.
        #-------------------------------------------------------------------
        currEpoch += 1    # one iteration of the iterator for a epoch
        loss_total_sigle_epoch = []   # record loss of every batch
        print('[train]: epoch {} begin'.format(currEpoch))
        epochTime = 0
        _tmp = time.time()
        iterator = batch_generator(XImageFileName, Labels, inputDim[0])
        for XImageFileName_batch, Labels_batch in iterator:
            if currIter >= solverParam.max_iter:
                break
            # Extract tiles and labels from axis information from Idx(prepare for a batch)
            for j in range(len(XImageFileName_batch)):
                X_batch[j, 0, :, :] = np.asarray(Image.open(XImageFileName_batch[j]))
                Y_batch[j] = Labels_batch[j]
            
            # when a batch data prepared, preprocess and put it to caffe
            X_batch /= 255
            solver.net.set_input_arrays(X_batch, Y_batch)
            
            # launch one step for gradient decent
            solver.step(1)
            
            # get loss and accuracy
            loss = float(solver.net.blobs['loss'].data)
            accuracy = float(solver.net.blobs['accuracy'].data)
            losses[currIter] = loss
            accuries[currIter] = accuracy
            
            loss_total_sigle_epoch.append(loss)  # Storage each batch loss to get the mean loss over the whole batch
            
            currIter += 1  # one batch for a iteration
        
        modelFileName = os.path.join(solverParam.snapshot_prefix, 'iter_{}.caffemodel'.format(currIter))
        net.save(str(modelFileName))
        epochTime += time.time() - _tmp
        print('[train]: epoch {0} finished in {1:.2f} seconds, {2:.2f} min'.format(currEpoch, epochTime, epochTime/60))
        print('[train]: loss:{}, saved {}'.format(np.mean(loss_total_sigle_epoch), modelFileName))       
        
        #-------------------------------------------------------------------
        # One epoch finished
        #-------------------------------------------------------------------
        # do Test
        #-------------------------------------------------------------------
        if doTest:
            accArray = []
            lossArray = []
            iteratorTest = batch_generator(XImageFileNameTest, LabelsTest, inputDim[0])
            for XImageFileNameTest_batch, LabelsTest_batch in iteratorTest:                       
                for j in range(len(XImageFileNameTest_batch)):
                    X_batch[j, 0, :, :] = np.asarray(Image.open(XImageFileNameTest_batch[j]))
                    Y_batch[j] = LabelsTest_batch[j]
                
                # when a batch data prepared, put it to caffe
                X_batch /= 255
                net.set_input_arrays(X_batch, Y_batch)
                
                # launch forward
                net.forward()
                
                # get loss and accuracy
                loss = float(net.blobs['loss'].data)
                accuracy = float(net.blobs['accuracy'].data)
                accArray.append(accuracy)
                lossArray.append(loss)
            print('[test]: loss:{}, acc:{}'.format(np.mean(lossArray), np.mean(accArray)))

    print('training completed')
    # return training informations
    return losses, accuries

    
if __name__=='__main__':
    
    caffe.set_mode_gpu()
    caffe.set_device(0)  
    plotResult = True    
    #######################
    # 1.load data
    #######################
    dataTxt = '../train.txt'
    XImageFileName, Labels = loadData(dataTxt)
    
    dataTestTxt = '../../test/test.txt'
    XImageFileNameTest, LabelsTest = loadData(dataTxt)
    
    #######################
    # 2. Load solver and train
    #######################
    solverFile = 'solver-mnist.prototxt'
    solver = caffe.get_solver(solverFile)
   
    losses, accuries = train_loop(solver.net, XImageFileName, Labels, XImageFileNameTest, LabelsTest)
    if plotResult:
        plt.plot(losses)
        plt.plot(accuries)