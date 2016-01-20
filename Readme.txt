# Using PyCaffe and MemoryData Layer to train mnist

I0120 14:50:17.284659   499 solver.cpp:47] Initializing solver from parameters: 
train_net: "net-mnist-train.prototxt"
base_lr: 0.01
display: 999999999
max_iter: 10000
lr_policy: "inv"
gamma: 0.0001
power: 0.75
momentum: 0.9
weight_decay: 0.0005
snapshot: 999999999
snapshot_prefix: "./models/"
I0120 14:50:17.284730   499 solver.cpp:80] Creating training net from train_net file: net-mnist-train.prototxt
I0120 14:50:17.284945   499 net.cpp:49] Initializing net from parameters: 
name: "Train"
state {
  phase: TRAIN
}
layer {
  name: "data"
  type: "MemoryData"
  top: "data"
  top: "label"
  memory_data_param {
    batch_size: 50
    channels: 1
    height: 28
    width: 28
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 50
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "pool2"
  top: "ip1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
I0120 14:50:17.284992   499 layer_factory.hpp:76] Creating layer data
I0120 14:50:17.285001   499 net.cpp:106] Creating Layer data
I0120 14:50:17.285006   499 net.cpp:411] data -> data
I0120 14:50:17.285013   499 net.cpp:411] data -> label
I0120 14:50:17.285154   499 net.cpp:150] Setting up data
I0120 14:50:17.285161   499 net.cpp:157] Top shape: 50 1 28 28 (39200)
I0120 14:50:17.285166   499 net.cpp:157] Top shape: 50 (50)
I0120 14:50:17.285168   499 net.cpp:165] Memory required for data: 157000
I0120 14:50:17.285172   499 layer_factory.hpp:76] Creating layer label_data_1_split
I0120 14:50:17.285177   499 net.cpp:106] Creating Layer label_data_1_split
I0120 14:50:17.285181   499 net.cpp:454] label_data_1_split <- label
I0120 14:50:17.285184   499 net.cpp:411] label_data_1_split -> label_data_1_split_0
I0120 14:50:17.285190   499 net.cpp:411] label_data_1_split -> label_data_1_split_1
I0120 14:50:17.285214   499 net.cpp:150] Setting up label_data_1_split
I0120 14:50:17.285219   499 net.cpp:157] Top shape: 50 (50)
I0120 14:50:17.285223   499 net.cpp:157] Top shape: 50 (50)
I0120 14:50:17.285225   499 net.cpp:165] Memory required for data: 157400
I0120 14:50:17.285228   499 layer_factory.hpp:76] Creating layer conv1
I0120 14:50:17.285234   499 net.cpp:106] Creating Layer conv1
I0120 14:50:17.285238   499 net.cpp:454] conv1 <- data
I0120 14:50:17.285241   499 net.cpp:411] conv1 -> conv1
I0120 14:50:17.285960   499 cudnn_conv_layer.cpp:194] Reallocating workspace storage: 20664
I0120 14:50:17.285979   499 net.cpp:150] Setting up conv1
I0120 14:50:17.285984   499 net.cpp:157] Top shape: 50 20 24 24 (576000)
I0120 14:50:17.285987   499 net.cpp:165] Memory required for data: 2461400
I0120 14:50:17.285995   499 layer_factory.hpp:76] Creating layer pool1
I0120 14:50:17.286001   499 net.cpp:106] Creating Layer pool1
I0120 14:50:17.286005   499 net.cpp:454] pool1 <- conv1
I0120 14:50:17.286010   499 net.cpp:411] pool1 -> pool1
I0120 14:50:17.286169   499 net.cpp:150] Setting up pool1
I0120 14:50:17.286176   499 net.cpp:157] Top shape: 50 20 12 12 (144000)
I0120 14:50:17.286180   499 net.cpp:165] Memory required for data: 3037400
I0120 14:50:17.286183   499 layer_factory.hpp:76] Creating layer conv2
I0120 14:50:17.286190   499 net.cpp:106] Creating Layer conv2
I0120 14:50:17.286192   499 net.cpp:454] conv2 <- pool1
I0120 14:50:17.286197   499 net.cpp:411] conv2 -> conv2
I0120 14:50:17.286859   499 cudnn_conv_layer.cpp:194] Reallocating workspace storage: 10272
I0120 14:50:17.286877   499 net.cpp:150] Setting up conv2
I0120 14:50:17.286882   499 net.cpp:157] Top shape: 50 50 8 8 (160000)
I0120 14:50:17.286886   499 net.cpp:165] Memory required for data: 3677400
I0120 14:50:17.286893   499 layer_factory.hpp:76] Creating layer pool2
I0120 14:50:17.286898   499 net.cpp:106] Creating Layer pool2
I0120 14:50:17.286901   499 net.cpp:454] pool2 <- conv2
I0120 14:50:17.286906   499 net.cpp:411] pool2 -> pool2
I0120 14:50:17.287060   499 net.cpp:150] Setting up pool2
I0120 14:50:17.287067   499 net.cpp:157] Top shape: 50 50 4 4 (40000)
I0120 14:50:17.287070   499 net.cpp:165] Memory required for data: 3837400
I0120 14:50:17.287073   499 layer_factory.hpp:76] Creating layer ip1
I0120 14:50:17.287080   499 net.cpp:106] Creating Layer ip1
I0120 14:50:17.287082   499 net.cpp:454] ip1 <- pool2
I0120 14:50:17.287087   499 net.cpp:411] ip1 -> ip1
I0120 14:50:17.289391   499 net.cpp:150] Setting up ip1
I0120 14:50:17.289402   499 net.cpp:157] Top shape: 50 500 (25000)
I0120 14:50:17.289404   499 net.cpp:165] Memory required for data: 3937400
I0120 14:50:17.289412   499 layer_factory.hpp:76] Creating layer relu1
I0120 14:50:17.289417   499 net.cpp:106] Creating Layer relu1
I0120 14:50:17.289420   499 net.cpp:454] relu1 <- ip1
I0120 14:50:17.289425   499 net.cpp:397] relu1 -> ip1 (in-place)
I0120 14:50:17.289569   499 net.cpp:150] Setting up relu1
I0120 14:50:17.289577   499 net.cpp:157] Top shape: 50 500 (25000)
I0120 14:50:17.289579   499 net.cpp:165] Memory required for data: 4037400
I0120 14:50:17.289582   499 layer_factory.hpp:76] Creating layer ip2
I0120 14:50:17.289597   499 net.cpp:106] Creating Layer ip2
I0120 14:50:17.289600   499 net.cpp:454] ip2 <- ip1
I0120 14:50:17.289604   499 net.cpp:411] ip2 -> ip2
I0120 14:50:17.289700   499 net.cpp:150] Setting up ip2
I0120 14:50:17.289705   499 net.cpp:157] Top shape: 50 10 (500)
I0120 14:50:17.289707   499 net.cpp:165] Memory required for data: 4039400
I0120 14:50:17.289712   499 layer_factory.hpp:76] Creating layer ip2_ip2_0_split
I0120 14:50:17.289716   499 net.cpp:106] Creating Layer ip2_ip2_0_split
I0120 14:50:17.289719   499 net.cpp:454] ip2_ip2_0_split <- ip2
I0120 14:50:17.289723   499 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
I0120 14:50:17.289728   499 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
I0120 14:50:17.289751   499 net.cpp:150] Setting up ip2_ip2_0_split
I0120 14:50:17.289754   499 net.cpp:157] Top shape: 50 10 (500)
I0120 14:50:17.289757   499 net.cpp:157] Top shape: 50 10 (500)
I0120 14:50:17.289760   499 net.cpp:165] Memory required for data: 4043400
I0120 14:50:17.289762   499 layer_factory.hpp:76] Creating layer accuracy
I0120 14:50:17.289767   499 net.cpp:106] Creating Layer accuracy
I0120 14:50:17.289769   499 net.cpp:454] accuracy <- ip2_ip2_0_split_0
I0120 14:50:17.289773   499 net.cpp:454] accuracy <- label_data_1_split_0
I0120 14:50:17.289777   499 net.cpp:411] accuracy -> accuracy
I0120 14:50:17.289783   499 net.cpp:150] Setting up accuracy
I0120 14:50:17.289786   499 net.cpp:157] Top shape: (1)
I0120 14:50:17.289788   499 net.cpp:165] Memory required for data: 4043404
I0120 14:50:17.289791   499 layer_factory.hpp:76] Creating layer loss
I0120 14:50:17.289795   499 net.cpp:106] Creating Layer loss
I0120 14:50:17.289798   499 net.cpp:454] loss <- ip2_ip2_0_split_1
I0120 14:50:17.289803   499 net.cpp:454] loss <- label_data_1_split_1
I0120 14:50:17.289805   499 net.cpp:411] loss -> loss
I0120 14:50:17.289811   499 layer_factory.hpp:76] Creating layer loss
I0120 14:50:17.289986   499 net.cpp:150] Setting up loss
I0120 14:50:17.289994   499 net.cpp:157] Top shape: (1)
I0120 14:50:17.289996   499 net.cpp:160]     with loss weight 1
I0120 14:50:17.290004   499 net.cpp:165] Memory required for data: 4043408
I0120 14:50:17.290006   499 net.cpp:226] loss needs backward computation.
I0120 14:50:17.290009   499 net.cpp:228] accuracy does not need backward computation.
I0120 14:50:17.290014   499 net.cpp:226] ip2_ip2_0_split needs backward computation.
I0120 14:50:17.290016   499 net.cpp:226] ip2 needs backward computation.
I0120 14:50:17.290019   499 net.cpp:226] relu1 needs backward computation.
I0120 14:50:17.290022   499 net.cpp:226] ip1 needs backward computation.
I0120 14:50:17.290025   499 net.cpp:226] pool2 needs backward computation.
I0120 14:50:17.290027   499 net.cpp:226] conv2 needs backward computation.
I0120 14:50:17.290030   499 net.cpp:226] pool1 needs backward computation.
I0120 14:50:17.290033   499 net.cpp:226] conv1 needs backward computation.
I0120 14:50:17.290036   499 net.cpp:228] label_data_1_split does not need backward computation.
I0120 14:50:17.290040   499 net.cpp:228] data does not need backward computation.
I0120 14:50:17.290042   499 net.cpp:270] This network produces output accuracy
I0120 14:50:17.290045   499 net.cpp:270] This network produces output loss
I0120 14:50:17.290052   499 net.cpp:283] Network initialization done.
I0120 14:50:17.290081   499 solver.cpp:59] Solver scaffolding done.
conv1[0] : (20, 1, 5, 5)
conv1[1] : (20,)
conv2[0] : (50, 20, 5, 5)
conv2[1] : (50,)
ip1[0] : (500, 800)
ip1[1] : (500,)
ip2[0] : (10, 500)
ip2[1] : (10,)
[train]: epoch 1 begin
I0120 14:50:17.304584   499 solver.cpp:236] Iteration 0, loss = 2.46
I0120 14:50:17.304606   499 solver.cpp:252]     Train net output #0: accuracy = 0.04
I0120 14:50:17.304613   499 solver.cpp:252]     Train net output #1: loss = 2.46 (* 1 = 2.46 loss)
I0120 14:50:17.304620   499 sgd_solver.cpp:106] Iteration 0, lr = 0.01
[train]: epoch 1 finished in 9.26 seconds, 0.15 min
[train]: loss:0.163135264422, saved ./models/iter_1200.caffemodel
[test]: loss:0.0884914355727, acc:0.971483334849
[train]: epoch 2 begin
[train]: epoch 2 finished in 9.13 seconds, 0.15 min
[train]: loss:0.0507860643595, saved ./models/iter_2400.caffemodel
[test]: loss:0.0375097152981, acc:0.988366671155
[train]: epoch 3 begin
[train]: epoch 3 finished in 9.14 seconds, 0.15 min
[train]: loss:0.0327808510058, saved ./models/iter_3600.caffemodel
[test]: loss:0.0258957033076, acc:0.992050004154
[train]: epoch 4 begin
[train]: epoch 4 finished in 9.23 seconds, 0.15 min
[train]: loss:0.0230981866306, saved ./models/iter_4800.caffemodel
[test]: loss:0.0204429812449, acc:0.993566670269
[train]: epoch 5 begin
[train]: epoch 5 finished in 9.17 seconds, 0.15 min
[train]: loss:0.0168845843832, saved ./models/iter_6000.caffemodel
[test]: loss:0.0164323287395, acc:0.994883336872
[train]: epoch 6 begin
[train]: epoch 6 finished in 9.23 seconds, 0.15 min
[train]: loss:0.0128102470961, saved ./models/iter_7200.caffemodel
[test]: loss:0.013231988512, acc:0.996083336522
[train]: epoch 7 begin
[train]: epoch 7 finished in 9.09 seconds, 0.15 min
[train]: loss:0.0101085660943, saved ./models/iter_8400.caffemodel
[test]: loss:0.0109632509301, acc:0.996983335863
[train]: epoch 8 begin
[train]: epoch 8 finished in 9.08 seconds, 0.15 min
[train]: loss:0.00837586788708, saved ./models/iter_9600.caffemodel
[test]: loss:0.00973979995804, acc:0.997233335723
[train]: epoch 9 begin
[train]: epoch 9 finished in 3.02 seconds, 0.05 min
[train]: loss:0.00680024245677, saved ./models/iter_10000.caffemodel
[test]: loss:0.00701264561313, acc:0.998350001276
training completed
