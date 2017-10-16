import gym
import numpy as np
import cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

#from modelAny import *

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope


env = gym.make('CartPole-v0')
# hyperparameters
hidden_n = 8 # number of hidden layer neurons
learning_rate = 1e-2
gamma = 0.99 # discount factor for reward
discount = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

modelBatchSz = 3 # Batch size when learning from model
envBatchSz = 3 # Batch size when learning from real environment

# model initialization
inDim = 4 # input dimensionality

tf.reset_default_graph()
observations = tf.placeholder(tf.float32, [None,4] , name="input_x")
layer1_Ws = tf.get_variable("layer1_Ws", shape=[4, hidden_n],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,layer1_Ws))
layer2_Ws = tf.get_variable("layer2_Ws", shape=[hidden_n, 1],
           initializer=tf.contrib.layers.xavier_initializer())
layer2_Ws = tf.matmul(layer1,layer2_Ws)
prob = tf.nn.sigmoid(layer2_Ws)

training_vars = tf.trainable_variables()
in_Y = tf.placeholder(tf.float32,[None,1], name="in_Y")
rewardSig = tf.placeholder(tf.float32,name="reward_signal")
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
layer1Grad = tf.placeholder(tf.float32,name="layer1Grad")
layer2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [layer1Grad,layer2Grad]
ll = tf.log(in_Y*(in_Y - prob) + (1 - in_Y)*(in_Y + prob))
loss = -tf.reduce_mean(ll * rewardSig) 
newGrads = tf.gradients(loss,training_vars)
updateGrads = adam.apply_gradients(zip(batchGrad,training_vars))

layerSize = 256 # model layer size

inData = tf.placeholder(tf.float32, [None, 5])
with tf.variable_scope('rnnlm'):
    softmaxWeights = tf.get_variable("softmaxWeights", [layerSize, 50])
    softmaxBias = tf.get_variable("softmaxBias", [50])

prevState = tf.placeholder(tf.float32, [None,5] , name="prevState")
ML1Ws = tf.get_variable("ML1Ws", shape=[5, layerSize],
           initializer=tf.contrib.layers.xavier_initializer())
ML1Bias = tf.Variable(tf.zeros([layerSize]),name="ML1Bias")
layer1M = tf.nn.relu(tf.matmul(prevState,ML1Ws) + ML1Bias)
ML2Ws = tf.get_variable("ML2Ws", shape=[layerSize, layerSize],
           initializer=tf.contrib.layers.xavier_initializer())
ML2Bias = tf.Variable(tf.zeros([layerSize]),name="ML2Bias")
layer2M = tf.nn.relu(tf.matmul(layer1M,ML2Ws) + ML2Bias)
weights0 = tf.get_variable("weights0", shape=[layerSize, 4],
           initializer=tf.contrib.layers.xavier_initializer())
WsR = tf.get_variable("WsR", shape=[layerSize, 1],
           initializer=tf.contrib.layers.xavier_initializer())
WsD = tf.get_variable("WsD", shape=[layerSize, 1],
           initializer=tf.contrib.layers.xavier_initializer())

bias0 = tf.Variable(tf.zeros([4]),name="bias0")
biasR = tf.Variable(tf.zeros([1]),name="biasR")
biasD = tf.Variable(tf.ones([1]),name="biasD")


pred_obsrv = tf.matmul(layer2M,weights0,name="pred_obsrv") + bias0
pred_reward = tf.matmul(layer2M,WsR,name="pred_reward") + biasR
pred_done = tf.sigmoid(tf.matmul(layer2M,WsD,name="pred_done") + biasD)

true_obsrv = tf.placeholder(tf.float32,[None,4],name="true_obsrv")
true_reward = tf.placeholder(tf.float32,[None,1],name="true_reward")
true_done = tf.placeholder(tf.float32,[None,1],name="true_done")


pred_state = tf.concat([pred_obsrv,pred_reward,pred_done], 1)

obsrv_state = tf.square(true_obsrv - pred_obsrv)

reward_loss = tf.square(true_reward - pred_reward)

done_loss = tf.multiply(pred_done, true_done) + tf.multiply(1-pred_done, 1-true_done)
done_loss = -tf.log(done_loss)

model_loss = tf.reduce_mean(obsrv_state + done_loss + reward_loss)

AdamOpt = tf.train.AdamOptimizer(learning_rate=learning_rate)
updateModel = AdamOpt.minimize(model_loss)


def resetGradBuff(buffer):
    for idx,grad in enumerate(buffer):
        buffer[idx] = grad * 0
    return buffer
        
def discountRewards(reward):
    """ take 1D float array of rewards and compute discounted reward """
    tempAdd = 0
    discntR = np.zeros_like(reward)
    for t in reversed(xrange(0, reward.size)):
        tempAdd = tempAdd * gamma + reward[t]
        discntR[t] = tempAdd
    return discntR


# This function uses our model to produce a new state when given a previous state and action
def stepModel(sess, xs, action):
    toFeed = np.reshape(np.hstack([xs[-1][0],np.array(action)]),[1,5])
    myPredict = sess.run([pred_state],feed_dict={prevState: toFeed})
    reward = myPredict[0][:,4]
    obsrv = myPredict[0][:,0:4]
    obsrv[:,0] = np.clip(obsrv[:,0],-2.4,2.4)
    obsrv[:,2] = np.clip(obsrv[:,2],-0.4,0.4)
    doneP = np.clip(myPredict[0][:,5],0,1)
    if doneP > 0.1 or len(xs)>= 300:
        done = True
    else:
        done = False
    return obsrv, reward, done

xs,drs,ys,ds = [],[],[],[]
running_reward = None
summedReward = 0
epochNum = 1
realEpoch = 1
init = tf.initialize_all_variables()
batchSize = envBatchSz

useModel = False #use model to get observations
trainModel = True 
trainPolicy = False
switchPnt = 1

# Launch the graph

with tf.Session() as sess:
    render = False
    sess.run(init)
    obsrv = env.reset()
    x = obsrv
    buffer = sess.run(training_vars)
    buffer = resetGradBuff(buffer)
    
    while epochNum <= 1000:
        # Start displaying environment once performance is acceptably high.
        if (summedReward/batchSize > 150 and useModel == False) or render == True : 
            env.render()
            render = True
            
        x = np.reshape(obsrv,[1,4])

        tfprob = sess.run(prob,feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        # record various intermediates (needed later for backprop)
        xs.append(x) 
        y = 1 if action == 0 else 0 
        ys.append(y)
        
        # step the  model or real environment and get new measurements
        if useModel == False:
            obsrv, reward, done, info = env.step(action)
        else:
            obsrv, reward, done = stepModel(sess,xs,action)
                
        summedReward += reward
        
        ds.append(done*1)
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: 
            
            if useModel == False: 
                realEpoch += 1
            epochNum += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this epoch
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs,drs,ys,ds = [],[],[],[] # reset array memory
            
            if trainModel == True:
                actions = np.array([np.abs(y-1) for y in epy][:-1])
                prevStates  = epx[:-1,:]
                prevStates  = np.hstack([prevStates ,actions])
                nextStates  = epx[1:,:]
                rewards = np.array(epr[1:,:])
                dones = np.array(epd[1:,:])
                state_nextsAll = np.hstack([nextStates ,rewards,dones])

                feed_dict={prevState: prevStates , true_obsrv: nextStates ,true_done:dones,true_reward:rewards}
                loss,pState,_ = sess.run([model_loss,pred_state,updateModel],feed_dict)
            if trainPolicy == True:
                discntEpochRewards = discountRewards(epr).astype('float32')
                discntEpochRewards -= np.mean(discntEpochRewards)
                discntEpochRewards /= np.std(discntEpochRewards)
                tGrad = sess.run(newGrads,feed_dict={observations: epx, in_Y: epy, rewardSig: discntEpochRewards})
                
                # If gradients becom too large, end training process
                if np.sum(tGrad[0] == tGrad[0]) == 0:
                    break
                for idx,grad in enumerate(tGrad):
                    buffer[idx] += grad
                
            if switchPnt + batchSize == epochNum: 
                switchPnt = epochNum
                if trainPolicy == True:
                    sess.run(updateGrads,feed_dict={layer1Grad: buffer[0],layer2Grad:buffer[1]})
                    buffer = resetGradBuff(buffer)

                running_reward = summedReward if running_reward is None else running_reward * 0.99 + summedReward * 0.01
                if useModel == False:
                    print 'Performance: Epoch {0}, Reward {1:0.4f}, Action: {2}, Avg Reward {3:0.4f}'.format(realEpoch,summedReward/envBatchSz,action, float(running_reward/envBatchSz))
                    if summedReward/batchSize > 200:
                        break
                summedReward = 0

                # Once the model has been trained on 100 episodes, we start alternating between training the policy
                # from the model and training the model from the real environment.
                if epochNum > 100:
                    useModel = not useModel
                    trainModel = not trainModel
                    trainPolicy = not trainPolicy
            
            if useModel == True:
                obsrv = np.random.uniform(-0.1,0.1,[4]) # Generate reasonable starting point
                batchSize = modelBatchSz
            else:
                obsrv = env.reset()
                batchSize = envBatchSz
                
print realEpoch

print pState
plt.figure(figsize=(8, 12))
for i in range(6):
    plt.subplot(6, 2, 2*i + 1)
    plt.plot(pState[:,i])
    plt.subplot(6,2,2*i+1)
    plt.plot(state_nextsAll[:,i])
plt.tight_layout()
plt.show()