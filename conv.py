""" COnvolutional Neural network learns a policy for pong. Uses OpenAI Gym. WORKS!~! """
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
import numpy as np
import theano 
import theano.tensor as T
import gym
import cPickle as pickle
import matplotlib.pyplot as plt
# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 1 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False
plot_grads=False
# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid

stopped=False # this is for automatically restarting when theano crashes
  

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]


def show_inputs(xs):
  for i in xs:
    plt.imshow(i.reshape(80,80),cmap='gray')
    plt.pause(0.01)

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).reshape(1,1,80,80)

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  #//TODO: speed things up by discounting the rewards using theano
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


def split(a,x,r):
  """Splits pixels x, rewards r, actions a into 2 minibatches so I don't crash gpu memory """
  assert a.shape[0]==x.shape[0]==r.shape[0]
  h=int(a.shape[0]/2)
  a1,x1,r1=a[:h],x[:h],r[:h]
  a2,x2,r2=a[h:],x[h:],r[h:]
  return a1,x1,r1,a2,x2,r2





##############################LASAGNE STUFF###########################################################
##############################LASAGNE STUFF###########################################################
env = gym.make("Pong-v0")
observation = env.reset()
#theano variables
input_var=T.tensor4('X')
a_var=T.vector('y')#fake label / 1 if action taken is up, 0 down
reward_var = T.vector('rewarD')#discounted reward at time t
param_var = T.matrix('generic param')

if resume:
  l_out = pickle.load(open('params.pickle', 'rb'))
else:
  l=InputLayer(input_var=input_var,shape=(None,1,80,80))
  l=ConvLayer(l,num_filters=32,filter_size=(8,8),nonlinearity=lasagne.nonlinearities.elu,stride=(3,3))
  l=ConvLayer(l,num_filters=64,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.elu,stride=(2,2))
  #l=ConvLayer(l,num_filters=64,filter_size=(3,3),nonlinearity=lasagne.nonlinearities.elu,stride=(1,1))
  l=DenseLayer(l,num_units=260,nonlinearity=lasagne.nonlinearities.elu)
  l_out=DenseLayer(l,num_units=1,nonlinearity=None)




params = lasagne.layers.get_all_params(l_out, trainable=True)



out=lasagne.layers.get_output(l_out,input_var).T
out=T.nnet.sigmoid(out)



pixels_to_prob = theano.function(
  inputs = [input_var],
  outputs = out,
  allow_input_downcast = True,
  name = 'pixels to prob'
  )

#this takes a x,reward,action taken 
rlogp = -2*(reward_var*((a_var-out)**2))
#rlogp*=(-2*a_var+1)
#rlogp=(rlogp**2)/2
rlogp=rlogp.sum()
rlogp1=reward_var*((a_var -out))

train_fn = theano.function(
  inputs = [input_var,reward_var,a_var], 
  outputs = rlogp,
  updates = lasagne.updates.rmsprop(rlogp,params,learning_rate=-1e-4,rho=decay_rate),
  name = 'train function',
  allow_input_downcast = True
  )

get_updates = theano.function(
  inputs = [input_var,reward_var,a_var], 
  outputs = T.grad(
    cost = rlogp,
    wrt = lasagne.layers.get_all_params(l_out),
    consider_constant=[reward_var,a_var,]
    ),
  name = '@@get gradients/updates function@@',
  allow_input_downcast = True)

#s=lasagne.updates.sgd((reward_var*(a_var-out)).mean(),params,learning_rate=1)

epdlogp_test=theano.function(
  inputs = [input_var,reward_var,a_var], 
  outputs = rlogp1,
  name = 'train function',
  allow_input_downcast = True
  )

##############################/LASAGNE STUFF###########################################################
##############################/LASAGNE STUFF###########################################################

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 7454
plt.ion()

plt.title('conv 32/8x8/3,3->conv64/3x3/2,2->fc260 elu LEARNING CURVE')
a_l=[]
print("starting training")
running_reward=-12.51
while True:
  if render: env.render()
  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros((1,1,80,80))
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob=pixels_to_prob(x.reshape(1,1,80,80))[0]
  #if np.absolute(aprob - aprob1) > 1e-15:
  #  print('!!!!!!!!!!!!')
  #  print("policy:{}, gpu policy:{}".format(aprob,aprob1))
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  y = 1 if action == 2 else 0 # a "fake label"
  a_l.append(y)
  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    epr = np.vstack(drs)
    xs,hs,drs = [],[],[] # reset array memory
    a_l=np.asarray(a_l)
    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    discounted_epr = discounted_epr.reshape(discounted_epr.shape[0],)

    try:
      if epx.shape[0] > 5000:
        print("big episode, {} frames. splitting!".format(epx.shape[0]))
        h=int(epx.shape[0]/2)
        grad_g=get_updates(epx[:h],discounted_epr[:h],a_l[:h])
        gg1=get_updates(epx[:h],discounted_epr[:h],a_l[:h])
        for p in range(len(grad_g)):
          grad_g[p]+=gg1[p]
      else:
        grad_g=get_updates(epx,discounted_epr,a_l)
        gg1=None
    except MemoryError:
      #if theano runs out of memory and dies, save the params and rerun the code!
      print("ran out of memory, restarting")
      #done=False#... or just try again
      print("..............................")
      pickle.dump(l_out,open('params.pickle','wb'))
      reward_sum=0
      observation = env.reset() # reset env
      prev_x = None
      a_l=[]
      continue
      #execfile('conv.py')

    #_=train_fn(epx,discounted_epr,a_l)
    for w in range(len(params)):
      #<3 ordered dicts thank you lasagne
      params[w].set_value(params[w].get_value()+learning_rate*grad_g[w].__array__())
    a_l=[]

    # perform rmsprop parameter update every batch_size episodes
    
    # boring book-keeping
    running_reward = -20.49 if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'episode {}: reward total was {}. running mean: {}'.format(episode_number,reward_sum, running_reward)
    #if episode_number % 100 == 0: pickle.dump(model, open('save2.pickle', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None
    if episode_number % 80 == 0:
      plt.scatter(episode_number,running_reward+21,color='g',marker='.')
      plt.pause(0.001)

