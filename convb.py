""" Residual Neural network learns a policy for pong. Uses OpenAI Gym. WORKS!~! """
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm, ExpressionLayer,PadLayer,NonlinearityLayer,ElemwiseSumLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import rectify, very_leaky_rectify
import numpy as np
import theano 
import theano.tensor as T
import gym
import dill as pickle
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(99999)
# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 1 # every how many episodes to do a param update?
learning_rate =9e-6
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False
plot_every=1#plot every x episodes
# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
save_file = 'resp1.pickle'
  

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


def split(a,x,r,train_fn,m=2500):
  """Splits pixels x, rewards r, actions a into 2 minibatches so I don't crash gpu memory 

       splits into minibatches of size m"""
  L=a.shape[0]#number of frames in episode
  if L <= m:
    _=train_fn(x,r,a)
    return _
  else:
    v=int(L/m)
    diff=L%m
    print("splitting {} frames into {} minibatches of size {} and 1 of size {}".format(L,v,m,diff))
    
    for h in range(v):
      _=train_fn(x[h*m:(h+1)*m],r[h*m:(h+1)*m],a[h*m:(h+1)*m])

    _=train_fn(x[-diff:],r[-diff:],a[-diff:])
    return _



def build_resnet(input_var):
  print("building resNet")
  def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu')))
        
        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
        
        return block
  l_in=InputLayer(input_var=input_var,shape=(None,1,80,80))
  l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(2,2), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu')))
  l=residual_block(l,increase_dim=True)
  l=residual_block(l,increase_dim=True)
  l=residual_block(l,increase_dim=True)
  l=residual_block(l,increase_dim=False)
  #l=residual_block(l,increase_dim=True)
  l=DenseLayer(l,num_units=512,nonlinearity=rectify)
  l_out=DenseLayer(l,num_units=1,nonlinearity=None)
  return l_out

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
  l_out = pickle.load(open(save_file, 'rb'))
  print("loaded network successfully")
else:
  l_out=build_resnet(input_var)




params = lasagne.layers.get_all_params(l_out)



out=lasagne.layers.get_output(l_out,input_var,deterministic=True).T
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
  updates = lasagne.updates.adadelta(rlogp,params,learning_rate=-2e-6),#,rho=decay_rate),
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
reward_sum = 0
episode_number = 6869
plt.ion()

plt.title('resnet double residual block NOTE: using adam lr 9e-6')
plt.xlabel('episodes')
plt.ylabel('average games won / episode')
a_l=[]
print("starting training")
running_reward=9.082
last_100=[0]
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
  observation, reward, done, info = env.step(action)
  reward_sum += reward


  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
  if len(drs) > 40000:
    print("LONG EPISODE, BREAKING to save memory")
    done=True
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
      _=split(a_l,epx,discounted_epr,train_fn,m=2000)


      #if epx.shape[0] > 2000:
      #  print("big episode, {} frames. splitting!".format(epx.shape[0]))
      #  h=int(epx.shape[0]/3)
      #  _=train_fn(epx[:h],discounted_epr[:h],a_l[:h])
      #  _=train_fn(epx[h:],discounted_epr[h:],a_l[h:])
      #  
      #else:
      #  _=train_fn(epx,discounted_epr,a_l)
    except MemoryError:
      #if theano runs out of memory and dies, save the params and rerun the code!
      print("ran out of memory, restarting")
      #done=False#... or just try again
      print("..............................")
      pickle.dump(l_out,open(save_file,'wb'))
      reward_sum=0
      observation = env.reset() # reset env
      prev_x = None
      a_l=[]
      continue
      #execfile('conv.py')
    a_l=[]

    #_=train_fn(epx,discounted_epr,a_l)
   
    # perform rmsprop parameter update every batch_size episodes
    
    # boring book-keeping
    running_reward = -20.49 if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    last_100.append(reward_sum)
    if len(last_100)>100:
      _=last_100.pop(0)
    mean100 = np.asarray(last_100).mean()

    print( '| ep {0} | reward sum: {1:.0f} | running av: {2:.3f} | mean from prev 100:{3:.1f} |'.format(
          episode_number,  reward_sum, running_reward,mean100 ))
    #if episode_number % 100 == 0: pickle.dump(model, open('save2.pickle', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None
    if episode_number % plot_every == 0:
      plt.scatter(episode_number,running_reward+21,color='b',marker='.')
      #if len(last_100) > 99:
      #  plt.scatter(episode_number,mean100+21,color='m',marker='.')

      plt.pause(0.001)
    if episode_number % 100 == 0:
      print('saving params')
      pickle.dump(l_out,open(save_file,'wb'))
      print("DONE")
      print("saving plot")
      plt.savefig('resnet.png')
      print("DONE")