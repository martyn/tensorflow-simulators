import tensorflow as tf
import numpy as np

def eventAge(person, dt):
    person['age']+=dt
    return person

def runOnEachPerson(callback):
    def inner(world, dt):
	for person in world['people']:
	    callback(person, dt)
	return world
    return inner

events = {
	    'age':runOnEachPerson(eventAge)
	}

def stepEvent(world, event, dt):
    return events[event](world, dt)
def step(world, events):
    for event in events:
	world = stepEvent(world,event, 1)
    return world

seed = {
	'people': [{'age':0}, {'age':0}]
	}
###
#tensorflow stuff


n=250000
m=100
dims = [n,m]
worldTf = tf.Variable(tf.zeros(dims))
ones =tf.ones([n, 1])
def filterIndex(idx):
    filter = np.zeros([m])
    filter[idx]=1
    return filter
stepAgeTf = tf.ones([n,m])*filterIndex(0)
stepTf = worldTf.assign(tf.add(worldTf,stepAgeTf))


i=0
initTf=tf.initialize_all_variables()
sess = tf.Session()
print(stepAgeTf)
sess.run(initTf)
print(sess.run(stepAgeTf))
while(i<50):
    #print step(seed, ['age'])
    print(sess.run(stepTf))
    i+=1

