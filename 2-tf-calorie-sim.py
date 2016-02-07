import tensorflow as tf
import numpy as np

n=1
m=4
dims = [n,m]
worldTf = tf.Variable(tf.zeros(dims))

ones =tf.ones([n, 1])
def filterIndex(idx):
    filter = np.zeros([m])
    filter[idx]=1
    return filter
stepAgeTf = tf.ones(dims)*filterIndex(0)

deltaCaloriesTf = tf.random_normal(dims)*filterIndex(1)*100

deltaCaloriesSlice=tf.slice(worldTf, [0,1], [n,1])
caloriesTf = (tf.zeros(dims)+deltaCaloriesSlice)*filterIndex(2)

caloriesSlice = tf.slice(worldTf, [0,2], [n,1])
caloriesMod = 0.01
ageSlice = tf.slice(worldTf, [0,0], [n,1])
ageTf = 100-ageSlice

currentWeight = caloriesSlice*caloriesMod-ageTf
weightTf = (currentWeight+100)*filterIndex(3)

ops = [stepAgeTf,deltaCaloriesTf,caloriesTf, weightTf]
stepTf={}
i=0
for op in ops:
    stepTf[i] = worldTf.assign(tf.add(worldTf,op))
    i+=1


i=0
initTf=tf.initialize_all_variables()
sess = tf.Session()
sess.run(initTf)
print(['age','dcal','cal','weight'])
while(i<5):
    step=None
    for key in stepTf:
	step=sess.run(stepTf[key])
    print(step)
    i+=1

