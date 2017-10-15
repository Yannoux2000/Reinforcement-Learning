import subprocess
import time

start = time.time()

def timer(start,end):
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

i = 0

for learning_rate in [5,1,0.5,0.05]:
	for intermediate_layer in [0]:
		for gamma in [0.95,0.97,0.98,0.99]:
			for beta in [0,0.135,-0.135,-0.25]:
				subprocess.call(["python","ActorCritic.py",str(learning_rate),str(intermediate_layer),str(gamma),str(beta)])
				i+=1
				current = time.time()
				timer(start, current)


print "### {} Tentatives ###".format(i)
timer(start,time.time())