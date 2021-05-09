from nn import NeuralNetwork as NN
from sklearn.datasets import load_digits as ld
import random
lay=[64,60,40,20,10]
r=0.0005

n=NN(lay,r)
x=[]
t=[]
labels=list(ld().target)
for il in list(zip(ld().images,ld().target)):
    img=il[0]
    label=il[1]
    y=[]
    for p in list(img):
        y+=list(p)
    x.append(y)
    tt=[0]*10
    tt[label]=1
    t.append(tt)
c=0
while c%150000!=0 or input()!="end":
    y=random.sample(list(zip(x,t)),50)
    z=[n.Forward(xx[0]) for xx in y]
    n.Back([tt[1]for tt in y],z)
    if (c+1)%5000==0:
        print(c)
        y=[]
        for i in range(10):
            j=random.randrange(10)
            while ld().target[j]!=i:
                j=random.randrange(10)
            a=n.Forward(x[j])[-1][i]
            print(j,a>0.7)
            y.append(a>0.7)
        if False not in y:
            print("OK")
            test=[0]*10
            Sum=[0]*10
            for xx,tt in zip(x,labels):
                Sum[tt]+=1
                test[tt]+=1 if n.Forward(xx)[-1][tt]>0.7 else 0
            for s,n in zip(Sum,test):
                print(n,"/",s,sep="")
    c+=1