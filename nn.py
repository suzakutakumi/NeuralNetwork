import sys
import numpy as np
input = sys.stdin.readline


class NeuralNetwork():
    def __init__(self, lays, r):
        self.rate = r  # 学習率
        self.lays = np.array(lays)  # 層全体(2次元)
        self.h=None # hの初期化
        self.hb=None # hbの初期化
        self.lenLay = len(self.lays)  # 層の数
        # 初期化1(層数分のリスト)
        self.w = [0]
        self.b = [0]
        # 初期化2(2層目以降のw,bを標準偏差(2/層のニューロン数)のルート)
        randn = np.random.randn
        for lb, la in zip(self.lays[:-1], self.lays[1:]):
            self.w.append(np.matrix([[randn()*np.sqrt(2/lb) for _ in range(lb)]for _ in range(la)]))
            self.b.append(np.matrix([randn()*np.sqrt(2/lb) for _ in range(la)]).T)
    def softmax(self):
        lz=np.dot(self.w[-1], self.z[-1])+self.b[-1]
        __max = np.max(lz.T)
        __sum = np.sum(np.exp(lz-__max))
        self.z.append(np.exp(lz-__max)/__sum)
    def liner(self):
        self.z.append(np.dot(self.w[-1], self.z[-1])+self.b[-1])
    def Forward(self, input_,endf=softmax):
        # 入力がmatrixじゃなければ変換する
        if type(input_) != "matrix":
            input_ = np.matrix(input_).T
        # z(層の出力)を初期化
        self.z = [input_]
        # MID
        for ww,bb in zip(self.w[1:-1],self.b[1:-1]):
            wz=np.dot(ww, self.z[-1])+bb
            self.z.append(np.matrix(np.where(wz<0.0, 0.0, wz)))
        # OUT(softmax)
        endf(self)
        return self.z
    def softmaxd(self,zz,tt):
        return zz-tt.T
    def Back(self, t, zs,endfd=softmaxd):  # t:答え zs:テストデータによる層の結果
        # 答えを行列にする
        if type(t) != "matrix":
            t = [np.matrix(i)for i in t]
        # 変化量dw,dbを初期化
        dw = [0]+[np.matrix(np.zeros(n*b).reshape((n,b))) for n,b in zip(self.lays[1:],self.lays[:-1])]
        # dw[0] = 0
        db = [0]+[np.matrix(np.zeros(lay)).T for lay in self.lays[1:]]
        # db[0] = 0
        for tt,zz in zip(t,zs):
            for j in range(self.lenLay-1, 0, -1):
                if j == self.lenLay-1:
                    bef = endfd(self,zz[j],tt)
                    db[j] += bef
                    dw[j] += np.dot(bef, zz[j-1].T)
                else:
                    bef = np.multiply(
                        np.dot(self.w[j+1].T, bef), np.where(zz[j] > 0.0, 1, 0))
                    db[j] += bef
                    dw[j] += np.dot(bef, zz[j-1].T)
        if self.h==None:
            self.h=[]
            for dww in dw:
                self.h.append(np.zeros_like(dww))
        if self.hb==None:
            self.hb=[]
            for dbb in db:
                self.hb.append(np.zeros_like(dbb))
        lz=len(zs)
        self.h=[np.array(h)+np.multiply(dww,dww)/lz/lz for h,dww in zip(self.h,dw)]
        self.hb=[np.array(hb)+np.multiply(dbb,dbb)/lz/lz for hb,dbb in zip(self.hb,db)]
        self.w = list(np.array(self.w)-self.rate*np.array(dw)/np.array([np.sqrt(h)+1e-7 for h in self.h])/lz)
        self.b = list(np.array(self.b)-self.rate*np.array(db)/np.array([np.sqrt(h)+1e-7 for h in self.hb])/lz)
