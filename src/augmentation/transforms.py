import numpy as np

def ah(a):return __import__('hashlib').sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def transform(a,m):
 a=np.rot90(a,m['rotation'])
 if m['flip0']:a=np.flip(a,0)
 if m['flip1']:a=np.flip(a,1)
 return np.ascontiguousarray(a)

def inverse(a,m):
 if m['flip1']:a=np.flip(a,1)
 if m['flip0']:a=np.flip(a,0)
 return np.ascontiguousarray(np.rot90(a,-m['rotation']))
