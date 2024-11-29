import numpy as np

voxel=[0.62109375, 0.62109375, 0.8]

a= [[361,245,151],
 [181,236,109],
 [313,193, 84],
 [130,286, 59],
 [338,283, 94],
 [145,254,154],
 [225,226, 97],
 [271,186,153],
 [326,229, 60],
 [283,227, 71],
 [219,260,153],
 [169,285, 70],
 [382,281,105],
 [358,319,117],
 [195,194,117],
 [150,324, 80]]


b=[[363, 68, 87],
 [266, 70, 67],
 [159, 67, 71],
 [264, 68,143]]

d=[]
Mobile_mode=[]


for i in b:
    for j in a:
        p1 = np.array(i)
        p2 = np.array(j)
        Mobile_mode.append(p2-p1)
        d.append(int(np.linalg.norm(p1 - p2)))
        
# 使用列表推导式分割列表
chunks = [d[i:i + 16] for i in range(0, len(d), 16)]
chunks_modile = [Mobile_mode[i:i + 16] for i in range(0, len(Mobile_mode), 16)]


chunks = np.array(chunks)
chunks_modile = np.array(chunks_modile)

min_mark_index=[]
for i in range(0,len(chunks[0,:])):
    min_mark_index.append(np.argmin(chunks[:,i]))
    
min_Mobile_mode=[]
for i in range(0,len(min_mark_index)):
    min_Mobile_mode.append([min_mark_index[i],chunks_modile[min_mark_index[i],i]])
    # print(chunks_modile[min_mark_index[i],i])

print(min_Mobile_mode)
# print("min_distance_index:",min_mark_index)















