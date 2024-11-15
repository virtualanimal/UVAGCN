import numpy as np
import os

def transform_joints(data):
    N, C, T, V, M = data.shape
    new_data = np.zeros((N, C, T, 25, M))

    new_data[:, :, :, 0] = (data[:, :, :, 11] + data[:, :, :, 12]) / 2
    new_data[:, :, :, 2] = (data[:, :, :, 3]+data[:, :, :, 4])/2
    new_data[:, :, :, 3] = (data[:, :, :, 1]+data[:, :, :, 2])/2
    new_data[:, :, :, 4] = data[:, :, :, 5] 
    new_data[:, :, :, 5] = data[:, :, :, 7]
    new_data[:, :, :, 6] = data[:, :, :, 9]
    offset1 = 0.1 * (data[:, :, :, 9] - data[:, :, :, 7]) 
    new_data[:, :, :, 7] = data[:, :, :, 9] + offset1  # Right shoulder
    new_data[:, :, :, 8] = data[:, :, :, 6]  
    new_data[:, :, :, 9] = data[:, :, :, 8]
    new_data[:, :, :, 10] = data[:, :, :, 10]
    offset2 = 0.1 * (data[:, :, :, 10] - data[:, :, :, 8])
    new_data[:, :, :, 11] = data[:, :, :, 10] + offset2  # Left knee
    new_data[:, :, :, 12] = data[:, :, :, 11]
    new_data[:, :, :, 13] = data[:, :, :, 13]
    new_data[:, :, :, 14] = data[:, :, :, 15]
    offset3 = 0.1 * (data[:, :, :, 15] - data[:, :, :, 13])
    new_data[:, :, :, 15] = data[:, :, :, 15] + offset3 # Right ankle
    new_data[:, :, :, 16] = data[:, :, :, 12]
    new_data[:, :, :, 17] = data[:, :, :, 14]
    new_data[:, :, :, 18] = data[:, :, :, 16]
    offset4 = 0.1 * (data[:, :, :, 16] - data[:, :, :, 14])
    new_data[:, :, :, 19] = data[:, :, :, 16] + offset4
    new_data[:, :, :, 20] = (data[:, :, :, 5]+data[:, :, :, 6])/2
    new_data[:, :, :, 21] = new_data[:, :, :, 7] + offset1
    new_data[:, :, :, 22] = new_data[:, :, :, 6] + 0.5 * offset1
    new_data[:, :, :, 23] = new_data[:, :, :, 11] + offset2
    new_data[:, :, :, 24] = new_data[:, :, :, 10] + 0.5 * offset2
    new_data[:, :, :, 1] = (new_data[:, :, :, 0]+new_data[:, :, :, 20])/2

    return new_data


def transform_joints_bone(data):
    N, C, T, V, M = data.shape
    new_data = np.zeros((N, C, T, 25, M))

    new_data[:, :, :, 0] = (data[:, :, :, 11] + data[:, :, :, 12]) / 2
    new_data[:, :, :, 3] = (data[:, :, :, 1]+data[:, :, :, 2])/2
    new_data[:, :, :, 4] = data[:, :, :, 5] 
    new_data[:, :, :, 5] = data[:, :, :, 7]
    new_data[:, :, :, 6] = data[:, :, :, 9]
    new_data[:, :, :, 7] = data[:, :, :, 9]  # Right shoulder
    new_data[:, :, :, 8] = data[:, :, :, 6]  
    new_data[:, :, :, 9] = data[:, :, :, 8]
    new_data[:, :, :, 10] = data[:, :, :, 10]
    new_data[:, :, :, 11] = data[:, :, :, 10] # Left knee
    new_data[:, :, :, 12] = data[:, :, :, 11]
    new_data[:, :, :, 13] = data[:, :, :, 13]
    new_data[:, :, :, 14] = data[:, :, :, 15]
    new_data[:, :, :, 15] = data[:, :, :, 15] # Right ankle
    new_data[:, :, :, 16] = data[:, :, :, 12]
    new_data[:, :, :, 17] = data[:, :, :, 14]
    new_data[:, :, :, 18] = data[:, :, :, 16]
    new_data[:, :, :, 19] = data[:, :, :, 16] 
    new_data[:, :, :, 20] = (data[:, :, :, 5]+data[:, :, :, 6])/2
    new_data[:, :, :, 21] = new_data[:, :, :, 7]
    new_data[:, :, :, 22] = new_data[:, :, :, 6]
    new_data[:, :, :, 23] = new_data[:, :, :, 11] 
    new_data[:, :, :, 24] = new_data[:, :, :, 10]
    new_data[:, :, :, 1] = (new_data[:, :, :, 0]+new_data[:, :, :, 20])/2
    new_data[:, :, :, 2] = (new_data[:, :, :, 3]+new_data[:, :, :, 20])/2

    return new_data


os.makedirs('uav_25')
os.makedirs('uav_25_bone')


data = np.load("../resources/data/train_joint.npy")
new_data = transform_joints(data)
print(new_data.shape)
np.save('./uav_25/train_joint.npy', new_data)

data = np.load("../resources/data/val_joint.npy")
new_data = transform_joints(data)
print(new_data.shape)
np.save('./uav_25/val_joint.npy', new_data)

data = np.load("../resources/data/test_joint.npy")
new_data = transform_joints(data)
print(new_data.shape)
np.save('./uav_25/test_joint.npy', new_data)



data = np.load("../resources/data/train_joint.npy")
new_data = transform_joints_bone(data)
print(new_data.shape)
np.save('./uav_25_bone/train_joint.npy', new_data)

data = np.load("../resources/data/val_joint.npy")
new_data = transform_joints_bone(data)
print(new_data.shape)
np.save('./uav_25_bone/val_joint.npy', new_data)

data = np.load("../resources/data/test_joint.npy")
new_data = transform_joints_bone(data)
print(new_data.shape)
np.save('./uav_25_bone/test_joint.npy', new_data)
