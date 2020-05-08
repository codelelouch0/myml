import pickle
import numpy as np
from os import path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm

np.set_printoptions(threshold=np.inf)

def transformCommand(command):
    if 'RIGHT' in str(command):
       return 1
    elif 'LEFT' in str(command):
        return 2
    else:
        return 0
    pass


def get_PingpongData(filename):
    Frames = []
    Balls = []
    Commands_1P = []
    Commands_2P = []
    PlatformPos_1P = []
    PlatformPos_2P = []
    Speeds = []
    Blocker = []
    log = pickle.load((open(filename, 'rb')))
    for sceneInfo in log:
        Frames.append(sceneInfo["frame"])
        Balls.append(sceneInfo["ball"])
        # Commands.append(sceneInfo.command)
        PlatformPos_1P.append(sceneInfo["platform_1P"])
        PlatformPos_2P.append(sceneInfo["platform_2P"])
        Commands_1P.append(transformCommand(sceneInfo["command_1P"]))
        Commands_2P.append(transformCommand(sceneInfo["command_2P"]))
        Speeds.append(sceneInfo["ball_speed"])
        Blocker.append(sceneInfo["blocker"])

    commands_1P_ary = np.array([Commands_1P])
    commands_1P_ary = commands_1P_ary.reshape((len(Commands_1P), 1))
    commands_2P_ary = np.array([Commands_2P])
    commands_2P_ary = commands_2P_ary.reshape((len(Commands_2P), 1))
    frame_ary = np.array(Frames)
    frame_ary = frame_ary.reshape((len(Frames), 1))
    data = np.hstack((frame_ary, Balls, PlatformPos_1P, PlatformPos_2P, commands_1P_ary, commands_2P_ary))
    data = np.hstack((data, Speeds, Blocker))#(9,10) (11,12)
    return data


if __name__ == '__main__':
    filename = path.join(path.dirname(__file__), 'pingpong_dataset.pickle')
    data = get_PingpongData(filename)
    
    direction=[]

    for i in range(len(data)-1):
        if(data[i,9]>=0 and data[i,10]>=0):
            direction.append(0) #球移動方向為右上為0
        elif(data[i,9]>0 and data[i,10]<0):
            direction.append(1) #球移動方向為右下為1
        elif(data[i,9]<0 and data[i,10]>0):
            direction.append(2) #球移動方向為左上為2
        elif(data[i,9]<0 and data[i,10]<0):
            direction.append(3) #球移動方向為左下為3
    direction = np.array(direction)
    direction = direction.reshape((len(direction),1))
    

    data = np.hstack((data[1:, :], direction))#13


    mask = [1, 2, 3, 9, 10] #mask for platform_1P_X, ballx, bally vectorsX, vectorsY
    X = data[:, mask]
    Y = data[:, 7]#command_1P 
    
    x_train , x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    
    platform_predict_clf = svm.SVC(C=10, gamma=0.1 ,decision_function_shape='ovo')
    
    
    platform_predict_clf.fit(x_train,y_train)        
        
    y_predict = platform_predict_clf.predict(x_test)
    print(y_predict)

    
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy(正確率) ={:8.3f}%".format(accuracy*100))                
    '''
    ax = plt.subplot(111, projection='3d')  
    ax.scatter(X[Y==0][:,1], X[Y==0][:,2], X[Y==0][:,3], c='#AE0000', alpha = 1)  
    ax.scatter(X[Y==1][:,1], X[Y==1][:,2], X[Y==1][:,3], c='#2828FF', alpha = 1)
    ax.scatter(X[Y==2][:,1], X[Y==2][:,2], X[Y==2][:,3], c='#007500', alpha = 1)
    plt.title("SVM Prediction")    
    ax.set_xlabel('Vectors_x')
    ax.set_ylabel('Vectors_y')
    ax.set_zlabel('Direction')    
               
    plt.show()
    '''
    with open('save/clf_SVMClassification_VectorsAndDirection.pickle', 'wb') as f:
        pickle.dump(platform_predict_clf, f)
    


