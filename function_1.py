# Base functions and classes
import random
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

def random_pick(some_list, probabilities):
    item = 0
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.type = type
        self.distance = 100000
        self.predecessor = None
        self.status = "F"

    def addNeighbour(self, nbr, dt):
        self.connectedTo[nbr] = dt

    def delNeighbor(self, nbr):
        if nbr in self.connectedTo:
            del self.connectedTo[nbr]

    def setStatus(self, newstate):
        self.status = newstate

    def getStatus(self):
        return self.status

    def settype(self, type):
        self.type = type

    def gettype(self):
        return self.type

    def getConnections(self):
        return list(self.connectedTo.keys())

    def getId(self):
        return self.id

    def getLinkInfor(self, nbr):
        if nbr in self.getConnections():
            return self.connectedTo[nbr]
        else:
            return None

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def setDistance(self, distance):
        self.distance = distance

    def getDistance(self):
        return self.distance

    def setPred(self, predecessor):
        self.predecessor = predecessor

    def getPred(self):
        return self.predecessor

class Network:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0
        self.traffic = {}
        self.bandwidth = {}

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key) 
        self.vertList[key] = newVertex
        return newVertex

    def delateVertex(self, key):
        if key in self.vertList:
            del self.vertList[key]
            self.numVertices = self.numVertices - 1
        for i in self.getVertics():
            self.vertList[i].delNeighbor(key)

    def changeResorce(self, key, ch_comp, ch_DU, ch_CU):
        self.vertList[key].resource[0] = self.vertList[key].resource[0] + ch_comp
        self.vertList[key].resource[1] = self.vertList[key].resource[1] + ch_DU
        self.vertList[key].resource[2] = self.vertList[key].resource[2] + ch_CU

    def getVertex(self, n):  
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n): 
        return n in self.vertList

    def addEdge(self, f, t, dt):  
        if f in self.vertList and t in self.vertList:
            self.vertList[f].addNeighbour(t, dt)
            self.vertList[t].addNeighbour(f, dt)

    def delEdge(self, f, t):
        if t in self.vertList[f].getConnections() or f in self.vertList[t].getConnections():
            if f in self.getVertics() and t not in self.getVertics():
                self.vertList[f].delNeighbor(t)
            elif f not in self.getVertics() and t in self.getVertics():
                self.vertList[t].delNeighbor(f)
            elif f in self.getVertics() and t in self.getVertics():
                self.vertList[f].delNeighbor(t)
                self.vertList[t].delNeighbor(f)

    def getVertics(self):
        Vertics = []
        for i in self.vertList.keys():
            Vertics.append(i)
        return Vertics

    def getNeighbors(self, vertex):
        Neighbors = self.vertList[vertex].getConnections()
        return Neighbors

    def __iter__(self):
        return iter(self.vertList.values())

    def find_the_paths(self, NodeA, NodeB, path, allpaths):
        self.vertList[NodeA].setStatus("T")
        path.append(NodeA)

        if NodeA == NodeB:
            mm = copy.deepcopy(path) # my god, so important
            allpaths.append(mm)

        else:
            for i in self.getNeighbors(NodeA):
                if self.vertList[i].getStatus() == 'F':
                    self.find_the_paths(i, NodeB, path, allpaths)
        path.pop()
        self.vertList[NodeA].setStatus('F')
        return allpaths


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]

#################################### Module1，Convert images to matrix  #########################################
# The cameras work individually, and each camera is responsible for its own part. Finally, all camera information is integrated into a map.

def merge_bller(new_position):
    merge_close_node = []
    record = []
    for i in new_position:
        record.append([])
        record[-1].append(i)
        for ii in new_position:
            if i != ii:
                if abs(i[0] - ii[0]) < 0.5 and abs(i[1] - ii[1]) < 0.8:
                    if ii not in record[-1]:
                        record[-1].append(ii)
    for i in record:
        record_x = []
        record_y = []
        if len(i) > 1:
            record_x.append([a[0] for a in i])
            record_y.append([a[1] for a in i])
            if [np.mean(record_x), np.mean(record_y)] not in merge_close_node:
                merge_close_node.append([np.mean(record_x), np.mean(record_y)])
        else:
            merge_close_node.append(i[0])
    return merge_close_node

def bridview(original_figure, output_figure, modification_matrix, size_matrix, acc):
    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'frozen_inference_graph.pb'

    model = cv2.dnn_DetectionModel(frozen_model, config_file)

    classLabels = []
    filename = 'Label.txt'
    with open(filename, 'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')

    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    img = cv2.imread(original_figure) # Read the test img
    IMAGE_H = len(img)
    print(IMAGE_H)
    IMAGE_W = len(img[0])
    print(IMAGE_H)
    src = np.float32([[modification_matrix[0], IMAGE_H], [modification_matrix[1], IMAGE_H], [0, 200], [IMAGE_W, 60]])
    dst = np.float32([[modification_matrix[2], IMAGE_H], [modification_matrix[3], IMAGE_H], [0, 450], [IMAGE_W, 100]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    # Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

    img_n = img[modification_matrix[4]:(0+IMAGE_H), modification_matrix[5]:IMAGE_W] # Apply np slicing for ROI crop
    warped_img_n = cv2.warpPerspective(img_n, M, (IMAGE_W, IMAGE_H)) # Image warping
    plt.imshow(cv2.cvtColor(warped_img_n, cv2.COLOR_BGR2RGB)) # Show results
    plt.show()
    cropped = warped_img_n[size_matrix[1][0]:size_matrix[1][1], size_matrix[0][0]:size_matrix[0][1]]
    cv2.imwrite(output_figure, cropped)

    # Figure detection
    img_1 = cv2.imread(output_figure)
    print(len(img_1[0]))

    ClassIndex, confidence, bbox = model.detect(img_1, confThreshold=acc)

    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN
    for ClassInd, config_file, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(img_1, boxes, (225, 0, 0), 2)
        # cv2.putText(img_1, classLabels[ClassInd-1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
    plt.show()
      # (x,y,w,h) 其中(x,y) 代表 bbox 的左上角，(w,h) 代表 bbox 的宽和高
    # (x,y,w,h) 其中(x,y) 代表 bbox 的中心，(w,h) 代表 bbox 的宽和高

    people_position = []
    for i in bbox:
        # if i[2] < 200 and i[3] < 200: # 有一些识别错误，或者很大的物体我们不考虑
        people_position.append([i[0] + i[2]/2, i[1]+i[3]]) # 注意是脚的位置，而不是中心位置

    return IMAGE_H, IMAGE_W, bbox, people_position


def updated_people_position(matrix, people_position, file):
    img = cv2.imread(file)  # Read the test img
    IMAGE_H = len(img)
    IMAGE_W = len(img[0])
    corrsponding_node = []
    for i in people_position:
        temporary = [i[0] / IMAGE_W * matrix[0], i[1] / IMAGE_H * matrix[1]]
        corrsponding_node.append(temporary)

    # print(corrsponding_node)
    merge_people_position = merge_bller(corrsponding_node)
    print(merge_people_position)

    return merge_people_position


def draw(merge_people_position):
    x = []
    y = []
    for i in merge_people_position:
        x.append(i[0])
        y.append(i[1])

    ax = plt.gca()

    ax.xaxis.set_ticks_position('top') 

    ax.plot(x, y, 'o', markersize=5)
    ax.invert_yaxis()
    plt.grid(linestyle='-.')
    # plt.title('algorithm comparison')
    # plt.xlabel('Measured Q-factor of test data set (dB)', fontsize=16, fontweight='bold')
    # plt.ylabel('Predicted Q-factor (dB)', fontsize=16, fontweight='bold')
    plt.xlim(0, initial_cood[0])
    plt.ylim(initial_cood[1], 0)
    # plt.legend(fontsize=16, loc='upper left')
    # plt.legend()
    plt.savefig('d.png', dpi=600, bbox_inches='tight')
    plt.show()
    # plt.ion()：

#################################### Module2，plot the objectev on the matrix  #########################################

def convert_to_matrix(new_position, avoiding_wide):
    matrix_example1 = []
    matrix_example2 = []
    for i in range(initial_cood[1] + 1):
        matrix_example1.append([])
        for i in range(initial_cood[0]):
            matrix_example1[-1].append(10)

    for i in range(initial_cood[1]):
        matrix_example2.append([])
        for i in range(initial_cood[0] + 1):
            matrix_example2[-1].append(10)

    # print(matrix_example1)
    # print(matrix_example2)
    for i in new_position:
        x_ceiling = math.ceil(i[0])
        x_floor = math.floor(i[0])
        y_ceiling = math.ceil(i[1])
        y_floor = math.floor(i[1])
        if i[0] + avoiding_wide >= x_ceiling:
            matrix_example2[y_floor][x_ceiling] = 0
            if i[1] + avoiding_wide >= y_ceiling:
                matrix_example1[y_ceiling][x_ceiling] = 0
                if y_ceiling < initial_cood[1]:
                    matrix_example2[y_ceiling][x_ceiling] = 0
            if i[1] - avoiding_wide <= y_floor:
                matrix_example1[y_floor][x_ceiling] = 0
                if y_floor >= 1:
                    matrix_example2[y_floor-1][x_ceiling] = 0

        if i[0] - avoiding_wide <= x_floor:
            matrix_example2[y_floor][x_floor] = 0
            if i[1] + avoiding_wide >= y_ceiling:
                if x_floor >= 1:
                    matrix_example1[y_ceiling][x_floor-1] = 0
                if y_ceiling < initial_cood[1]:
                    matrix_example2[y_ceiling][x_floor] = 0
            if i[1] - avoiding_wide <= y_floor:
                if x_floor >= 1:
                    matrix_example1[y_floor][x_floor-1] = 0
                if y_floor >= 1:
                    matrix_example2[y_floor-1][x_floor] = 0


        if i[1] + avoiding_wide >= y_ceiling:
            matrix_example1[y_ceiling][x_floor] = 0

        if i[1] - avoiding_wide <= y_floor:
            matrix_example1[y_floor][x_floor] = 0
    return matrix_example1, matrix_example2

######################################### Module3，convert matrix into graph ##########################################

def change_matrix_to_fullconnected_matrix(matrix1, matrix2):
    full_matrix1 = copy.deepcopy(matrix1)
    full_matrix2 = copy.deepcopy(matrix2)
    for i in range(len(matrix1)):
        for ii in range(len(matrix1[i])):
            full_matrix1[i][ii] = 1
    for i in range(len(matrix2)):
        for ii in range(len(matrix2[i])):
            full_matrix2[i][ii] = 1
    return full_matrix1, full_matrix2

def from_matric_to_graph(matrix1, matrix2):
    Net = Network()
    len1 = len(matrix2[0]) # length
    len2 = len(matrix1) # wrigth
    total_number = range(len1 * len2)
    total_node = []
    for i in total_number:
        total_node.append('Node' + str(i + 1))
    for i in total_node:
        Net.addVertex(i)
    for i in range(len(matrix1)):
        for ii in range(len(matrix1[0])):
            if matrix1[i][ii] != 0:
                Net.addEdge('Node' + str(i * len(matrix2[0]) + ii + 1), 'Node' + str(i * len(matrix2[0]) + ii + 2), matrix1[i][ii])
    for i in range(len(matrix2)):
        for ii in range(len(matrix2[0])):
            if matrix2[i][ii] != 0:
                Net.addEdge('Node' + str(i * len(matrix2[0]) + ii + 1), 'Node' + str(i * len(matrix2[0]) + ii + 1 + len(matrix2[0])), matrix2[i][ii])

    full_matrix1, full_matrix2 = change_matrix_to_fullconnected_matrix(matrix1, matrix2)
    Net1 = Network()
    len3 = len(full_matrix2[0]) 
    len4 = len(full_matrix1) 
    total_number = range(len3 * len4)
    total_node = []
    for i in total_number:
        total_node.append('Node' + str(i + 1))
    for i in total_node:
        Net1.addVertex(i)
    for i in range(len(full_matrix1)):
        for ii in range(len(full_matrix1[0])):
            if full_matrix1[i][ii] != 0:
                Net1.addEdge('Node' + str(i * len(full_matrix2[0]) + ii + 1), 'Node' + str(i * len(full_matrix2[0]) + ii + 2),
                            full_matrix1[i][ii])
    for i in range(len(full_matrix2)):
        for ii in range(len(full_matrix2[0])):
            if full_matrix2[i][ii] != 0:
                Net1.addEdge('Node' + str(i * len(full_matrix2[0]) + ii + 1),
                            'Node' + str(i * len(full_matrix2[0]) + ii + 1 + len(full_matrix2[0])), full_matrix2[i][ii])

    return Net, Net1


###################################### Module4，routing provisioning ########################################
def find_the_paths(Net, NodeA, NodeB):
    path = []
    allpaths = []
    Net.vertList[NodeA].setStatus("T")
    path.append(NodeA)

    if NodeA == NodeB:
        mm = copy.deepcopy(path)  # very important
        allpaths.append(mm)

    else:
        for i in Net.getNeighbors(NodeA):
            if Net.vertList[i].getStatus() == 'F':
                Net.find_the_paths(i, NodeB, path, allpaths)
    path.pop()
    Net.vertList[NodeA].setStatus('F')
    for i in range(len(allpaths)):
        length = 0
        q = 0
        while q < len(allpaths[i]) - 1:
            dis = Net.vertList[allpaths[i][q]].getWeight(allpaths[i][q + 1])
            length += dis
            q += 1
        allpaths[i].append(length)
    return allpaths

def choose_the_path(Net1, Net2, source_node, destination):
    path = find_the_paths(Net1, source_node, destination)
    # print(path)
    selected_path = []
    if path != []:
        distance_list = []
        for i in path:
            distance_list.append(i[-1])
        min_index = distance_list.index(min(distance_list))
        selected_path = path[min_index][:-1]
    else:
        print('no path')
    return selected_path

def generate_command(real_length, initial_cood, path):
    overall_command = []
    i = 0
    while i < len(path)-1:
        if len(path[i]) == 5:
            itself = int(path[i][4])
            if len(path[i+1]) == 5:
                next_point = int(path[i+1][4])
            else:
                next_point = int(path[i+1][4:6])
            difference = next_point - itself
        else:
            itself = int(path[i][4:6])
            if len(path[i+1]) == 5:
                next_point = int(path[i+1][4])
            else:
                next_point = int(path[i+1][4:6])
            difference = next_point - itself
        if difference == 1:
            overall_command.append(['left', real_length[0]/initial_cood[0]])
        elif difference > 1:
            overall_command.append(['forward', real_length[1]/initial_cood[1]])
        elif difference == -1:
            overall_command.append(['right', real_length[0]/initial_cood[0]])
        elif difference < -1:
            overall_command.append(['back', real_length[1]/initial_cood[1]])
        i += 1
    return overall_command


def convert(new_position, initial_cood):
    oppsite_position = []
    for i in new_position:
        temp = [initial_cood[0] - i[0], initial_cood[1] - i[1]]
        oppsite_position.append(temp)
    return oppsite_position
























