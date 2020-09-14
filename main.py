from Node import *
from depth_first import *
from breadth_first import *
# from A_star import *
from config import originate, target
import time

if __name__ == '__main__':
    # 深度优先算法
    node1 = Node(None, originate, 0)
    node2 = Node(None, target, 0)
    depth = degth_search(node1, node2, 10, 3)
    breadth = breadth_search(node1, node2, 10, 3)
    Now_d = time.time()
    flag_d = depth.search()
    end_d = time.time()
    Now_b = time.time()
    flag_b = breadth.search()
    end_b = time.time()
    cost_d = end_d - Now_d
    cost_b = end_b - Now_b
    if (flag_d):
        print('The result of DFS')
        depth.showLine()
        print('Spent time：%f s\n\n' % (cost_d))
    else:
        print('error')

    if (flag_d):
        print('The result of BFS')
        breadth.showLine()
        print('Spent time：%f s' % (cost_b))
    else:
        print('error')

    # if (flag_d):
    #     print('a_star算法:已经找到路径')
    #     A_star.showLine()
    #     print('a_star算法共用时%f秒' % (cost_b))
    # else:
    #     print('未找到路径')
