from Node import *
from DFS import *
from BFS import *
from config import originate, target
import time

if __name__ == '__main__':
    Node1 = Node(None, originate, 0)
    Node2 = Node(None, target, 0)
    DFS = DFS(Node1, Node2, 10, 3)
    BFS = BFS(Node1, Node2, 10, 3)
    a_star = a_star(Node1, Node2, 10, 3)

    #深度优先
    start_d = time.time()
    flag_d = DFS.search()
    end_d = time.time()
    cost_d = end_d - start_d

    #广度优先
    start_b = time.time()
    flag_b = BFS.search()
    end_b = time.time()
    cost_b = end_b - start_b

    if (flag_d):
        print('The result of DFS')
        DFS.showLine()
        print('Spent time：%f s\n\n' % (cost_d))
    else:
        print('error')

    if (flag_b):
        print('The result of BFS')
        BFS.showLine()
        print('Spent time：%f s' % (cost_b))
    else:
        print('error')

