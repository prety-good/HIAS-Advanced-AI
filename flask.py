"""
@Description:   参考顺序：MCTS_algo.py-->my_algo.py-->algo.py-->forflask_algo.py，简化并改进，不再严格限制时间func_timeout
                本脚本不再支持本地下棋方式，
                调用方法：
                （1）将从前端返回的棋盘矩阵board转为numpy格式
                （2）调用AIPlayer生成实例aiplayer（可以设置各种初始化参数），然后aiplayer.get_move(board)即可返回合法位置（2,3）
@File       :   algo.py
@Contact    :   xiaxiaoshux@163.com

@Modify Time      @Author       @Version    @Change
---------------   -----------   --------    -----------
2022/9/17 18:31    xiaxiaoshux      1.0     原来的黑子是人类玩家，白子是电脑。需要玩家自己挑选执黑白子
2022/9/18 18:31    xiaxiaoshux      1.0     原来的是光记录子结点后不动了，但是没随机扩展（我都不知道他怎么训练出来蒙特卡洛树的）
2022/9/19 18:31    xiaxiaoshux      1.0     回溯时的反馈值现在以胜负量化，或者以胜子数量化
2022/9/20 20:31    xiaxiaoshux      1.0     将force_loss转为game_over_count
2022/9/20 21:31    xiaxiaoshux      1.0     改为numpy，坐标为统一从0开始的数字
2022/9/21 18:31    xiaxiaoshux      1.0     改为numpy，白旗0，黑旗1，空子是-1，可以下的点是2
2022/9/22 18:31    xiaxiaoshux      1.0     将ReversiBoard从自定义改为传输定义
"""
import datetime
from func_timeout import func_timeout, FunctionTimedOut
import random
from time import time
from copy import deepcopy
import numpy as np
from numpy import sqrt, log


class ReversiBoard:
    def __init__(self, board):
        """
        拿到传输的数据boardMat，来初始化当前棋盘
        """
        self.empty = 2  # 未落子状态
        self.boardMat = board

    def show(self):
        """
        显示当前棋盘
        :return:
        """
        print('  ', str(list(range(8))).lstrip('[').rstrip(']').replace(',',''))  # 打印列名,开头空一格
        for i in range(8):  # 打印行名和棋盘
            print(str(i), self.boardMat[i])

    def is_on_board(self, x, y):
        """
        判断坐标是否出界
        :param x: row 行坐标
        :param y: col 列坐标
        :return: True or False
        """
        return (0 <= x <= 7) and (0 <= y <= 7)

    def reverse_flip(self, action, color):
        """
        _move函数中调用的函数，检测落子是否合法,如果不合法，返回 False，否则返回反转子的坐标列表
        :param action: 下子位置
        :param color: [X,0,.] 棋子状态
        :return: False or 反转对方棋子的坐标列表
        """
        xstart, ystart = action

        # 如果该位置已经有棋子或者出界，返回 False
        if not self.is_on_board(xstart, ystart) or self.boardMat[xstart, ystart] != self.empty:
            return False

        # 临时将color放到指定位置
        self.boardMat[xstart][ystart] = color

        # 棋手
        op_color = 0 if color == 1 else 1

        # 要被翻转的棋子
        flipped_pos = []  # 数字坐标

        for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:  # 八个方向依次遍历
            x, y = xstart, ystart
            x += xdirection
            y += ydirection
            # 如果(x,y)在棋盘上，而且为对方棋子,则在这个方向上继续前进，否则循环下一个角度。
            if self.is_on_board(x, y) and self.boardMat[x, y] == op_color:
                x += xdirection
                y += ydirection
                # 进一步判断点(x,y)是否在棋盘上，如果不在棋盘上，继续循环下一个角度,如果在棋盘上，则进行while循环。
                if not self.is_on_board(x, y):
                    continue
                # 一直走到出界或不是对方棋子的位置
                while self.boardMat[x, y] == op_color:
                    # 如果一直是对方的棋子，则点（x,y）一直循环，直至点（x,y)出界或者不是对方的棋子。
                    x += xdirection
                    y += ydirection
                    # 点(x,y)出界了和不是对方棋子
                    if not self.is_on_board(x, y):
                        break
                # 出界了，则没有棋子要翻转OXXXXX
                if not self.is_on_board(x, y):
                    continue

                # 是自己的棋子OXXXXXXO
                if self.boardMat[x, y] == color:
                    while True:
                        x -= xdirection
                        y -= ydirection
                        # 回到了起点则结束
                        if x == xstart and y == ystart:
                            break
                        # 需要翻转的棋子
                        flipped_pos.append([x, y])

        # 将前面临时放上的棋子去掉，即还原棋盘(可以下的棋子已经在要翻转的列表中了)
        self.boardMat[xstart, ystart] = self.empty  # restore the empty space

        # 没有要被翻转的棋子，则走法非法。返回 False
        if len(flipped_pos) == 0:
            return False

        # 走法正常，返回翻转棋子的棋盘坐标
        return flipped_pos

    def get_legal_actions(self, color):
        """
        按照黑白棋的规则获取棋子的合法走法
        :param color: 当前颜色的棋子，X-黑棋，O-白棋
        :return: 生成合法的落子坐标，用list()方法可以获取所有的合法坐标
        """

        # 表示棋盘坐标点的8个不同方向坐标，比如方向坐标[0][1]则表示坐标点的正上方。
        direction = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        op_color = 0 if color == 1 else 1
        # 统计 op_color 一方邻近的未落子状态的位置
        op_color_near_points = []

        boardMat = self.boardMat
        for i in range(8):
            # i 是行数，从0开始，j是列数，也是从0开始
            for j in range(8):
                # 判断棋盘[i][j]位子棋子的属性，如果是op_color，则继续进行下一步操作，
                # 否则继续循环获取下一个坐标棋子的属性
                if boardMat[i, j] == op_color:
                    # dx，dy 分别表示[i][j]坐标在行、列方向上的步长，direction 表示方向坐标
                    for dx, dy in direction:
                        x, y = i + dx, j + dy
                        # 表示x、y坐标值在合理范围，棋盘坐标点board[x][y]为未落子状态，
                        # 而且（x,y）不在op_color_near_points 中，统计对方未落子状态位置的列表才可以添加该坐标点
                        if self.is_on_board(x,y) and boardMat[x, y] == self.empty and \
                                (x, y) not in op_color_near_points:
                            op_color_near_points.append((x, y))
        l = [0, 1, 2, 3, 4, 5, 6, 7]
        for p in op_color_near_points:
            # 判断落位是否合法，合法则进行下一步
            if self.reverse_flip(p, color):
                # print(f"2、p={p}")
                yield p

    def _move(self, action, color):
        """
        落子，并获取反转棋子的坐标
        :param action: 落子的坐标 如(2,3)
        :param color: [O,X,.] 表示棋盘上不同的棋子
        :return: 返回反转棋子的坐标列表，落子失败则返回False
        """

        fliped = self.reverse_flip(action, color)

        if fliped:
            # 有就反转对方棋子坐标
            for flip in fliped:
                x, y = flip
                self.boardMat[x, y] = color  # 将待翻转列表全部变成自己的color

            # 落子坐标
            x, y = action  # 落子坐标改为我要下的地方坐标
            # 更改棋盘上 action 坐标处的状态，修改之后该位置属于 color[X,O,.]等三状态
            self.boardMat[x, y] = color
            return fliped
        else:
            # 没有反转子则落子失败
            return False

    def game_over_count(self):
        """
        统计棋盘，计算胜子数量
        :return:    胜者winner，胜子数win_num
        """
        count = {1:(self.boardMat==1).sum(), 0:(self.boardMat==0).sum(), 2:(self.boardMat==2).sum()}

        # 黑棋胜
        if count[1] > count[0]:
            return 0, count[1] - count[0]
        # 白棋胜
        elif count[0] > count[1]:
            return 1, count[0] - count[1]
        # 平局
        else:
            return 2, 0

class TreeNode:
    """
    蒙特卡洛树
    """
    def __init__(self, parent, color):
        self.parent = parent    # 记录父结点
        self.reward = 0         # 经过当前结点所得收益之和
        self.n = 1e-6              # 访问当前结点的次数
        self.color = color
        self.child = dict()     # 键值对为{要下的位置坐标:对应的TreeNode}

class BaseGame:
    def __init__(self, black_player, white_player):
        self.black_player = black_player  # 黑棋一方
        self.white_player = white_player  # 白棋一方
        self.black_player.color = 1
        self.white_player.color = 0
        self.board = None
        self.cur_player = None

    def next_player(self):
        """切换轮次"""
        if self.cur_player is None or self.cur_player == self.white_player:
            return self.black_player
        else:
            return self.white_player

    def game_over(self):
        """
        判断游戏是否结束
        :return: True/False 游戏结束/游戏没有结束
        """
        # 根据当前棋盘，判断棋局是否终止
        # 如果当前选手没有合法下棋的位子，则切换选手；如果另外一个选手也没有合法的下棋位置，则比赛停止。
        b_list = list(self.board.get_legal_actions(1))
        w_list = list(self.board.get_legal_actions(0))

        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

        return is_over

class Simulate_Game(BaseGame):
    """
    从结点M出发，蒙特卡洛模拟扩展搜索树，直到找到一个终止结点（即胜负）
    """
    def __init__(self, black_player, white_player, board, cur_player):
        super(Simulate_Game, self).__init__(black_player, white_player)

        self.board = deepcopy(board)
        self.cur_player = cur_player
        self.winner = None  # 胜者
        self.win_num = None # 胜子数

    def run(self):
        """开始模拟"""
        while True:
            # 切换当前玩家,如果当前玩家是 None 或者白棋 white_player，则返回黑棋 black_player;
            #  否则返回 white_player。
            self.cur_player = self.next_player()

            color = 1 if self.cur_player==self.black_player else 0

            # 获取当前下棋方合法落子位置
            legal_actions = list(self.board.get_legal_actions(color))
            # print("%s合法落子坐标列表："%color,legal_actions)
            if len(legal_actions) == 0:
                # 判断游戏是否结束
                if self.game_over():
                    # 游戏结束，双方都没有合法位置
                    self.winner, self.win_num = self.board.game_over_count()  # 得到赢家 0,1,2
                    break
                else:
                    # 另一方有合法位置,切换下棋方
                    continue

            # 在有合法落子位置前提下，根据不同的策略（random or roxanne）落子
            # change:get_move的参数将legal_actions重用，不再使用board
            action = self.cur_player.get_move(legal_actions)
            self.board._move(action=action, color=color)

        return self.winner, self.win_num


class SimulateStrategyPlayer:
    """
    模拟玩家，有多种策略，此为基类
    """
    def __init__(self, color):
        self.color = color

    def get_move(self, action_list):
        pass

class SimRoxannePlayer(SimulateStrategyPlayer):
    """模拟Roxanne策略的玩家"""

    def __init__(self, color):
        super(SimRoxannePlayer, self).__init__(color=color)
        self.roxanne_mat =  [
             [(0, 0), (0, 7), (7, 0), (7, 7)],
             [(2, 2), (2, 5), (5, 2), (5, 5)],
             [(3, 2), (3, 5), (4, 2), (4, 5), (2, 3), (2, 4), (5, 3), (5, 4)],
             [(2, 0), (2, 7), (5, 0), (5, 7), (0, 2), (0, 5), (7, 2), (7, 5)],
             [(3, 0), (3, 7), (4, 0), (4, 7), (0, 3), (0, 4), (7, 3), (7, 4)],
             [(2, 1), (2, 6), (5, 1), (5, 6), (1, 2), (1, 5), (6, 2), (6, 5)],
             [(3, 1), (3, 6), (4, 1), (4, 6), (1, 3), (1, 4), (6, 3), (6, 4)],
             [(1, 1), (1, 6), (6, 1), (6, 6)],
             [(1, 0), (1, 7), (6, 0), (6, 7), (0, 1), (0, 6), (7, 1), (7, 6)]
        ]

    def get_move(self, action_list):
        """
        落子策略的具体实现
        :param action_list:   得到的合法落子位置
        :return:    返回一个落子位置
        """
        # 若无合法落子位置，返回None
        if len(action_list) == 0:
            return None
        # 若有合法落子位置，进一步挑选符合Roxanne策略的落子位置
        else:
            # 级别高的优先选择，同级别的随机选择一个返回
            for roxanne_list in self.roxanne_mat:
                random.shuffle(roxanne_list)
                for action in roxanne_list:
                    if action in action_list:
                        return action

class SimRandomPlayer(SimulateStrategyPlayer):
    """
    模拟随机策略的玩家
    """
    def __init__(self, color):
        super(SimRandomPlayer, self).__init__(color=color)

    def get_move(self, action_list):
        """
        落子策略的具体实现
        :param action_list:
        :return:    返回一个落子位置
        """
        # 若无合法落子位置，返回None
        if len(action_list) == 0:
            return None
        # 若有合法落子位置，进一步随机挑选落子位置返回
        else:
            random.shuffle(action_list)
            return action_list[0]

class AIPlayer:
    """
    蒙特卡罗树搜索智能算法
    """

    def __init__(self, color, time_limit=5, sim_strategy="roxanne", back_prop_strategy="win"):
        """
        :param color:
        :param time_limit:  限定搜索时长，默认5s
        :param sim_strategy:两种模拟策略，默认贪心"roxanne"，还有"random"
        :param back_prop_strategy:  回溯值的计数策略，默认胜负计"win"，还有胜子数计"win_num"

        """
        self.color = color
        self.time_limit = time_limit
        # 记录开始搜索的时间，
        self.tick = 0
        # TODO:扩展的方式由随机策略->Roxanne策略
        if sim_strategy=="roxanne":
            self.simulate_black = SimRoxannePlayer(1)
            self.simulate_white = SimRoxannePlayer(0)
        elif sim_strategy=="random":
            self.simulate_black = SimRandomPlayer(1)
            self.simulate_white = SimRandomPlayer(0)

        self.back_prop_strategy = back_prop_strategy

    def select(self, node, board):
        """
        selection，向下递归选择子结点，直到叶子结点或者还未扩展的结点L
        :param node:
        :param board:
        :return:
        """
        # 叶子结点或还未扩展的结点：无子结点
        if len(node.child) == 0:
            return node
        # 否则继续递归选择子结点
        else:
            best_score = -np.inf
            best_move = None
            # 遍历所有子结点，选择最优子结点
            for k in node.child.keys():
                # 若当前子结点没有访问过，则其UCB1值必然无穷大，扩展该结点
                if node.child[k].n == 0:
                    best_move = k
                    break
                # 否则若访问过，计算UCB1值
                else:
                    # 记录临时变量，方便查看公式
                    N = node.n  # 当前父结点的访问次数
                    n = node.child[k].n  # 当前子结点的访问次数
                    reward = node.child[k].reward  # 当前子结点的收益分数
                    # 随着访问次数的增加，加号后面的值越来越小，因此我们的选择会更加倾向于选择那些还没怎么被统计过的节点
                    # 避免了蒙特卡洛树搜索会碰到的贪心陷阱
                    score = reward / n + sqrt(2 * log(N) / n)
                    # 若当前结点的UCB1值最大，记录为最佳
                    if score > best_score:
                        best_score = score
                        best_move = k

            # best_move即要下的位置
            board._move(action=best_move, color=node.color)
            return self.select(node.child[best_move], board)

    def expand(self, node, board):
        """
        expansion，如果结点L不是个终止结点，则扩展它未被扩展的子结点M。采用随机扩展策略
        :param node:    结点L，注意，这里都是引用，是对存在于mcts函数中的蒙特卡洛树的扩展
        :param board:
        :return:        随机扩展结点M
        """
        # 结点L有多个未被扩展的子结点M
        legal_actions = list(board.get_legal_actions(node.color))
        if len(legal_actions) == 0:
            return node

        for action in legal_actions:
            # 其子结点采用键值对记录。键：棋子的合法位置；值：子结点（颜色与父结点相反）
            oppo_color = 0 if node.color == 1 else 1
            node.child[action] = TreeNode(parent=node, color=oppo_color)

        # 新增：随机扩展结点M并返回
        return node.child[random.choice(legal_actions)]

    def simulate(self, node, board):
        """
        simulation，从结点M出发，模拟扩展搜索树，直到找到一个终止结点。
        模拟过程中使用的策略和采用UCB1算法实现的过程并不相同，模拟通常会使用比较简单的策略，
        如随机策略，Roxanne策略等
        :param node:
        :param board:
        :return:
        """
        if node.color==0:
            cur_player = self.simulate_black
        else:
            cur_player = self.simulate_white
        simulation = Simulate_Game(black_player=self.simulate_black,
                                   white_player=self.simulate_white,
                                   board=board,
                                   cur_player=cur_player)
        return simulation.run()

    def back_propagation(self, node, score):
        """
        蒙特卡洛树搜索，反向传播，回溯更新模拟路径中的节点奖励
        """
        node.n += 1
        node.reward += score
        # 只要不是root，就一直回溯
        if node.parent is not None:
            if self.back_prop_strategy == "win_num":
                self.back_propagation(node=node.parent, score=-score)  # 当reward量化策略调整为(胜子数/8*8), score = -score
            else:
                self.back_propagation(node=node.parent, score=1-score)

    def mcts(self, board):
        """
        进行蒙特卡洛树搜索
        :param board:   当前棋盘状态
        :return:
        """

        root = TreeNode(parent=None, color=self.color)
        print("root.color = ", root.color)

        # 限定搜索规模
        while time() - self.tick < self.time_limit:
            # 每次都深拷贝一个当前棋盘状态
            simulate_board = deepcopy(board)
            # step1:selction    选择最优子结点L
            choice_L = self.select(node=root, board=simulate_board)
            print("choice_L = ", root.child)
            print("choice_L.color = ", choice_L.color)
            # step2:expansion   扩展L的子结点M
            choice_M = self.expand(choice_L, simulate_board)
            print("choice_M = ", choice_L.child)
            print("choice_M.color = ", choice_M.color)
            # step3:simulation  不同的策略模拟
            winner, win_num = self.simulate(node=choice_M, board=simulate_board)
            # step4:back propagation    回溯
            # reward量化策略：模拟得到的胜者（黑、白、平局）决定了反馈值
            if self.back_prop_strategy == "win_num":
                if choice_M.color == 1:
                    # 若到的结点M执白，则白子在胜者为（黑、白、平局）的终局下的反馈值分别为[负子数,胜子数,0]
                    back_score = [-win_num, win_num, 0][winner]
                else:
                    # 若到的结点M执黑，则黑子在胜者为（黑、白、平局）的终局下的反馈值分别为[负子数,胜子数,0]
                    back_score = [win_num, -win_num, 0][winner]
            else:# self.back_prop_strategy == "win"
                if choice_M.color == 1:
                    # 若到的结点M执白，则白子在胜者为（黑、白、平局）的终局下的反馈值分别为[0,1,0.5]
                    back_score = [0, 1, 0.5][winner]
                else:
                    # 若到的结点M执黑，则黑子在胜者为（黑、白、平局）的终局下的反馈值分别为[1,0,0.5]
                    back_score = [1, 0, 0.5][winner]
            self.back_propagation(node=choice_M, score=back_score)

        # 搜索时间到，返回当前root下的最优解
        best_n = -1
        best_move = None
        for k in root.child.keys():
            if root.child[k].n > best_n:
                best_n = root.child[k].n
                best_move = k
        return best_move

    def get_move(self, board):
        """
        每个轮次传入一个棋盘状态，进行蒙特卡洛搜索
        :param board:
        :return:
        """
        self.tick = time()
        action = self.mcts(board=deepcopy(board)) # 拿着当前的棋盘状态深拷贝

        return action

if __name__ == '__main__':
    # AI 玩家 白棋初始化
    white_player = AIPlayer(0)

