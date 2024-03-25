### HIAS Advanced AI course
- 运行`python gui.py`即可运行程序，程序运行后点击开始按钮开始下棋。
- 调整`main.py->AIPlayer.__init__()->self.max_times`参数即可调整蒙特卡洛的搜索最大迭代次数。
- 在`gui.py->GameThread.__init__()`中，调整`black_player`和`white_player`可以设置人机对战、人人对战或两个ai对战。