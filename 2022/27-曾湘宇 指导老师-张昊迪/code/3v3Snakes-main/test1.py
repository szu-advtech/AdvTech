# 定义网格 grid
grid = [  [1, 3, 1],
  [1, 5, 1],
  [4, 2, 1]
]

# 获取网格的行数和列数
m = len(grid)
n = len(grid[0])

# 建立二维数组 dp，其中 dp[i][j] 表示到达网格的第 i 行第 j 列的最小路径和
dp = [[0] * n for _ in range(m)]

# 初始化 dp[0][0] 的值为网格的第一个数
dp[0][0] = grid[0][0]

# 遍历网格的第一行
for j in range(1, n):
  # 使用递推公式求出 dp[0][j] 的值
  dp[0][j] = dp[0][j-1] + grid[0][j]

# 遍历网格的第一列
for i in range(1, m):
  # 使用递推公式求出 dp[i][0] 的值
  dp[i][0] = dp[i-1][0] + grid[i][0]

# 遍历网格的剩余部分
for i in range(1, m):
  for j in range(1, n):
    # 使用递推公式求出 dp[i][j] 的值
    dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

# 打印 dp[m-1][n-1]，即为从左上角到右下角的最小路径和
print(dp[m-1][n-1])
