import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

# 초기 상태의 미로 모습
fig = plt.figure(figsize=(8, 8))

# 격자 표시
for i in range(1, 6):
    plt.plot([0, 5], [i, i], color='gray', linewidth=1)
    plt.plot([i, i], [0, 5], color='gray', linewidth=1)

# reward 표시
plt.text(4.5, 2.5, '10.00', ha='center')
plt.text(0.5, 0.5, '-10.00', ha='center')
plt.text(1.5, 0.5, '-10.00', ha='center')
plt.text(2.5, 0.5, '-10.00', ha='center')
plt.text(3.5, 0.5, '-10.00', ha='center')
plt.text(4.5, 0.5, '-10.00', ha='center')

# 그림을 그릴 범위 및 눈금 제거 설정
ax = plt.gca()
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

# S15에 녹색 원으로 현재 위치 표시
line, = ax.plot([0.5], [4.5], marker="o", color='g', markersize=60)

# Q-learning algorithm으로 행동가치 함수 Q 수정
def Q_learning(s, a, r, s_next, Q, eta, gamma):
    if s_next == 14:  # 목표 지점에 도달한 경우
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next, :]) - Q[s, a])
    return Q

def simple_convert_into_pi_from_theta(theta):
    '''단순 비율 계산'''
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0, m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

# 줄은 상태 0~24, 열은 행동방향(상,우,하,좌 순)를 나타낸다
theta_0 = np.array([[np.nan, 1, 1, np.nan], # s0
                    [np.nan, 1, np.nan, 1], # s1
                    [np.nan, 1, np.nan, 1], # s2
                    [np.nan, 1, np.nan, 1], # s3
                    [np.nan, np.nan, 1, 1], # s4

                    [1, np.nan, 1, np.nan], # s5
                    [np.nan, np.nan, np.nan, np.nan], # s6 (wall)
                    [np.nan, np.nan, np.nan, np.nan], # s7 (wall)
                    [np.nan, np.nan, np.nan, np.nan], # s8 (wall)
                    [1, np.nan, 1, np.nan], # s9

                    [1, np.nan, 1, np.nan], # s10
                    [np.nan, np.nan, np.nan, np.nan], # s11 (wall)
                    [np.nan, np.nan, np.nan, np.nan], # s12 (wall)
                    [np.nan, np.nan, np.nan, np.nan], # s13 (wall)
                    [np.nan, np.nan, np.nan, np.nan], # s14 (goal)

                    [1, 1, 1, np.nan], # s15 (시작 지점)
                    [np.nan, 1, 1, 1], # s16
                    [np.nan, 1, 1, 1], # s17
                    [np.nan, 1, 1, 1], # s18
                    [1, np.nan, 1, 1], # s19

                    [1, 1, np.nan, np.nan], # s20 (cliff)
                    [1, 1, np.nan, 1], # s21 (cliff)
                    [1, 1, np.nan, 1], # s22 (cliff)
                    [1, 1, np.nan, 1], # s23 (cliff)
                    [1, np.nan, np.nan, 1]]) # s24 (cliff)

# 무작위 행동정책 pi_0을 계산
pi_0 = simple_convert_into_pi_from_theta(theta_0)
print(pi_0)

# 행동가치 함수 Q의 초기 상태
[a, b] = theta_0.shape
#Q = np.random.rand(a, b) * theta_0 * 0.1
Q = np.random.rand(a, b) * (theta_0 == 1) * 0.1 # 통째로 nan인 열이 있으므로 nan값은 0으로 초기화
#print(Q)

# ε-greedy 알고리즘 구현
def get_action(s, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    if np.random.rand() < epsilon:
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        next_direction = direction[np.nanargmax(Q[s, :])]
    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3
    return action

def get_s_next(s, a):
    if a == 0:
        s_next = s - 5
    elif a == 1:
        s_next = s + 1
    elif a == 2:
        s_next = s + 5
    elif a == 3:
        s_next = s - 1
    return s_next

def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 15
    a = a_next = get_action(s, Q, epsilon, pi)
    s_a_history = [[15, np.nan]]
    while (1):
        a = a_next
        s_a_history[-1][1] = a
        s_next = get_s_next(s, a)
        s_a_history.append([s_next, np.nan])
        if s_next == 14:
            r = 10
            a_next = np.nan
        elif s_next in [20, 21, 22, 23, 24]:
            r = -10
            a_next = get_action(s_next, Q, epsilon, pi)
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)
        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)
        if s_next == 14:
            break
        else:
            s = s_next
    return [s_a_history, Q]

def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    ax.plot([0.5], [4.5], marker="s", color=cm.jet(V[i][0]), markersize=85) # S0
    ax.plot([1.5], [4.5], marker="s", color=cm.jet(V[i][1]), markersize=85) # S1
    ax.plot([2.5], [4.5], marker="s", color=cm.jet(V[i][2]), markersize=85) # S2
    ax.plot([3.5], [4.5], marker="s", color=cm.jet(V[i][3]), markersize=85) # S3
    ax.plot([4.5], [4.5], marker="s", color=cm.jet(V[i][4]), markersize=85) # S4

    ax.plot([0.5], [3.5], marker="s", color=cm.jet(V[i][5]), markersize=85) # S5
    ax.plot([1.5], [3.5], marker="s", color='gray', markersize=85) # S11
    ax.plot([2.5], [3.5], marker="s", color='gray', markersize=85) # S12
    ax.plot([3.5], [3.5], marker="s", color='gray', markersize=85) # S13
    ax.plot([4.5], [3.5], marker="s", color=cm.jet(V[i][9]), markersize=85) # S9

    ax.plot([0.5], [2.5], marker="s", color=cm.jet(V[i][10]), markersize=85) # S10
    ax.plot([1.5], [2.5], marker="s", color='gray', markersize=85) # S11
    ax.plot([2.5], [2.5], marker="s", color='gray', markersize=85) # S12
    ax.plot([3.5], [2.5], marker="s", color='gray', markersize=85) # S13
    ax.plot([4.5], [2.5], marker="s", color='g', markersize=85) # S14 (goal)

    ax.plot([0.5], [1.5], marker="s", color='orange', markersize=85) # S15 (start)
    ax.plot([1.5], [1.5], marker="s", color=cm.jet(V[i][16]), markersize=85) # S16
    ax.plot([2.5], [1.5], marker="s", color=cm.jet(V[i][17]), markersize=85) # S17
    ax.plot([3.5], [1.5], marker="s", color=cm.jet(V[i][18]), markersize=85) # S18
    ax.plot([4.5], [1.5], marker="s", color=cm.jet(V[i][19]), markersize=85) # S19

    ax.plot([0.5], [0.5], marker="s", color='red', markersize=85) # S20 (cliff)
    ax.plot([1.5], [0.5], marker="s", color='red', markersize=85) # S21 (cliff)
    ax.plot([2.5], [0.5], marker="s", color='red', markersize=85) # S22 (cliff)
    ax.plot([3.5], [0.5], marker="s", color='red', markersize=85) # S23 (cliff)
    ax.plot([4.5], [0.5], marker="s", color='red', markersize=85) # S24 (cliff)
    
    state = s_a_history[i][0] # 현재 위치
    x = (state % 5) + 0.5
    y = 4.5 - int(state / 5)
    line.set_data(x, y)

    return (line,)

# q-learning
eta = 0.1
gamma = 0.9
epsilon = 0.1
v = np.nanmax(Q, axis=1)
is_continue = True
episode = 1
V = []
V.append(np.nanmax(Q, axis=1))

while is_continue:
    print("에피소드: " + str(episode))
    epsilon = epsilon / 2
    [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)
    new_v = np.nanmax(Q, axis=1)
    print(np.sum(np.abs(new_v - v)))
    #print(s_a_history)
    v = new_v
    V.append(v)
    
    print("목표 지점에 이르기까지 걸린 단계 수는 " + str(len(s_a_history) - 1) + "단계입니다")

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(s_a_history), interval=200, repeat=False)

    episode = episode + 1
    if episode > 100:
        break




plt.show()
