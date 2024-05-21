# use Q-learning

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm # color map

# 초기 상태의 미로 모습

#전체 그림의 크기 및 그림을 나타내는 변수 선언
fig = plt.figure(figsize=(8,8))

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
ax = plt.gca(); # 현재 그림의 축(axis) 가져옴
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

# S15에 녹색 원으로 현재 위치 표시, 여러 개 중 1개 반환할 때 쉼표 사용
line, = ax.plot([0.5], [1.5], marker="o", color='g', markersize=60)

# Q-learning algorithm으로 행동가치 함수 Q 수정
def Q_learning(s, a, r, s_next, Q, eta, gamma):
    if s_next == 14: # 목표 지점에 도달한 경우
        Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
    else:
        Q[s, a] = Q[s, a] + eta * (r + gamma * np.nanmax(Q[s_next,: ]) - Q[s, a])
    return Q

def simple_convert_into_pi_from_theta(theta):
    '''단순 비율 계산'''
    [m, n] = theta.shape # theta의 행렬 크기를 구함
    pi = np.zeros((m, n)) # mxn 배열 초기화
    for i in range(0, m): # m개의 행에 대한 반복문
        if np.nansum(theta[i, :]) == 0:
            pi[i, :] = 0  # 모든 값이 NaN이면 0으로 설정
        else:
            pi[i, :] = theta[i, :] / np.nansum(theta[i, :])  # 비율 계산
        # 모든 열에 대해서 pi[I,:] 값을 계산
    pi = np.nan_to_num(pi) # nan을 0으로 변환
    return pi

# 줄은 상태 0~7, 열은 행동방향(상,우,하,좌 순)를 나타낸다
theta_0 = np.array([[np.nan, 1, 1, np.nan], # s0
                    [np.nan, 1, np.nan, 1], # s1
                    [np.nan, 1, np.nan, 1], # s2
                    [np.nan, 1, np.nan, 1], # s3
                    [np.nan, np.nan, 1, 1], # s4

                    [1, np.nan, 1, np.nan], # s5
                    [np.nan, np.nan, np.nan, np.nan], # s6
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [1, np.nan, 1, np.nan], 

                    [1, np.nan, 1, np.nan], # s10
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    #[1, np.nan, 1, np.nan],  #목표지점: 정책x

                    [1, 1, 1, np.nan], # s15 (시작 지점)
                    [np.nan, 1, 1, 1],  
                    [np.nan, 1, 1, 1], 
                    [np.nan, 1, 1, 1], 
                    [1, np.nan, 1, 1],

                    [1, 1, np.nan, np.nan], # s20
                    [1, 1, np.nan, 1],
                    [1, 1, np.nan, 1],
                    [1, 1, np.nan, 1],
                    [1, np.nan, np.nan, 1], # s24
                    ])

# 무작위 행동정책 pi_0을 계산
pi_0 = simple_convert_into_pi_from_theta(theta_0)
print(pi_0)

# 행동가치 함수 Q의 초기 상태
[a, b] = theta_0.shape # theta_0의 열과 행의 갯수를 변수 a, b에 저장
Q = np.random.rand(a, b) * theta_0 * 0.1
# * theta0 로 요소 단위 곱

# ε-greedy 알고리즘 구현
def get_action(s, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    # 행동을 결정
    if np.random.rand() < epsilon:
        # 확률 ε로 무작위 행동을 선택함
        next_direction = np.random.choice(direction, p=pi_0[s, :])
    else:
        # Q값이 최대가 되는 행동을 선택함
        next_direction = direction[np.nanargmax(Q[s, :])]
    # 행동을 인덱스로 변환
    if next_direction == "up":
        action = 0
    elif next_direction == "right":
        action = 1
    elif next_direction == "down":
        action = 2
    elif next_direction == "left":
        action = 3

    return action

def get_s_next(s, a, Q, epsilon, pi_0):
    direction = ["up", "right", "down", "left"]
    next_direction = direction[a] # 행동 a의 방향
    
    # 행동으로 다음 상태를 결정
    if next_direction == "up":
        s_next = s - 5 # 위로 이동하면 상태값이 5 줄어든다
    elif next_direction == "right":
        s_next = s + 1 # 오른쪽으로 이동하면 상태값이 1 늘어난다
    elif next_direction == "down":
        s_next = s + 5 # 아래로 이동하면 상태값이 5 늘어난다
    elif next_direction == "left":
        s_next = s - 1 # 왼쪽으로 이동하면 상태값이 1 줄어든다
    
    return s_next

def goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi):
    s = 15 # 시작 지점
    a = a_next = get_action(s, Q, epsilon, pi) # 첫 번째 행동
    s_a_history = [[15, np.nan]]
    # 에이전트의 행동 및 상태의 히스토리를 기록하는 리스트
    while (1): # 목표 지점에 이를 때까지 반복
        a = a_next # 행동 결정
        s_a_history[-1][1] = a
        # 현재 상태(마지막으로 인덱스가 -1)을 히스토리에 추가
        
        s_next = get_s_next(s, a, Q, epsilon, pi)
        # 다음 단계의 상태를 구함

        s_a_history.append([s_next, np.nan])
        # 다음 상태를 히스토리에 추가, 행동은 아직 알 수 없어서 nan으로

        if s_next == 14:
            r = 10 # 목표 지점에 도달했다면 보상을 부여
            a_next = np.nan
        elif s_next in [20, 21, 22, 23, 24]:
            r = -10
            a_next = get_action(s_next, Q, epsilon, pi)    
        else:
            r = 0
            a_next = get_action(s_next, Q, epsilon, pi)
            # 다음 행동 a_next를 계산

        # 가치함수를 수정
        Q = Q_learning(s, a, r, s_next, Q, eta, gamma)

        # 종료 여부 판정
        if s_next == 14: # 목표 지점에 도달하면 종료
            break
        else:
            s = s_next
    return [s_a_history, Q]



# Q러닝 알고리즘으로 미로 빠져나오기
eta = 0.1 # 학습률
gamma = 0.9 # 시간할인율
epsilon = 0.5 # ε-greedy 알고리즘 epsilon 초기값
v = np.nanmax(Q, axis=1) # 각 상태마다 가치의 최댓값을 계산
is_continue = True
episode = 1
V = [] # 에피소드 별로 상태가치를 저장
V.append(np.nanmax(Q, axis=1))
# 상태 별로 행동가치의 최댓값을 계산

while is_continue: # is_continue의 값이 False가 될 때까지 반복
    print("에피소드: " + str(episode))

    # ε 값을 조금씩 감소시킴
    epsilon = epsilon / 2

    # Q러닝으로 미로를 빠져나온 후, 행동 히스토리와 Q값 변수에 저장
    [s_a_history, Q] = goal_maze_ret_s_a_Q(Q, epsilon, eta, gamma, pi_0)

    # 상태가치의 변화
    new_v = np.nanmax(Q, axis=1) # 각 상태마다 행동가치의 최댓값
    print(np.sum(np.abs(new_v - v))) # 상태가치 함수의 변화를 출력
    v = new_v
    V.append(v) # 현재 에피소드가 끝난 시점의 상태가치 함수를 추가

    print("목표 지점에 이르기까지 걸린 단계 수는 " + str(len(s_a_history)- 1) + "단계입니다")

    def init():
        # 배경색 초기화
        line.set_data([], [])
        return (line,)
    
    def animate(i):
        for j in range(5):
            for k in range(5):
                s = j * 5 + k
                if s == 14:
                    color = 'green'
                elif s == 15:
                    color = 'orange'
                elif s in [20, 21, 22, 23, 24]:
                    color = 'red'
                elif s in [6, 7, 8, 11, 12, 13]:
                    color = 'gray'
                else:
                    color = cm.jet(V[i][s] / np.nanmax(V))
                ax.plot([k + 0.5], [4.5 - j], marker="s", color=color, markersize=85)

        # state = state_history[i] # 현재 위치
        # x = (state % 5) + 0.5 # x좌표 : (state % 5) 계산 결과 0, 1, 2
        # y = 4.5 - int(state / 5) # y좌표 : (state / 5) 계산 결과 0, 1, 2
        # line.set_data(x, y)
        return (line,)
    
    # 초기화 함수와 프레임 단위로 그림을 그리는 함수로 애니메이션을 생성
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(V), interval=200, repeat=False)


    # 100 에피소드 반복
    episode = episode + 1
    if episode > 100:
        break



plt.show()
