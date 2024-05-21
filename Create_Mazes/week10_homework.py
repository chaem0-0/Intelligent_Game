# 구현에 사용할 패키지 임포트하기
import numpy as np
import matplotlib.pyplot as plt


# 초기 상태의 미로 모습

# 전체 그림의 크기 및 그림을 나타내는 변수 선언
fig = plt.figure(figsize=(8,8))

# 벽 그리기
plt.plot([0,1], [1,1], color='red', linewidth=2)
plt.plot([2,2], [0,1], color='red', linewidth=2)
plt.plot([2,3], [1,1], color='red', linewidth=2)
plt.plot([3,3], [1,3], color='red', linewidth=2)
plt.plot([1,3], [2,2], color='red', linewidth=2)
plt.plot([2,3], [3,3], color='red', linewidth=2)
plt.plot([1,1], [2,4], color='red', linewidth=2)
plt.plot([1,2], [4,4], color='red', linewidth=2)
plt.plot([3,3], [4,5], color='red', linewidth=2)
plt.plot([4,4], [4,5], color='red', linewidth=2)
plt.plot([4,4], [1,3], color='red', linewidth=2)
plt.plot([4,5], [2,2], color='red', linewidth=2)

# 그림을 그릴 범위 및 눈금 제거 설정
ax = plt.gca(); # 현재 그림의 축(axis) 가져옴
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
plt.tick_params(axis='both', which='both', bottom=False, top=False,
labelbottom=False, right=False, left=False, labelleft=False)

# S0에 녹색 원으로 현재 위치 표시, 여러 개 중 1개 반환할 때 쉼표 사용
line, = ax.plot([0.5], [5], marker="o", color='g', markersize=60)



# 정책 결정하는 초기 파라미터값 설정
theta_0 = np.array(
    [[np.nan, 1, 1, np.nan ], # up, right, down, left
     [np.nan, 1, np.nan, 1 ],
     [np.nan, np.nan, 1, 1 ],
     [np.nan, np.nan, 1, np.nan ],
     [np.nan, np.nan, 1, np.nan ], #s4
     [1, np.nan, 1, np.nan ], #s5
     [np.nan, 1, 1, np.nan ], #s6 
     [1, 1, np.nan, 1 ], #s7
     [1, 1, 1, 1 ],
     [1, np.nan, np.nan, 1 ], #s9
     [1, np.nan, 1, np.nan ], #s10
     [1, np.nan, np.nan, np.nan ],
     [np.nan, np.nan, np.nan, 1 ], #s12
     [1, np.nan, 1, np.nan ], #s13
     [1, np.nan, np.nan, np.nan ], #s14
     [1, 1, np.nan, np.nan],
     [np.nan, 1, 1, 1], #s16
     [np.nan, np.nan, np.nan, 1 ], #s17
     [1, np.nan, 1, np.nan ], #s18
     [np.nan, np.nan, 1, np.nan ], #s19
     [np.nan, 1, np.nan, np.nan ], #s20
     [1, np.nan, np.nan, 1 ],
     [np.nan, 1, np.nan, np.nan ], #s22
     [1, 1, np.nan, 1 ],
     #[1, np.nan, 1, 1 ],
     ]) 

def simple_convert_into_pi_from_theta(theta):
    '''단순히 값의 비율을 계산'''
    [m, n] = theta.shape # theta의 행렬 크기를 구함
    pi = np.zeros((m, n)) # m x n 행렬을 0으로 채움
    for i in range(0, m): # m개의 행에 대한 반복
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :]) # 비율 계산
                                                        # nansum: nan 제외하고 합산
    pi = np.nan_to_num(pi)                              # nan을 0으로 변환
    
    return pi

# 정책 파라미터 theta를 행동 정책 pi로 변환하는 함수
pi_0 = simple_convert_into_pi_from_theta(theta_0)
# 초기 정책 pi_0을 출력
print(pi_0)


def get_next_s(pi, s): # 현재 상태 s에서 정책 pi를 따라 행동한 후 next state 계산
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p=pi[s, :])
    # pi[s,:]의 확률에 따라, direction 값이 선택된다
    if next_direction == "up":
        s_next = s - 5 # 위로 이동하면 상태값이 5 줄어든다
    elif next_direction == "right":
        s_next = s + 1 # 오른쪽으로 이동하면 상태값이 1 늘어난다
    elif next_direction == "down":
        s_next = s + 5 # 아래로 이동하면 상태값이 5 늘어난다
    elif next_direction == "left":
        s_next = s - 1 # 왼쪽으로 이동하면 상태값이 1 줄어든다
    
    return s_next

# 목표 지점에 이를 때까지 에이전트를 계속 이동
def goal_maze(pi):
    s = 0 # 시작 지점 : S0에서 시작
    state_history = [0] # 에이전트의 경로를 기록하는 리스트 초기화
    while (1): # 목표 지점에 이를 때까지 반복
        next_s = get_next_s(pi, s)
        state_history.append(next_s) # 경로 리스트에 다음 상태(위치)를 추가
        if next_s == 24: # 목표 지점에 이르면 종료
            break
        else:
            s = next_s
    return state_history

# 목표 지점에 이를 때까지 미로 안을 이동
state_history = goal_maze(pi_0) 

print(state_history)
print("목표 지점에 이르기까지 걸린 단계 수는 " + str(len(state_history) - 1) + "단계입니다")

import matplotlib.animation as animation
def init():
    '''배경 이미지 초기화'''
    line.set_data([], [])
    return (line,)

def animate(i):
    '''프레임 단위로 이미지 생성'''
    state = state_history[i] # 현재 위치
    x = (state % 5) + 0.5 # x좌표 : (state % 5) 계산 결과 0, 1, 2
    y = 4.5 - int(state / 5) # y좌표 : (state / 5) 계산 결과 0, 1, 2
    line.set_data(x, y)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history), interval=200, repeat=False)

plt.show()
