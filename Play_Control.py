import sys
import pypot.dynamixel
import time
import math
import pygame
from pygame.locals import *
import numpy as np
from multiprocessing import shared_memory

# グローバル定数の定義
ANGLE_LIMIT = 85          # モーターの角度制限
DEFAULT_TORQUE_LIMIT = 100  # デフォルトトルク制限（%）
TORQUE_MARGIN = 5         # トルク余裕値（%）
RETURN_STEPS = 20         # 初期位置復帰ステップ数
DELAY_TIME = 0.01        # 基本遅延時間

# Dynamixel Protocol 2.0のアドレス定義
GOAL_POSITION_ADDRESS = 116  # Goal Positionのアドレス（Protocol 2.0）
GOAL_POSITION_LENGTH = 4     # Goal Positionのデータ長（4バイト）

BIAS_RATE = 0.3            # バイアス変化率
MAX_BIAS = 45.0              # バイアスの最大値
BIAS_DECAY_RATE = 0.5     # バイアス減衰率

# 横方向捻転のためのパラメータ
ROLL_RATE = 1           # 捻転角速度
MAX_ROLL = 180.0          # 最大捻転角度（制限なし）
ROLL_DECAY_RATE = 0    # 捻転角減衰率

# PS4コントローラーのデッドゾーン設定
DEADZONE = 0.2            # アナログスティックのデッドゾーン

# 捻転螺旋運動用の定数
SPIRAL_GRIP_TORQUE = 100    # 螺旋運動時の把持トルク制限
TORQUE_MONITOR_INTERVAL = 0.05  # トルク監視間隔

# バイアス管理用のグローバル変数
current_bias = 0           # 現在のバイアス値
bias_velocity = 0          # バイアスの変化速度

# 捻転管理用のグローバル変数
current_roll = 0           # 現在の捻転角度
roll_velocity = 0          # 捻転の変化速度

# 匍匐運動用の操舵制御パラメータ
STEERING_ANGLE = 30.0        # 操舵時の指定角度（度）
STEERING_PROPAGATION_TIME = 5  # 角度伝播の指定時間（秒）
STEERING_STEPS = 10          # 角度伝播のステップ数

# 操舵制御用のグローバル変数
steering_queue = []          # 角度伝播のキュー
steering_active = False      # 操舵制御が有効かどうか
last_steering_time = 0       # 最後の操舵入力時刻

# 指令値(21個)、現在角度(21個)、バイアス(1個)、捻転角度(1個)
shared_data = np.zeros(23, dtype=np.int64)
shm = shared_memory.SharedMemory(create=True, size=shared_data.nbytes, name="mujoco_data")
shared_buffer = np.ndarray(shared_data.shape, dtype=shared_data.dtype, buffer=shm.buf)
shared_buffer[:] = shared_data[:]

# コントローラーの状態管理
class ControllerState:
    def __init__(self):
        self.controller = None
        self.connected = False
        self.axis_values = {}
        self.button_states = {}
        
    def update(self):
        """コントローラーの状態を更新（無線接続の切断検知付き）"""
        if not self.connected:
            return
            
        try:
            # アナログスティックの値を取得
            self.axis_values['left_x'] = self.apply_deadzone(self.controller.get_axis(0))
            self.axis_values['left_y'] = self.apply_deadzone(self.controller.get_axis(1))
            self.axis_values['right_x'] = self.apply_deadzone(self.controller.get_axis(2))
            self.axis_values['right_y'] = self.apply_deadzone(self.controller.get_axis(3))
            
            # 十字キーの値を取得（複数のパターンに対応）
            dpad_x_value = 0.0
            dpad_y_value = 0.0
            
            # パターン1: 軸として認識される場合
            if self.controller.get_numaxes() > 6:
                dpad_x_value = self.controller.get_axis(6)
            
            if self.controller.get_numaxes() > 7:
                dpad_y_value = self.controller.get_axis(7)
            
            # パターン2: ボタンとして認識される場合（複数の可能性をチェック）
            if dpad_x_value == 0.0:  # 軸で値が取得できない場合
                # 一般的なPS4コントローラーのボタン配置
                button_mappings = [
                    (14, 15),  # 左、右
                    (7, 5),    # 別のマッピング
                    (11, 12),  # さらに別のマッピング
                ]
                
                for left_btn, right_btn in button_mappings:
                    if (self.controller.get_numbuttons() > max(left_btn, right_btn)):
                        dpad_left = self.controller.get_button(left_btn)
                        dpad_right = self.controller.get_button(right_btn)
                        if dpad_left or dpad_right:
                            dpad_x_value = -1.0 if dpad_left else (1.0 if dpad_right else 0.0)
                            break
            
            # パターン3: すべてのボタンをチェックしてデバッグ
            if dpad_x_value == 0.0:
                # デバッグ用：すべてのボタンの状態を確認
                pressed_buttons = []
                for i in range(self.controller.get_numbuttons()):
                    if self.controller.get_button(i):
                        pressed_buttons.append(i)
                
                # 押されているボタンがあれば表示（デバッグ用）
                if pressed_buttons:
                    print(f"押されているボタン: {pressed_buttons}")
            
            self.axis_values['dpad_x'] = dpad_x_value
            self.axis_values['dpad_y'] = dpad_y_value
            
            # ボタンの状態を取得
            for i in range(self.controller.get_numbuttons()):
                self.button_states[i] = self.controller.get_button(i)
                
        except pygame.error as e:
            print(f"\nコントローラー接続が切断されました: {e}")
            print("キーボード操作に切り替えます。")
            self.connected = False
            self.controller = None
        except Exception as e:
            print(f"\nコントローラー更新エラー: {e}")
            self.connected = False
    
    def apply_deadzone(self, value):
        """デッドゾーンを適用"""
        if abs(value) < DEADZONE:
            return 0.0
        return value
    
    def get_axis(self, axis_name):
        """軸の値を取得"""
        return self.axis_values.get(axis_name, 0.0)
    
    def get_button(self, button_id):
        """ボタンの状態を取得"""
        return self.button_states.get(button_id, False)

# PS4コントローラーの初期化
def initialize_controller():
    """PS4コントローラーを初期化（無線接続対応）"""
    controller_state = ControllerState()
    
    try:
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        
        if joystick_count == 0:
            print("PS4コントローラーが見つかりません。")
            print("無線接続の確認事項:")
            print("1. PS4コントローラーがBluetoothでペアリングされているか確認")
            print("2. PSボタンを押してコントローラーを起動")
            print("3. コントローラーのライトバーが点灯しているか確認")
            print("キーボード操作を使用します。")
            return controller_state
        
        # PS4コントローラーを検索
        ps4_controller = None
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            controller_name = joystick.get_name().lower()
            
            # PS4コントローラーの識別
            if any(keyword in controller_name for keyword in ['ps4', 'dualshock', 'wireless controller', 'sony']):
                ps4_controller = joystick
                controller_index = i
                break
            else:
                joystick.quit()
        
        if ps4_controller is None:
            print(f"PS4コントローラーが見つかりません。検出されたコントローラー:")
            for i in range(joystick_count):
                joystick = pygame.joystick.Joystick(i)
                joystick.init()
                print(f"  {i}: {joystick.get_name()}")
                joystick.quit()
            print("キーボード操作を使用します。")
            return controller_state
        
        controller_state.controller = ps4_controller
        controller_state.connected = True
        
        print(f"PS4コントローラーが接続されました: {ps4_controller.get_name()}")
        print(f"コントローラー番号: {controller_index}")
        print(f"軸数: {ps4_controller.get_numaxes()}")
        print(f"ボタン数: {ps4_controller.get_numbuttons()}")
        
        # 接続テスト
        print("接続テスト中...")
        test_passed = True
        try:
            # 基本的な軸とボタンの動作確認
            for i in range(min(4, ps4_controller.get_numaxes())):
                ps4_controller.get_axis(i)
            for i in range(min(10, ps4_controller.get_numbuttons())):
                ps4_controller.get_button(i)
            print("接続テスト成功！")
        except Exception as e:
            print(f"接続テスト失敗: {e}")
            test_passed = False
        
        if not test_passed:
            print("コントローラーの動作に問題があります。キーボード操作を使用します。")
            controller_state.connected = False
            ps4_controller.quit()
        
    except Exception as e:
        print(f"コントローラー初期化エラー: {e}")
        print("キーボード操作を使用します。")
    
    return controller_state

# 接続確認と電源チェック関数
def check_connection_and_power():
    print("Checking connection and power...")
    try:
        port = "COM4"  # 固定のポート
        try:
            print(f"Attempting to connect to port {port}...")
            dxl_io = pypot.dynamixel.DxlIO(port)
            ids = dxl_io.scan()

            if ids:
                print(f"Detected {len(ids)} motors. IDs: {ids}")
                print("Power is ON and motors are responsive.")
                dxl_io.close()
                return True
            else:
                print("No motors detected. Please check if the robot is powered on.")
                return False
        except Exception as e:
            print(f"Error on port {port}: {e}")
            return False

    except Exception as e:
        print(f"Connection check error: {e}")
        return False

# モーター接続確認関数
def check_motor_connections(dxl_io, ids):
    """
    モーターの接続を確認
    scan()で既に検出されているため、追加のpingは不要
    """
    # scan()で取得したIDsをそのまま返す
    # ping処理は通信エラーを起こす可能性があるため削除
    print(f"Confirmed {len(ids)} motors connected")
    return ids

# トルク監視関数
def monitor_torque(dxl_io, motor_id):
    """
    指定されたモーターの現在のトルク値を取得
    """
    try:
        # Dynamixelの現在負荷値を取得（0-1023の範囲）
        load_value = dxl_io.get_present_load([motor_id])[0]
        # 負荷値をトルクパーセンテージに変換
        torque_percentage = abs(load_value) / 1023.0 * 100
        return torque_percentage
    except Exception as e:
        print(f"トルク監視エラー (Motor {motor_id}): {e}")
        return 0

# 安全な初期位置復帰関数
def safe_return_to_zero(dxl_io, ids, steps=RETURN_STEPS, delay=0.1):
    try:
        current_positions = dxl_io.get_present_position(ids)
        position_dict = dict(zip(ids, current_positions))

        dxl_io.set_torque_limit(dict(zip(ids, [30] * len(ids))))

        for step in range(steps):
            target_positions = {}
            for motor_id in ids:
                remaining_ratio = (steps - step) / steps
                target_positions[motor_id] = position_dict[motor_id] * remaining_ratio

            if not safe_set_goal_position(dxl_io, target_positions):
                raise Exception("Failed to execute safe return")
            time.sleep(delay)

        final_positions = dict(zip(ids, [0] * len(ids)))
        safe_set_goal_position(dxl_io, final_positions)

        dxl_io.set_torque_limit(dict(zip(ids, [DEFAULT_TORQUE_LIMIT] * len(ids))))

    except Exception as e:
        print(f"Error during safe return: {e}")
        raise

# 安全な位置設定関数（Sync Write対応版）
def safe_set_goal_position(dxl_io, target_positions, retries=3, delay=0.1):
    """
    Sync Writeを使ってモーターの目標位置を1回の通信で安全に設定
    現在角度も共有メモリに書き込む
    """
    values = [*target_positions.values()]

    # 指令値を共有メモリの前半に書き込む
    shared_buffer[:21] = values

    # バイアスと捻転角度を書き込む
    shared_buffer[21] = int(current_bias)
    shared_buffer[22] = int(current_roll)

    # Sync Writeで一括送信
    motor_ids = list(target_positions.keys())
    positions = list(target_positions.values())

    for attempt in range(retries):
        try:
            # 角度を位置値に変換（Dynamixel Protocol 2.0）
            # pypotは角度（degree）を使用し、内部で位置値に変換
            # 位置範囲: 0-4095 (0x000-0xFFF)
            # 角度範囲: -150度～+150度
            motor_values = []
            for angle in positions:
                # 角度を位置値に変換（-150度=0, 0度=2047.5, +150度=4095）
                position_value = int((angle + 150.0) * 4095.0 / 300.0)
                # 範囲制限
                position_value = max(0, min(4095, position_value))
                motor_values.append(position_value)

            # Sync Writeで一括送信（1回の通信で全モーターに送信）
            # データ形式: [[motor1_value], [motor2_value], ...]
            data = [[value] for value in motor_values]
            dxl_io._DxlIO__controller.sync_write(
                motor_ids,
                data,
                GOAL_POSITION_ADDRESS,
                GOAL_POSITION_LENGTH
            )
            return True

        except AttributeError:
            # sync_writeが使えない場合は従来の方法にフォールバック
            if attempt == 0:
                print("Warning: Sync Write not available, using standard method")
            try:
                dxl_io.set_goal_position(target_positions)
                return True
            except Exception as e:
                print(f"Position setting error: {e}")
                time.sleep(delay)

        except Exception as e:
            print(f"Sync write error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)

    print("Failed to set position after multiple attempts")
    return False

# 捻転角度に基づいてバイアスを適用するモーターの種類を決定する関数
def determine_bias_motor_type(roll_angle):
    """
    捻転角度に基づいて、バイアスを適用すべきモーターの種類を決定する
    返り値: "yaw" または "pitch"
    """
    # 角度を0-360度の範囲に正規化
    normalized_angle = roll_angle % 360
    if normalized_angle < 0:
        normalized_angle += 360
    
    # 要件に従ってモーター種類を決定
    # -45～45度, 135～225度 → ヨー軸モーター
    # 45～135度, 225～315度 → ピッチ軸モーター
    
    # 角度範囲の判定
    if (0 <= normalized_angle <= 45) or (135 <= normalized_angle <= 225) or (315 <= normalized_angle < 360):
        return "yaw"
    else:  # (45 < normalized_angle < 135) or (225 < normalized_angle < 315)
        return "pitch"

# 蛇行運動関数（捻転機能を追加）
def serpenoid_motion(dxl_io, ids, phase, bias, roll=0, amplitude=60, frequency=0.025, direction=0):
    """
    蛇行運動の実装（捻転角度を考慮してバイアスを適用するモーターを切り替え）
    """
    target_positions = {id: 0 for id in ids}
    total_segments = len(ids) // 2
    
    # 捻転角度に基づいてバイアスを適用するモーターの種類を決定
    bias_motor_type = determine_bias_motor_type(roll)
    
    # バイアスを適用するモーターの種類を表示
    bias_display = "ヨー軸" if bias_motor_type == "yaw" else "ピッチ軸"
    
    for i, motor_id in enumerate(ids):
        segment_index = i // 2
        is_yaw_motor = (i % 2 == 0)  # 偶数インデックスはヨー軸
        
        if is_yaw_motor:  # 横向きモーター（ヨー軸）
            # 基本の蛇行形状を計算
            base_angle = amplitude * math.sin(2 * math.pi * frequency * -(phase) + segment_index * math.pi / 3)
            
            # ヨー軸モーターにバイアスを適用する場合
            if bias_motor_type == "yaw":
                base_angle += bias
                
            # 方向調整を追加
            yaw_angle = base_angle + direction * (total_segments - segment_index) / total_segments * 20
            
            # 捻転の影響を考慮（ロール角に応じて振幅変調）
            roll_factor = math.cos(math.radians(roll))
            target_positions[motor_id] = yaw_angle * roll_factor
        else:  # 縦向きモーター（ピッチ軸）
            # 基本の縦方向の動きを計算
            base_pitch_angle = amplitude * math.sin(2 * math.pi * frequency * -(phase) + segment_index * math.pi / 3) * math.sin(math.radians(roll))
            
            # ピッチ軸モーターにバイアスを適用する場合
            if bias_motor_type == "pitch":
                # 捻転角度に応じてバイアスの効果を調整
                sin_roll = math.sin(math.radians(roll))
                base_pitch_angle += bias * sin_roll if sin_roll != 0 else bias
                
            target_positions[motor_id] = base_pitch_angle

    safe_set_goal_position(dxl_io, target_positions)

# 捻転螺旋運動関数
def spiral_torsion_motion(dxl_io, ids, phase, amplitude=90, frequency=0.05, direction=0):
    """
    トルク制御付き捻転螺旋運動の実装
    """
    target_positions = {id: 0 for id in ids}

    # トルク制限の設定
    torque_limits = dict(zip(ids, [SPIRAL_GRIP_TORQUE] * len(ids)))
    dxl_io.set_torque_limit(torque_limits)

    for i, motor_id in enumerate(ids):
        if i % 2 == 0:  # 横向きモーター（ヨー軸）
            angle = amplitude * math.sin(2 * math.pi * frequency * phase + i * math.pi / 10)
            target_positions[motor_id] = angle
        else:  # 縦向きモーター（ピッチ軸）
            angle = amplitude * math.cos(2 * math.pi * frequency * phase + i * math.pi / 10)
            target_positions[motor_id] = angle

        # トルク監視と調整
        current_torque = monitor_torque(dxl_io, motor_id)
        if current_torque > SPIRAL_GRIP_TORQUE - TORQUE_MARGIN:
            target_positions[motor_id] *= 0.95

    # 上昇・下降制御
    if direction != 0:
        for i in range(len(ids)):
            if i % 2 == 0:  # ヨー軸モーターのみに上昇・下降の影響を適用
                target_positions[ids[i]] += direction * 15

    safe_set_goal_position(dxl_io, target_positions)

# PS4コントローラー用バイアス計算関数
def calculate_dynamic_bias_controller(controller_state):
    global current_bias, bias_velocity
    
    # 右スティックの左右入力を使用（左スティックから変更）
    right_stick_x = controller_state.get_axis('right_x')
    
    # バイアス変化量の計算
    if abs(right_stick_x) > 0:
        # スティックの入力に応じてバイアス速度を調整
        bias_velocity = -right_stick_x * BIAS_RATE
    else:
        # 入力がない場合はバイアスを減衰
        bias_velocity *= BIAS_DECAY_RATE
    
    current_bias += bias_velocity
    
    # バイアスの最大値制限
    current_bias = max(min(current_bias, MAX_BIAS), -MAX_BIAS)
    
    return current_bias

# PS4コントローラー用捻転角度計算関数
def calculate_roll_angle_controller(controller_state):
    global current_roll, roll_velocity
    
    # 十字キーの左右入力を使用
    dpad_x = controller_state.get_axis('dpad_x')
    
    # 捻転角変化量の計算
    if abs(dpad_x) > 0:
        # 十字キーの入力に応じて捻転速度を調整
        roll_velocity = -dpad_x * ROLL_RATE
    else:
        # 入力がない場合は捻転速度を減衰
        roll_velocity *= ROLL_DECAY_RATE
    
    current_roll += roll_velocity
    
    # 捻転角度は制限なし（360度以上回転可能）
    return current_roll

# キーボード用バイアス計算関数（従来版）
def calculate_dynamic_bias_keyboard(keys):
    global current_bias, bias_velocity
    
    # 左右キーの入力強度を検出
    left_intensity = keys[K_LEFT]
    right_intensity = keys[K_RIGHT]
    
    # バイアス変化量の計算
    if left_intensity:
        # 左キー押下時は負のバイアス
        bias_velocity += BIAS_RATE
    elif right_intensity:
        # 右キー押下時は正のバイアス
        bias_velocity -= BIAS_RATE
    else:
        # キー入力がない場合はバイアスを減衰
        bias_velocity *= BIAS_DECAY_RATE
    
    # バイアス値の制限
    bias_velocity = max(min(bias_velocity, BIAS_RATE), -BIAS_RATE)
    current_bias += bias_velocity
    
    # バイアスの最大値制限
    current_bias = max(min(current_bias, MAX_BIAS), -MAX_BIAS)
    
    return current_bias

# キーボード用捻転角度計算関数（従来版）
def calculate_roll_angle_keyboard(keys):
    global current_roll, roll_velocity
    
    # A/Dキーの入力強度を検出
    a_intensity = keys[K_a]
    d_intensity = keys[K_d]
    
    # 捻転角変化量の計算
    if a_intensity:
        # Aキー押下時は正の捻転（左回転）
        roll_velocity += ROLL_RATE
    elif d_intensity:
        # Dキー押下時は負の捻転（右回転）
        roll_velocity -= ROLL_RATE
    else:
        # キー入力がない場合は捻転速度を減衰
        roll_velocity *= ROLL_DECAY_RATE
    
    # 捻転速度の制限
    roll_velocity = max(min(roll_velocity, ROLL_RATE * 2), -ROLL_RATE * 2)
    current_roll += roll_velocity
    
    # 捻転角度は制限なし（360度以上回転可能）
    return current_roll

# 体幹を維持した捻転運動
def sinus_lifting_motion(dxl_io, ids, phase, amplitude=60, frequency=0.025, direction=0):
    target_positions = {id: 0 for id in ids}
    for i, motor_id in enumerate(ids):
        if i % 2 != 0:  # 縦向きモーターのみ制御
            target_positions[motor_id] = amplitude *math.sin(10 * phase * np.pi/180)* math.sin(2 * math.pi + i * math.pi / 4)
        else:
            target_positions[motor_id] = amplitude *math.cos(10 * phase * np.pi/180)* math.sin(2 * math.pi + i * math.pi / 4)

    safe_set_goal_position(dxl_io, target_positions)

# 制御状態表示関数
def display_control_status(phase, bias, roll, controller_connected=False):
    # 表示用に-180～180の範囲に正規化した捻転角
    display_roll = ((roll + 180) % 360) - 180
    
    # 捻転角度に基づいてバイアスを適用するモーターの種類を決定
    bias_motor_type = determine_bias_motor_type(roll)
    bias_display = "ヨー軸" if bias_motor_type == "yaw" else "ピッチ軸"
    
    # 制御状態を一行で表示
    controller_status = "PS4" if controller_connected else "キーボード"
    print(f"\r[{controller_status}] バイアス: {bias:.1f} ({bias_display}), 捻転角: {display_roll:.1f}°  ", end='', flush=True)

# クローラー運動関数（論文のS-crawler gaitを実装）
def crawler_motion(dxl_io, ids, phase, amplitude=30, frequency=0.02):
    """
    クローラー運動の実装（論文のS-crawler gait）
    L = 26L0, B1 = 4.8π/L, B2 = 2.5π/L の設定を使用
    """
    target_positions = {id: 0 for id in ids}
    
    # 論文の式(18)に基づくパラメータ設定
    L = 21 * 0.0684 # セグメント長を考慮した周期長
    B1 = 0.048 * math.pi / L
    B2 = 0.025 * math.pi / L
    
    for i, motor_id in enumerate(ids):
        segment_index = i / 2  # セグメント位置
        is_yaw_motor = (i % 2 == 0)  # 偶数インデックスはヨー軸
        
        # 現在の位置パラメータ
        s = segment_index + phase * frequency
        
        if is_yaw_motor:  # ヨー軸モーター
            # 論文の式(18): κ(s) = 4π/L + B1·sin(4πs/L - π/2)
            curvature = (4 * math.pi / L) + B1 * math.sin(4 * math.pi * s / L - math.pi / 2)
            # ψ(s) = 0 の場合のヨー軸角度計算
            target_positions[motor_id] = amplitude * curvature * math.cos(0)
            
        else:  # ピッチ軸モーター
            # 論文の式(18): τ(s) = B2·sin(2πs/L)
            torsion = B2 * math.sin(2 * math.pi * s / L)
            # ピッチ軸への影響を計算
            target_positions[motor_id] = amplitude * torsion
    
    safe_set_goal_position(dxl_io, target_positions)

# 操舵制御クラス
class SteeringController:
    def __init__(self, total_motors):
        self.total_motors = total_motors
        self.yaw_motor_count = (total_motors + 1) // 2  # ヨー軸モーターの数
        self.propagation_queue = []  # 各モーターの角度伝播状態
        self.active = False
        
    def start_steering(self, angle):
        """操舵制御を開始"""
        self.active = True
        self.propagation_queue.clear()
        
        # 各ヨー軸モーターに対して伝播タイミングを設定
        for i in range(self.yaw_motor_count):
            motor_index = i * 2  # ヨー軸モーターのインデックス（0, 2, 4, ...）
            delay = i * (STEERING_PROPAGATION_TIME / self.yaw_motor_count)
            self.propagation_queue.append({
                'motor_index': motor_index,
                'target_angle': angle,
                'start_time': time.time() + delay,
                'active': False,
                'current_angle': 0.0
            })
    
    def update(self):
        """操舵制御を更新"""
        if not self.active:
            return {}
        
        current_time = time.time()
        steering_angles = {}
        all_finished = True
        
        for prop in self.propagation_queue:
            if not prop['active'] and current_time >= prop['start_time']:
                prop['active'] = True
            
            if prop['active']:
                # 角度を徐々に目標角度に近づける
                diff = prop['target_angle'] - prop['current_angle']
                if abs(diff) > 0.1:  # 閾値以上の差がある場合
                    prop['current_angle'] += diff * 0.1  # 10%ずつ近づける
                    all_finished = False
                else:
                    prop['current_angle'] = prop['target_angle']
                
                steering_angles[prop['motor_index']] = prop['current_angle']
            else:
                all_finished = False
        
        # すべての伝播が完了したら操舵制御を終了
        if all_finished:
            self.active = False
            self.propagation_queue.clear()
        
        return steering_angles
    
    def stop(self):
        """操舵制御を停止"""
        self.active = False
        self.propagation_queue.clear()

# グローバルな操舵制御インスタンス
steering_controller = None

# 修正された匍匐運動関数
def peristaltic_motion(dxl_io, ids, phase, amplitude=30, frequency=0.045, direction=0):
    """
    匍匐運動の実装（操舵制御付き）
    """
    global steering_controller
    
    # 操舵制御インスタンスの初期化
    if steering_controller is None:
        steering_controller = SteeringController(len(ids))
    
    target_positions = {id: 0 for id in ids}
    
    # 基本の匍匐運動
    for i, motor_id in enumerate(ids):
        if i % 2 != 0:  # ピッチ軸モーター（奇数インデックス）
            target_positions[motor_id] = amplitude * math.sin(2 * math.pi * frequency * phase + i * math.pi / 3)
        else:  # ヨー軸モーター（偶数インデックス）
            target_positions[motor_id] = 0
    
    # 操舵制御の角度を重ね合わせ
    steering_angles = steering_controller.update()
    for motor_index, steering_angle in steering_angles.items():
        if motor_index < len(ids):
            motor_id = ids[motor_index]
            target_positions[motor_id] += steering_angle
    
    safe_set_goal_position(dxl_io, target_positions)

# 操舵入力を処理する関数
def handle_steering_input(controller_state=None, keys=None):
    """
    操舵入力を処理し、必要に応じて操舵制御を開始
    """
    global steering_controller, last_steering_time
    
    if steering_controller is None:
        return
    
    current_time = time.time()
    
    # 入力の検出
    steering_input = 0
    
    if controller_state and controller_state.connected:
        # PS4コントローラーの右スティック左右
        right_stick_x = controller_state.get_axis('right_x')
        if abs(right_stick_x) > DEADZONE:
            steering_input = right_stick_x
    elif keys:
        # キーボードの左右キー
        if keys[K_LEFT]:
            steering_input = -1
        elif keys[K_RIGHT]:
            steering_input = 1
    
    # 操舵制御の開始（入力間隔を制御）
    if steering_input != 0 and (current_time - last_steering_time) > 0.1:
        steering_angle = STEERING_ANGLE * steering_input
        steering_controller.start_steering(steering_angle)
        last_steering_time = current_time
        print(f"操舵制御開始: {steering_angle:.1f}度")

# 初期化時に操舵制御を停止する関数
def initialize_steering_control():
    """
    操舵制御の初期化
    """
    global steering_controller
    if steering_controller:
        steering_controller.stop()

# 終了処理に追加する関数
def cleanup_steering_control():
    """
    操舵制御の終了処理
    """
    global steering_controller
    if steering_controller:
        steering_controller.stop()
        steering_controller = None

# メイン関数
def main():
    """
    メインプログラムの実行
    """
    try:
        # システム初期化と接続確認
        if not check_connection_and_power():
            raise IOError("System initialization failed")

        port = "COM4"  # 固定のポート
        dxl_io = pypot.dynamixel.DxlIO(port)
        print('Connected to port:', port)

        # モーターの検出と初期化
        ids = dxl_io.scan()
        if not ids:
            raise IOError("No motors detected")
        print('Found motors:', ids)

        # モーター接続の確認
        connected_ids = check_motor_connections(dxl_io, ids)
        if len(connected_ids) != len(ids):
            raise IOError("Some motors are not responding")

        # 初期トルク設定
        torque_limits = dict(zip(ids, [DEFAULT_TORQUE_LIMIT] * len(ids)))
        dxl_io.set_torque_limit(torque_limits)

        # 安全な初期位置への移動
        safe_return_to_zero(dxl_io, ids)

        # 操舵制御の初期化
        initialize_steering_control()

        # Pygame初期化
        pygame.init()
        pygame.display.set_mode((100, 100))

        # PS4コントローラーの初期化
        controller_state = initialize_controller()

        # 運動パラメータの初期化
        current_motion = None
        phase = 0

        # 操作説明の表示
        print("\n" + "="*50)
        print("PS4コントローラー無線接続対応 ヘビ型ロボット制御")
        print("="*50)
        
        if controller_state.connected:
            print("✓ PS4コントローラー（無線）で操作中")
            print("\n--- PS4コントローラー操作 ---")
            print("△ボタン      : 蛇行運動")
            print("□ボタン       : 匍匐運動")
            print("○ボタン       : 体幹を維持した捻転運動")
            print("×ボタン       : クローラー運動")
            print("L1ボタン      : 捻転螺旋運動")
            print("左スティック↑↓: 前進/後退")
            print("右スティック←→: ロボット方向転換（蛇行運動）/ 操舵制御（匍匐運動）")
            print("十字キー←→    : 横方向への転がり移動（捻転）")
            print("OPTIONSボタン  : 終了")
            print("\n※ コントローラーが切断された場合、自動でキーボード操作に切り替わります")
        else:
            print("⚠ PS4コントローラーが検出されません - キーボードで操作")
            print("\n--- キーボード操作 ---")
            print("1キー    : 蛇行運動")
            print("2キー    : 匍匐運動")
            print("3キー    : 体幹を維持した捻転運動")
            print("4キー    : クローラー運動")
            print("5キー    : 捻転螺旋運動")
            print("W/Sキー  : 前進/後退")
            print("←/→キー : ロボット方向転換（蛇行運動）/ 操舵制御（匍匐運動）")
            print("A/Dキー  : 横方向への転がり移動（捻転）")
            print("ESCキー  : 終了")
        
        print("\n" + "="*50)
        print("捻転角度に応じたバイアス制御:")
        print("・0～90度, 180～270度  : ヨー軸モーターにバイアス")
        print("・90～180度, 270～360度 : ピッチ軸モーターにバイアス")
        print("="*50)
        print("匍匐運動時の操舵制御:")
        print("・左右キー/右スティック左右で操舵")
        print("・角度は後方のヨー軸モーターに順番に伝播")
        print("・伝播時間: 0.5秒, 操舵角度: 30度")
        print("="*50)
        print("捻転螺旋運動:")
        print("・トルク制御により把持力を調整")
        print("・上昇/下降制御対応")
        print("・把持トルク制限: 30%")
        print("="*50 + "\n")

        # メインループ
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            # コントローラーの状態を更新
            if controller_state.connected:
                controller_state.update()
                
                # PS4コントローラーでの操作
                # OPTIONSボタン（ボタン9）で終了
                if controller_state.get_button(9):  # OPTIONSボタンは通常ボタン9
                    running = False

                # モーション選択（実際のPS4コントローラーに合わせて修正）
                if controller_state.get_button(3):  # △ボタン（実際はボタン3）
                    current_motion = serpenoid_motion
                    print("蛇行運動を選択")
                elif controller_state.get_button(1):  # ○ボタン（実際はボタン1）
                    current_motion = sinus_lifting_motion
                    print("体幹を維持した捻転運動を選択")
                elif controller_state.get_button(0):  # ×ボタン（実際はボタン0）
                    current_motion = crawler_motion  # peristaltic_motionから変更
                    print("クローラー運動を選択")
                elif controller_state.get_button(2):  # □ボタン（実際はボタン2）
                    current_motion = peristaltic_motion
                    print("匍匐運動を選択")
                elif controller_state.get_button(4):  # L1ボタン（実際はボタン4）
                    current_motion = spiral_torsion_motion
                    print("捻転螺旋運動を選択")

                # 前進/後退（左スティック上下）
                left_stick_y = controller_state.get_axis('left_y')
                if abs(left_stick_y) > 0:
                    phase -= left_stick_y * 0.1  # 上が負の値なので符号を反転
                
            else:
                # キーボードでの操作（従来通り）
                keys = pygame.key.get_pressed()
                
                # ESCキーで終了
                if keys[K_ESCAPE]:
                    running = False
                
                # モーション選択
                if keys[K_1]:
                    current_motion = serpenoid_motion
                    print("蛇行運動を選択")
                elif keys[K_2]:
                    current_motion = peristaltic_motion
                    print("匍匐運動を選択")
                elif keys[K_3]:
                    current_motion = sinus_lifting_motion
                    print("体幹を維持した捻転運動を選択")
                elif keys[K_4]:
                    current_motion = crawler_motion
                    print("クローラー運動を選択")
                elif keys[K_5]:
                    current_motion = spiral_torsion_motion
                    print("捻転螺旋運動を選択")

                # 基本動作制御
                if keys[K_w]:
                    phase += 0.1
                elif keys[K_s]:
                    phase -= 0.1

            # 選択された運動の実行
            if current_motion:
                try:
                    if current_motion == serpenoid_motion:
                        if controller_state.connected:
                            # PS4コントローラーでの制御
                            dynamic_bias = calculate_dynamic_bias_controller(controller_state)
                            roll_angle = calculate_roll_angle_controller(controller_state)
                        else:
                            # キーボードでの制御
                            keys = pygame.key.get_pressed()
                            dynamic_bias = calculate_dynamic_bias_keyboard(keys)
                            roll_angle = calculate_roll_angle_keyboard(keys)
                        
                        # 蛇行運動に捻転角度を追加
                        current_motion(dxl_io, ids, phase, bias=dynamic_bias, roll=roll_angle)
                        
                        # 制御状態の表示
                        display_control_status(phase, dynamic_bias, roll_angle, controller_state.connected)
                    elif current_motion == peristaltic_motion:
                        # 操舵入力の処理
                        if controller_state.connected:
                            handle_steering_input(controller_state=controller_state)
                        else:
                            keys = pygame.key.get_pressed()
                            handle_steering_input(keys=keys)
                        
                        # 匍匐運動の実行
                        current_motion(dxl_io, ids, phase)
                        
                        # 操舵制御の状態表示
                        if steering_controller and steering_controller.active:
                            print(f"\r匍匐運動中 - 操舵制御実行中  ", end='', flush=True)
                        else:
                            print(f"\r匍匐運動中  ", end='', flush=True)
                    elif current_motion == sinus_lifting_motion:
                        current_motion(dxl_io, ids, phase)
                    elif current_motion == crawler_motion:
                        current_motion(dxl_io, ids, phase)
                        print(f"\rクローラー運動中  ", end='', flush=True)
                    elif current_motion == spiral_torsion_motion:
                        # 捻転螺旋運動の実行
                        direction = 0
                        if controller_state.connected:
                            # PS4コントローラーの右スティック上下で上昇/下降制御
                            right_stick_y = controller_state.get_axis('right_y')
                            if abs(right_stick_y) > DEADZONE:
                                direction = -right_stick_y  # 上が負の値なので符号を反転
                        else:
                            # キーボードのQ/Eキーで上昇/下降制御
                            keys = pygame.key.get_pressed()
                            if keys[K_q]:
                                direction = 1  # 上昇
                            elif keys[K_e]:
                                direction = -1  # 下降
                        
                        current_motion(dxl_io, ids, phase, direction=direction)
                        
                        # 状態表示
                        direction_str = "上昇" if direction > 0 else ("下降" if direction < 0 else "水平")
                        print(f"\r捻転螺旋運動中 - {direction_str}  ", end='', flush=True)

                except Exception as e:
                    print(f"Motion execution error: {e}")
                    # エラー発生時は安全のため停止
                    running = False

            # 制御周期の調整
            time.sleep(DELAY_TIME)

    except KeyboardInterrupt:
        print("\nプログラムがユーザーによって中断されました")
    except Exception as e:
        print(f"\n重大なエラーが発生しました: {e}")
    finally:
        # 終了処理
        try:
            # 操舵制御の終了処理
            cleanup_steering_control()

            if 'dxl_io' in locals() and 'ids' in locals():
                print("\n初期位置に戻しています...")
                safe_return_to_zero(dxl_io, ids)
                time.sleep(1)
                print("モーターを初期位置に戻しました")
                dxl_io.close()

        except Exception as e:
            print(f"終了処理中のエラー: {e}")

        pygame.quit()
        print("プログラムを終了しました")

        # 共有メモリの解放
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass  # 共有メモリが既に解放されている場合は無視

if __name__ == "__main__":
    main()