from lux.utils import direction_to  # 2点間の方向（移動アクション）を決定するユーティリティ関数
import sys
import numpy as np

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        # プレイヤーIDを保存
        self.player = player
        # 敵のプレイヤーIDを決定（"player_0"なら敵は"player_1"、その逆）
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        # 自分のチームIDを設定（"player_0"ならチーム0、"player_1"ならチーム1）
        self.team_id = 0 if self.player == "player_0" else 1
        # 敵チームIDの設定（自チームが0なら敵は1、逆も同様）
        self.opp_team_id = 1 if self.team_id == 0 else 0

        # デバッグや再現性のために乱数シードを固定
        np.random.seed(0)
        # 環境設定情報（env_cfg）を保持
        self.env_cfg = env_cfg
        
        # 見つけたレリックノードの位置を保持するリスト
        self.relic_node_positions = []
        # 既に発見済みのレリックノードのIDを保持するセット
        self.discovered_relic_nodes_ids = set()
        # 各ユニットごとに探索先のランダムな目標地点を記録する辞書
        self.unit_explore_locations = dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        各利用可能なユニットに対して送るアクションを決定する関数
        :param step: 現在のゲームのタイムステップ (0から始まり、最大ステップ数まで)
        :param obs: 環境からの観測情報
        :param remainingOverageTime: 残りの余分な計算時間（デフォルトは60）
        :return: 各ユニットに対するアクションを表す配列
        """
        # 自分のチームのユニットが存在するかどうかのマスク (Trueならそのユニットが存在)
        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape: (max_units,)
        # 自分のチームのユニットの位置情報 (x, y座標)
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape: (max_units, 2)
        # 自分のチームのユニットのエネルギー情報
        unit_energys = np.array(obs["units"]["energy"][self.team_id])  # shape: (max_units, 1)
        # 観測可能なレリックノードの位置情報（全レリックノードの位置。見えていない場合は -1 になる）
        observed_relic_node_positions = np.array(obs["relic_nodes"])  # shape: (max_relic_nodes, 2)
        # 観測可能なレリックノードかどうかのマスク（Trueならそのレリックノードが見えている）
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])  # shape: (max_relic_nodes,)
        # 各チームの現在の得点情報。自分のチームは team_points[self.team_id]
        team_points = np.array(obs["team_points"])
        
        # 現在利用可能なユニットのID（unit_maskがTrueのユニットのインデックス）
        available_unit_ids = np.where(unit_mask)[0]
        # 観測可能なレリックノードのID（maskがTrueのレリックノードのインデックス）をセットにする
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        # アクション配列を初期化。形状は (max_units, 3) で、各ユニットのアクションを保持する
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # 基本戦略：
        # - 一部のユニットはランダムに探索を行い、他のユニットはエネルギー収集を行う
        # - もしレリックノードが見つかったら、全ユニットをそのノードの周辺に移動させ、得点を狙う
        # - 見つけたレリックノードの情報は次のマッチでも利用するために保存する

        # 新たに観測したレリックノード情報を保存する
        for id in visible_relic_node_ids:
            # まだ発見していないレリックノードの場合
            if id not in self.discovered_relic_nodes_ids:
                # 発見済みとして記録
                self.discovered_relic_nodes_ids.add(id)
                # 該当レリックノードの位置情報を保存
                self.relic_node_positions.append(observed_relic_node_positions[id])
            
        # 利用可能な各ユニットに対してアクションを決定する
        for unit_id in available_unit_ids:
            # ユニットの現在位置とエネルギーを取得
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            
            # もし、既に発見したレリックノードがある場合は、その最初のノードを目標とする
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                # マンハッタン距離を計算（x座標とy座標の差の絶対値の和）
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                
                # レリックノードの近く（マンハッタン距離4以内）なら、周辺をランダムに動いて得点を狙う
                if manhattan_distance <= 4:
                    # 0から4までのランダムな方向を選択（0は何もしない、1～4は各方向）
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # それ以外の場合は、レリックノードに向かって移動する
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            else:
                # まだレリックノードが見つかっていない場合は、ランダム探索を行う
                # 20ターンごと、またはまだ目的地が設定されていないユニットの場合に新しい目標地点を設定
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    # マップ全体の中からランダムな位置を選ぶ
                    rand_loc = (
                        np.random.randint(0, self.env_cfg["map_width"]),
                        np.random.randint(0, self.env_cfg["map_height"])
                    )
                    self.unit_explore_locations[unit_id] = rand_loc
                # 設定した探索先に向かって移動する
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        
        # 全ユニットに対するアクション配列を返す
        return actions
