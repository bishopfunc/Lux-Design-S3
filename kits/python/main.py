import json
from typing import Dict
import sys
from argparse import Namespace
import os
import numpy as np

from agent import Agent
# from lux.config import EnvConfig
from lux.kit import from_json

### DO NOT REMOVE THE FOLLOWING CODE ###
# Kaggle環境ではコードが直接インポートされるため、複数のエージェントを格納する辞書を用意する
agent_dict = dict()  # 各プレイヤーごとのAgentインスタンスを保持する辞書
agent_prev_obs = dict()  # 前回の観測情報を保持するための辞書（必要に応じて利用）

def agent_fn(observation, configurations):
    """
    Kaggle提出用のエージェント定義関数
    :param observation: 現在の観測情報
    :param configurations: ゲーム設定情報（例：env_cfg）
    :return: エージェントのアクション（JSON形式で返す）
    """
    global agent_dict  # グローバルのagent_dictを使用
    obs = observation.obs
    # 観測情報が文字列の場合、JSON形式としてパースする
    if type(obs) == str:
        obs = json.loads(obs)
    
    # 現在のステップ、プレイヤーID、残りのオーバータイムを取得
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime

    # 最初のステップなら、新たにAgentインスタンスを作成し、辞書に登録する
    if step == 0:
        agent_dict[player] = Agent(player, configurations["env_cfg"])
    
    # 設定情報に"__raw_path__"がある場合、そのパスからディレクトリ名を取得する
    if "__raw_path__" in configurations:
        dirname = os.path.dirname(configurations["__raw_path__"])
    else:
        # なければ、現在のファイルのディレクトリ名を取得する
        dirname = os.path.dirname(__file__)

    # ディレクトリパスをシステムパスに追加して、必要なモジュールが見つかるようにする
    sys.path.append(os.path.abspath(dirname))

    # 対応するプレイヤーのAgentインスタンスを取得
    agent = agent_dict[player]
    # Agentのactメソッドを呼び出し、行動を決定する
    # from_jsonを用いて、観測情報を適切な形式に変換する
    actions = agent.act(step, from_json(obs), remainingOverageTime)
    
    # actionsはnumpy配列なので、リストに変換して辞書形式で返す
    return dict(action=actions.tolist())

if __name__ == "__main__":
    
    def read_input():
        """
        標準入力から入力を読み込む関数
        """
        try:
            return input()
        except EOFError as eof:
            # 入力がなくなった場合はプログラムを終了する
            raise SystemExit(eof)
    
    # 初期化用変数
    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    
    # 無限ループで入力を待ち続ける
    while True:
        # 標準入力から1行読み込み
        inputs = read_input()
        # 読み込んだ文字列をJSONとしてパースする
        raw_input = json.loads(inputs)
        
        # 必要な情報（ステップ、観測情報、残りオーバータイム、プレイヤーID、info）をNamespaceオブジェクトに変換
        observation = Namespace(**dict(
            step=raw_input["step"],
            obs=raw_input["obs"],
            remainingOverageTime=raw_input["remainingOverageTime"],
            player=raw_input["player"],
            info=raw_input["info"]
        ))
        
        # 最初の入力のとき、環境設定（env_cfg）とプレイヤーIDを保持する
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        
        # agent_fnを呼び出し、現在の観測情報に基づいたアクションを取得する
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        
        # 得られたアクションをJSON形式で標準出力に送信する
        print(json.dumps(actions))
