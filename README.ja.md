# ComfyUI-CausalForcing（日本語の概要）

[Causal Forcing](https://arxiv.org/abs/2602.02214) / [Causal Forcing++](https://arxiv.org/abs/2605.15141)
（[thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing)）を ComfyUI から使うための
非公式統合です。原著者・清華大学・Shengshu・Alibaba (Wan)・Comfy Org とは無関係です。

- **公式重み・公式推論経路**：`zhuhz22/Causal-Forcing` の公式 checkpoint
  （Causal Forcing++ frame-wise 2-step、比較用の Causal Forcing frame-wise 4-step）を、
  commit 固定の upstream コードを無改変で import して実行します。同じ prompt・seed では
  公式 `inference.py` と初期ノイズ・latent・8bit フレームがすべて一致することを確認しました。
- **2-step の forward 数**：通常フレームは 2 step、最初のフレームだけ公式設定どおり 4 step、
  さらにフレームごとの KV 更新 1 回で、1 本あたり 65 回（4-step は 105 回）。実際の呼び出し回数を
  数え、設定と一致しなければジョブを失敗させます。
- **実測**（RTX 5090 / WSL2、832×480・81 フレーム、warm 6 本の平均）：generator は 2-step
  7.05 秒・4-step 10.97 秒、テキストエンコード＋generator＋VAE は 12.31 秒・16.29 秒。
  16 FPS は再生フレームレートで、生成速度の主張ではありません。論文の数値とは比較できません。
- **画質**：6 組を目視した範囲では両モデルとも破綻なし。4-step はカメラ・被写体の動きが大きく、
  2-step は構図が安定しがちです。画質の優劣や論文同等を主張するものではありません。
- **I2V**：frame-wise モデルで、公式と同じ前処理（832×480 にリサイズ・正規化・VAE エンコード）
  の image-to-video に対応（2-step で実機確認）。
- **未検証**：1-step と chunk-wise 4-step は実行対象として受け付けますが、今回は実機で
  検証していません。
- **安全設計**：workflow から実行ファイルやコマンドは指定できません（runtime_id で管理者設定を
  選ぶだけ）。`shell=True` なし、環境変数は allowlist、キャンセル・タイムアウト時は自分のジョブの
  プロセスグループだけを停止します。サンドボックスではありません。
- ComfyUI-MonarchRT（同じ作者）とは別の手法・別のパッケージです。

セットアップ：[docs/SETUP.md](docs/SETUP.md)（英語）、詳細な数値：[docs/BENCHMARKS.md](docs/BENCHMARKS.md)
