# ark-nova-ai

<p align="center">
  <img src="./Ark%20Nova%20Arena.png" alt="Ark Nova Arena" width="900" />
</p>

`ark-nova-ai` 是一个围绕 Ark Nova 原型开发的代码仓库，当前把本地对战、规则引擎拆分、数据集维护和自博弈强化学习实验放在同一个项目里。

当前仓库已经可以：

- 运行本地 2 人命令行对局
- 使用内置卡牌与地图数据进行规则开发和回归测试
- 训练、评估并加载 RL checkpoint
- 用脚本维护卡牌、地图和效果覆盖率数据

这个项目仍然是持续开发中的原型，不应被视为完整、最终版的 Ark Nova 规则实现。

## 环境要求

- Python 3.13
- `venv`
- 如果要做 RL 训练、checkpoint 对战或评估，需要额外安装 `torch`

当前 `requirements.txt` 只包含：

- `pypdf`
- `pytest`
- `numpy`
- `Pillow`

## 安装

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install torch
```

如果你不想激活虚拟环境，下面所有命令都可以继续使用 `.venv/bin/python` 和 `.venv/bin/pytest` 的写法。

## 快速开始

### 本地 2 人对局

主入口是 [`main.py`](./main.py)。默认是本地双人命令行对战。

```bash
.venv/bin/python main.py --seed 7
```

常用参数：

- `--quiet`：减少逐回合日志输出

示例：

```bash
.venv/bin/python main.py --seed 7 --quiet
```

## 强化学习工作流

RL 入口在 [`tools/rl/train_self_play.py`](./tools/rl/train_self_play.py)。训练前请先安装 `torch`。

### 自博弈训练

```bash
.venv/bin/python tools/rl/train_self_play.py \
  --updates 100 \
  --episodes-per-update 8 \
  --output-dir runs/self_play
```

常用参数：

- `--device auto|cpu|cuda[:index]`
- `--rollout-workers`
- `--checkpoint-interval`
- `--hidden-size`
- `--lstm-size`
- `--action-hidden-size`
- `--step-reward-scale`
- `--terminal-reward-scale`
- `--endgame-trigger-reward`
- `--endgame-speed-bonus`
- `--terminal-win-bonus`
- `--terminal-loss-penalty`
- `--resume-from`

checkpoint 会写入你指定的 `--output-dir`，默认文件名形如 `checkpoint_0005.pt`。

### 和训练好的 checkpoint 对战

命令行方式：

```bash
.venv/bin/python tools/rl/play_vs_checkpoint.py \
  --checkpoint runs/self_play/checkpoint_0010.pt \
  --human-seat 1
```

浏览器方式：

```bash
.venv/bin/python tools/rl/web_play_vs_checkpoint.py \
  --checkpoint runs/self_play/checkpoint_0010.pt \
  --host 127.0.0.1 \
  --port 8765
```

启动后访问 `http://127.0.0.1:8765`。

### checkpoint 对 checkpoint 评估

```bash
.venv/bin/python tools/rl/eval_self_play.py \
  --checkpoint-a runs/self_play/checkpoint_0010.pt \
  --checkpoint-b runs/self_play/checkpoint_0020.pt \
  --episodes 16
```

## 仓库结构

- [`main.py`](./main.py)：本地对局主入口，包含 CLI、合法行动生成和行动应用
- [`arknova_engine/`](./arknova_engine)：拆分中的规则引擎模块，包括开局、地图、计分、卡牌效果和共用规则
- [`arknova_rl/`](./arknova_rl)：RL 编码器、模型、训练与评估逻辑
- [`tools/`](./tools)：数据抓取、地图工具、图像工具和覆盖率报告脚本
- [`tools/rl/`](./tools/rl)：RL 训练、评估、人机对战和浏览器对战入口
- [`data/`](./data)：卡牌、地图、叠图和相关静态数据
- [`tests/`](./tests)：pytest 测试集
- [`docs/`](./docs)：规则书 PDF、提取文本和参考资料

## 数据文件说明

主要卡牌数据文件：

- [`data/cards/cards.base.json`](./data/cards/cards.base.json)：基础游戏卡牌数据
- [`data/cards/cards.marine_world.json`](./data/cards/cards.marine_world.json)：Marine World 相关数据
- [`data/cards/cards.promo.json`](./data/cards/cards.promo.json)：Promo 卡牌数据
- [`data/cards/card_effect_coverage.tsv`](./data/cards/card_effect_coverage.tsv)：卡牌效果覆盖率报告
- [`ark_nova_card_effect_audit.md`](./ark_nova_card_effect_audit.md)：卡牌效果审计笔记
- [`ark_nova_card_effect_audit.tsv`](./ark_nova_card_effect_audit.tsv)：审计结果表格

地图相关数据文件：

- [`data/maps/maps.json`](./data/maps/maps.json)：地图元数据
- [`data/maps/images/`](./data/maps/images)：地图底图
- [`data/maps/tiles/`](./data/maps/tiles)：地图六边形叠图与标注

如果你要编辑卡牌数据，需要注意这个仓库目前同时存在“数据驱动”和“运行时硬编码”两部分逻辑：

- 像 `appeal`、`conservation`、`reputation`、`required_icons`、`max_appeal` 这类字段会按通用逻辑生效
- `effects` 字段只是描述信息的一部分，一些卡牌仍然依赖 [`main.py`](./main.py) 或 [`arknova_engine/card_effects.py`](./arknova_engine/card_effects.py) 里的专门分支
- 某些赞助商徽章或特殊结算也可能来自运行时覆盖，而不完全来自 JSON

## 维护脚本

仓库里已经带了不少维护脚本，常用的有：

- [`tools/fetch_cards.py`](./tools/fetch_cards.py)：抓取或更新卡牌数据
- [`tools/fetch_maps.py`](./tools/fetch_maps.py)：抓取或更新地图数据
- [`tools/fetch_card_images.py`](./tools/fetch_card_images.py)：下载并切分卡图，供浏览器 UI 使用
- [`tools/validate_map_tiles.py`](./tools/validate_map_tiles.py)：校验地图 tile 数据
- [`tools/render_map_overlay_svg.py`](./tools/render_map_overlay_svg.py)：生成地图叠图
- [`tools/report_card_effect_coverage.py`](./tools/report_card_effect_coverage.py)：生成效果覆盖率报告

其中浏览器 UI 会优先读取 `data/cards/images/cards/manifest.json` 和对应卡图；如果你还没跑过 [`tools/fetch_card_images.py`](./tools/fetch_card_images.py)，界面仍然可以启动，但不会显示这些额外卡图资源。

## 当前状态与边界

- 当前最稳定的能力是本地双人命令行对局、规则回归测试和数据维护
- RL 管线已经可用，但属于基线实现，仍在持续调整
- `--marine-world` 目前只扩展终局计分卡池，不等于完整扩展规则支持
- 仓库内含规则书和提取文本用于实现与校验，但代码与数据仍可能存在未覆盖的规则分支
