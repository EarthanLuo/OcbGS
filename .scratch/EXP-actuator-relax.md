# 实验:让控制器真的能挪预算 → 测 yes/no

> 分支 `exp-actuator-relax`。一个问题:**控制器能挪预算时,A+B 比 A-only(≈不挪)更好吗?**
> 绿灯 → 想法有戏,继续帕累托/一体化。红灯 → 诚实转向(B 降级,主打可控+效率)。

---

## 改了什么(就一处,默认不变)

新增 CLI:`--grow_relax_scale`(默认 `1.0` = 当前行为,**不开就和现在一模一样**)。

它把控制器生长路径 `anchor_growing_capped` 里的候选生成阈值乘上这个系数。设 `<1`(如 `0.1`)→ 低梯度的格子也能生候选 → B 指出的"梯度漏掉的远景/细节格"**第一次能真的长出 anchor**。

- 改动文件:`ocbgs/arguments/__init__.py`(声明)、`ocbgs/scene/gaussian_model.py`(读入 + 在 capped 路径乘 threshold)。
- 总增长仍被 plan 的 `Σdelta ≤ B_total` 和每格 `min(delta, n_cand)` 限住 → **最终 anchor 数不会爆**;只是 gather 时候选张量变大(显存),4090 应该扛得住。

---

## 第 0 步:先确认"放松"真的解开了 73%(否则后面数据都不可信)

挑 amsterdam、seed 0,先只跑 arm_c(A+B)带 relax,看活跃 anchor 数有没有冲过 73%:

```bash
cd ~/OcbGS && git fetch && git checkout exp-actuator-relax
SRC=/root/autodl-tmp/bungeenerf/amsterdam
BT=$(cat /root/autodl-tmp/exp4/bungeenerf/amsterdam/BTOTAL_amsterdam)
python ocbgs/train.py -s $SRC -m /root/autodl-tmp/relax/amsterdam/c_relax01 \
  --fork 2 --base_layer 10 --visible_threshold 0.0 --dist2level round --update_ratio 0.2 \
  --progressive --levels -1 --dist_ratio 0.99 --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
  --iterations 30000 --update_until 25000 --test_iterations 25000 30000 --save_iterations 25000 30000 \
  --seed 0 --port 6131 --B_total $BT \
  --b_enabled --fusion_lambda 1.0 --b_camlist_size 16 --b_refresh_period 10 \
  --grow_relax_scale 0.1 2>&1 | tee /root/autodl-tmp/relax/c_relax01.log
```

跑到 update_until 看最终 anchor 数(grep 训练日志里的 anchor 数 / 或 results 里的计数):

- **冲过 ~85%+ → 放松起效了**,继续第 1 步。
- **还卡在 ~73% → relax 不够**,把 `--grow_relax_scale` 降到 `0.05` 或 `0.01` 重试第 0 步。
- **OOM → 调高到 `0.3`** 再试(候选少一点)。

---

## 第 1 步:决定性对比(2 个 arm,同场景同预算同 seed,都带 relax)

用第 0 步确认有效的那个 relax 值(下面假设 `0.1`):

```bash
# arm_b: A-only + relax
python ocbgs/train.py -s $SRC -m /root/autodl-tmp/relax/amsterdam/b_relax01 \
  --fork 2 --base_layer 10 --visible_threshold 0.0 --dist2level round --update_ratio 0.2 \
  --progressive --levels -1 --dist_ratio 0.99 --init_level -1 --extra_ratio 0.25 --extra_up 0.01 \
  --iterations 30000 --update_until 25000 --test_iterations 25000 30000 --save_iterations 25000 30000 \
  --seed 0 --port 6132 --B_total $BT --grow_relax_scale 0.1 \
  2>&1 | tee /root/autodl-tmp/relax/b_relax01.log

# arm_c: A+B + relax  (第 0 步那个就是,可直接复用)
```

(可选参照:再跑一个 `--grow_relax_scale 1.0` 的 arm_c,= 现在卡住的行为,用来证明"放松确实改变了结果"。)

---

## 比什么 / 怎么读

| 看 | arm_b (A-only) | arm_c (A+B) | 判读 |
|---|---|---|---|
| 最终活跃 anchor 数 | — | — | 两者应都冲过 73%(挪动起效) |
| 测试集 PSNR/SSIM/LPIPS | — | — | 第一信号 |
| **远景视角 PSNR** | — | — | **关键**:B 的主场 |

- **绿灯**:arm_c 的(尤其远景)PSNR **明显 > arm_b**,且两者分配可见地不同 → B 能落地、想法有戏。
- **红灯**:arm_c ≈ arm_b → B 即使能动也不帮忙 → 诚实转向。

> 远景 PSNR 需要把测试相机按视距分组。如果现在还没有这个分组脚本,**先看整体 PSNR + 活跃数差异**当粗信号;远景分组作为确认性的下一步(对应新 spec 的 Exp B)。

---

## 诚实声明(必读)

- 这段代码我**没法在本地 GPU 上验证**(纯 CUDA 训练路径)。逻辑我逐行核过、flag 默认不改变行为,但**真正的测试就是这次服务器跑**。先用第 0 步那个单跑确认不崩、不 OOM、且 73% 被解开,再投第 1 步的对比。
- relax 是**实验开关,不是最终方法**——它只为回答 yes/no。绿灯之后,把"挪预算"做干净是另一回事。
- 这是 amsterdam 单场景单 seed 的快速探针,不是定论;绿灯后再扩场景/seed。
