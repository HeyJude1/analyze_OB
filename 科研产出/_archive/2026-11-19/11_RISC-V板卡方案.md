# 11 RISC-V 板卡方案：租用 / 云主机 / 采购（2026-11 调研）

> **2026-11-12 复测更新（重要）**：`gdriscv.com` 在 Lead 环境与用户浏览器均**打不开**（DNS 可解析但 HTTPS 无响应）。**正确入口是 `https://cloud.spacemit.com/`**（Lead 实测 HTTP 200，页面标题就是 "Bianbu Cloud"），原来的 `gdriscv.com` 可能是旧域名/已失效。请用新域名注册并申请 K1 实例；如果新域名也打不开，按"Cloud-V → 买板 → 校内/导师找板"顺序执行，不要卡在注册上。Cloud-V（cloud-v.co）Lead 实测 HTTP 200 可访问。

> 版本 v1.0｜2026-11-12｜作者：experiment-designer
> 依据：`科研产出/08b_RISC-V真机实验协议.md`（真机怎么测，本文不重复）、`04_实验设计与评估协议.md`（性能只在原生 x86 出主结论，RISC-V 做跨架构正确性/真机加速比）
> 证据分层：**✅已核实**（本次联网抓取到页面原文，链接可点开复核）｜**◐待核实**（页面存在但内容 JS 渲染/被拦截，或价格为历史报价）｜**✗推断**（基于常识的估算，不进论文）
> 所有价格均为调研时点报价，**会变动**；写进论文前必须重新截图核对。

---

## 0. 一句话结论

**不用先买板：先进迭时空官方的 Bianbu Cloud 免费租 K1 实例（RVV 1.0 / VLEN=256 / 8 核 / root / SSH 隧道，申请可选 48 小时、7 天或 1 个月），同时免费申请 Cloud-V 的物理板（BPI-F3 / Milk-V Jupiter / Orange Pi RV2，SSH 无 sudo）；两者都不可持续时，再花 ¥300–800 买 Orange Pi RV2（$30–49.9）或 BPI-F3（$63–100），淘宝/京东 2–5 天到货。Scaleway EM-RV1 虽然最便宜（€0.042/小时）、SSH 最顺，但它是 T-Head C910，向量扩展是 0.7.1 级（XTheadVector），不是 RVV 1.0——买/租它做不了本课题的 K1 类 RVV 实验。** 现金支出可控制在 **¥0（先租）→ ¥300–800（兜底买板+配件）**。

---

## 1. 先定门槛：什么环境才算"能用"

| 类别 | 要求 | 理由 |
|---|---|---|
| **硬门槛（缺一不可）** | RVV **1.0**、**VLEN=256**、8 核（2 簇×4）、≥4GB RAM、可编译运行静态 ELF、能跑 OpenBLAS（或链接用户已有 `.a`） | 用户语料/工具链全部按 `-march=rv64gcv_zvl256b_zba_zbb_zbc_zbs` 构建；VLEN≠256 要重编 OpenBLAS 且实验结果不可比 |
| **硬门槛** | **root 或 sudo**（能改 `/sys/devices/system/cpu/*/cpufreq`） | 论文计时必须锁频并记录；无 root 只能做正确性，不能进性能主表 |
| **硬门槛** | SSH 或可持久访问 ≥1 周、数据不丢 | 45 算子×多条件矩阵需要反复编译/上传/重跑 |
| 加分项 | 裸金属（非容器/非共享 vCPU）、cpufreq 可用、8GB+、千兆网、串口/救援通道 | 决定计时 RSD 能否压到 3% 以内（`04` §6.7） |
| **红线** | QEMU/仿真时间**不得**写进性能表 | `08b` §5.1：QEMU 只做正确性前置 |

---

## 2. 云主机 / 远程实验室逐项核查

| # | 方案 | 2026-11 可用性 | 硬件与 RVV 规格 | 价格 | SSH / root | 租期 | 申请门槛 | 证据链接 | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **Bianbu Cloud（进迭时空 SpacemiT 官方云平台）** | ✅ 可用（gdriscv.com 在线） | **K1** 实例（8 核 X60，RV64GCVB，**RVV 1.0，VLEN=256**，与用户目标硬件同款）；镜像 Bianbu v2.0rc2 / Kylin / Deepin 等 | **✅ 免费试用**：官方公告"**8000 个 K1 实例免费申领**"（2026-02/03）；超出试用后的计费**◐待核实** | ✅ **SSH 可用**：申请实例→拿动态连接地址→装 **Stunnel**→ `ssh -p 2222 root@localhost`；另有 VNC `localhost:5901`；**root 密码 `bianbu`**（论坛回复）；Web 远程桌面也可 | ✅ 申请时自选 **48 小时 / 7 天 / 1 个月**；到期可否续**◐待核实** | 手机号或微信/QQ 注册，实名**◐待核实**；无需单位证明 | [平台](https://gdriscv.com/)｜[申请+实例配置指南（2026-02-13）](https://forum.spacemit.com/t/topic/972)｜[本地 SSH/VNC 连接指南（2025-10-27）](https://forum.spacemit.com/t/topic/803)｜[BPI 官方论坛"8000 个 K1 免费"公告（2026-03-06）](https://forum.banana-pi.org/t/openclaw-lands-on-risc-v-cloud-based-intelligent-assistant-deployed/26870) | **★ 首选（最快、0 元、K1+root）** |
| **2** | **Cloud-V（10xEngineers，RISC-V International 首个官方 Lab Partner）** | ✅ 可用（服务页/文档 2026 在线） | ✅ 物理板列表明确含 **Banana Pi F3**（8 核，1.6GHz，4/16GB，Vector 扩展串含 `zve64d/zfh/zba...`）、**Milk-V Jupiter**（8 核 1.8GHz，16GB）、**Orange Pi RV2**（8 核 1.6GHz，8GB）；另有 VisionFive 2（无 V）、Pioneer（`xtheadvector`=0.7.1）、QEMU | **✅ 免费**：SLA 规定 RISC-V International 会员**或"有公开开源仓库"**即享免费额度（20 核 或 10 块物理板、每仓库 1 条免费 CI）；付费 Standard/Premium 档价格未公开◐ | **SSH 有，但无 sudo**（官方文档：unprivileged，仅普通用户）；可自行从源码编译；sudo 权限"即将提供"◐ | 按资源排队；CI 长期可用，SSH sandbox"视需求与资源而定"◐ | 填 [contactus](https://cloud-v.co/contactus) 申请；**开源仓库即可，不需要买会员**（SLA 第 1 条） | [服务页](https://cloud-v.co/services)｜[SLA（2023-12 生效）](https://cloud-v.co/risc-v-labs-sla)｜[实例规格表](https://10x-engineers.github.io/riscv-ci-partners/compute_instances/)｜[SSH 权限说明](https://10x-engineers.github.io/riscv-ci-partners/sandboxing/) | **★ 备选（免费物理板，但无 root→只能做正确性）** |
| 3 | **RISC-V International「RISC-V Labs」目录** | ✅ 页面存在（`riscv.org/developers/labs/`）；具体可用实验室名单需浏览器查看◐ | 伞形计划，云实验室由伙伴提供（Cloud-V 为首个官方 Lab Partner） | 由各 Lab 定 | 由各 Lab 定 | — | 需是 RISC-V International 会员或经 Lab 审批 | [Labs 页](https://riscv.org/developers/labs/)｜[Cloud-V 成为首个官方 Lab Partner 海报](https://10xengineers.ai/wp-content/uploads/Poster-Cloud-V_-Becoming-the-first-official-risc-v-lab.pdf)｜[lab-partners 邮件列表存档](https://lists.riscv.org/g/lab-partners/messages) | 走 #1/#2 即可，无需单独申请 |
| 4 | **RISE RISC-V Runners** | ✅ 在线（ci.riseproject.dev） | ⚠️ 硬件是 **Scaleway EM-RV1**（C910，4 核 1.85GHz，**0.7.1 级向量**）；跑 GitHub Actions，`runs-on: ubuntu-24.04-riscv` | **✅ 开源项目免费** | ❌ 无 SSH：CI runner **ephemeral**（每个 job 全新环境），不能交互 | 单 job 级 | 安装 GitHub App，公开仓库 | [服务首页](https://ci.riseproject.dev/)｜[六周总结博客（2026-05-12）](https://riseproject.dev/2026/05/12/rise-risc-v-runners-six-weeks-in/) | ✗ 不能做计时实验；只能跑通用 rv64gc 构建 |
| 5 | **Scaleway Elastic Metal RV1** | ✅ 在售（fr-par-2） | ⚠️ **T-Head C910**，4 核 1.85GHz，16GB LPDDR4，128GB eMMC，100Mb/s；官方 FAQ 列的 ISA：`RV64IMAFDC_Zicsr_Zifencei_Zfh_XTheadc`+XIE/XMAE，**无 RVV 1.0**（XTheadVector 属 0.7.1 级） | ✅ **€0.042/小时；€15.99/月**（不含税） | ✅ 完整 root、SSH、串口控制台 | 按小时/按月 | 注册 Scaleway 账号 + 绑卡（境外） | [产品规格与价格页](https://labs.scaleway.com/en/em-rv1/)｜[FAQ（含 GhostWrite、ISA）同页](https://labs.scaleway.com/en/em-rv1/) | ✗ **便宜但 ISA 不对**，不能用于本课题 RVV 1.0 实验；仅可作"RVV 0.7.1 对照"（若论文需要） |
| 6 | **阿里云 / 腾讯云 / 华为云 RISC-V 实例** | ◐ **未检索到公开可租的 RVV 1.0 云实例**（2026-11）；阿里有平头哥玄铁生态与"云上实验室"（见 #7），腾讯/华为公开产品是 ARM（鲲鹏/倚天），未见 RISC-V ECS 售卖 | — | — | — | — | — | 检索无公开售卖页；[阿里云平头哥社区（2022 云实验室文章）](https://developer.aliyun.com/article/903432) | ✗ 目前不存在；不要在这上面花时间 |
| 7 | **玄铁云上实验室（T-Head xrvm.cn）** | ◐ 页面存在（`xrvm.cn/overview/cloudlab`），内容 JS 渲染无法抓取；是否仍开放申请待核实 | ◐ 历史提供 C906/C910 类板卡（RVV 0.7.1 级），**无 RVV 1.0 证据** | ◐ 待核实 | ◐ 待核实 | ◐ | 需注册玄铁社区 | [cloudlab](https://www.xrvm.cn/overview/cloudlab)｜[阿里云文章（2022）](https://developer.aliyun.com/article/903432) | ◐ 可能性低（ISA 不对），申请成本低可顺带试 |
| 8 | riscfive.cloud（"为全球开发者提供便捷的 RISC-V 云平台"） | ◐ 本次抓取失败（页面不可达/需 JS） | ◐ 待核实 | ◐ | ◐ | ◐ | ◐ | 站点：https://riscfive.cloud/ | ◐ 仅记录，不依赖 |
| 9 | **算能 SG2042 / Milk-V Pioneer 云租用** | ✗ 未检索到公开出租服务（只有整机售卖） | SG2042 64×C920，**RVV 0.71**，128GB，120W | 整机：全新约 $1,199–1,300 档；**二手成交价 $1,000**（2026-11-11，加拿大卖家） | 自购自管 | — | 直接购买 | [milkv.io/pioneer（规格）](https://milkv.io/pioneer)｜[二手成交记录 2026-10/11](https://bobsummerwill.com/2026/10/29/milk-v-pioneer-box-for-sale/) | ✗ 又贵又 ISA 不对 |
| 10 | SpacemiT MUSE Shelf（K1 集群服务器）/ K3 新平台 | ◐ 产品页在线，未见零售/租赁 | MUSE Shelf：K1 集群整机；K3：16 核（2026-06 评测），RVV 规格待核实 | ◐ | 自购/企业合作 | — | 企业渠道 | [MUSE Shelf](https://www.spacemit.com/community/development-kit/k1-muse-shelf)｜[K3 评测 2026-06](https://www.cnx-software.com/2026/06/21/spacemit-k3-pico-itx-risc-v-chassis-kit-review-part-2-what-works-what-doesnt-in-bianbu-os-4-0/) | ✗ 超出毕设预算；K3 只做 Future Work 提及 |
| 11 | QEMU（用户已有 `qemu-riscv64`） | ✅ | 用户工具链 `-cpu rv64,v=true,vlen=256` | 0 | 本机 | — | — | `08b` §2 | ✗ 只做开发/正确性，不做性能 |

> **关键结论**：#1（Bianbu Cloud）与 #2（Cloud-V）是仅有的两条"免费 + 真 K1/RVV1.0 + 可 SSH"路径，应**同时申请**；#5（Scaleway）是"最便宜的完整 root 云主机"，但 ISA 不对，不能当主方案。

---

## 3. 买板方案逐项核查

| 板卡 | SoC / RVV | 内存/存储 | 价格（时点） | 到货 | 配件/门槛 | 判定 |
|---|---|---|---|---|---|---|
| **Banana Pi BPI-F3** | SpacemiT **K1**，8×X60，**RVV 1.0，VLEN=256**，1.6GHz；2 TOPS NPU | 2/4GB LPDDR4；8/16GB eMMC；microSD；M.2(PCIe2.1×2)；双千兆；WiFi5 | **$63.16（2GB/8GB，缺货）～ $73.69（4GB/16GB，AliExpress，2024-05 报道）+运费；Amazon $89（4GB/16GB）**；2026-11 Amazon 在售 2+8GB 套装标价 **HKD 783.44≈$100**（动态价，◐） | 淘宝/京东 2–5 天；AliExpress 7–20 天（✗推断） | 需 **12V/3A 电源**（DC 口或 USB-C PD）、5V 风扇（◐约 ¥20）、microSD；**与用户现有交叉编译脚本完全一致** | **★ 买板首选（硬件与目标一致）** |
| **Orange Pi RV2** | **Ky X1 = SpacemiT K1 换标**（8×X60 RV64GCVB，1.6GHz，**RVV 1.0**） | 2/4/8GB LPDDR4X；eMMC 模块可选；双千兆；WiFi5；双 M.2 | **$30 起，最高 $49.90（2025-03 官方发布报道）** | 淘宝/京东 2–5 天 | 需 Type-C PD 电源/风扇（◐）；固件/RVV 支持需按 §6 验收 | **★ 最省钱买板（¥250–400 档）** |
| **Milk-V Jupiter** | SpacemiT K1/M1，8×X60，**RVA22 + RVV 1.0，256-bit**；**K1 1.6GHz / M1 1.8GHz**（注意频率不同！） | 4/8/16GB LPDDR4X；Mini-ITX；双千兆；WiFi6；M.2 NVMe；12V DC/ATX | ◐ **待核实**（官方淘宝 `milkv.taobao.com`、京东 `item.jd.com/10108225846745.html`、ARACE 国际；本次抓取被 429/JS 拦截） | 国内 2–5 天 | 12V DC 或 ATX；建议 16GB 版做 dgemm 大尺寸；**M1 是 1.8GHz，论文必须注明** | 备选（16GB + Mini-ITX 更稳，但贵且价待核实） |
| Milk-V Pioneer（SG2042） | 64×C920，**RVV 0.71**，2GHz，128GB | mATX/整机 120W | $1,000–1,300 | 现货/二手 | 整机 | ✗ 不推荐（ISA 不对+贵+功耗大） |
| Lichee Pi 4A（TH1520） | 4×C910，**RVV 0.7.1 级（XTheadVector）** | 8/16GB | **$191.99（8GB，Newegg）～ $248.19（16GB，Amazon CA）** | 5–15 天 | — | ✗ 不推荐（ISA 不对，且比 RV2 贵） |
| VisionFive 2（JH7110） | 4×U74，**无向量扩展**（ISA 串 `rv64imafdc_...` 无 `v`） | 2/4/8GB | $50–90 档（◐） | 现货 | — | ✗ 完全不能做 RVV |
| 二手/群友转让 | K1 类板卡 | — | 参考 Pioneer 二手案例：挂牌 $1,300→成交 $1,000（2026-11） | 视卖家 | 注意电源/固件/保修 | ◐ 可遇不可求；论坛/闲鱼蹲 |

**购买链接（官方/可靠）**：

- BPI-F3 官方资料：https://docs.banana-pi.org/en/BPI-F3/BananaPi_BPI-F3 ；官方 wiki：https://wiki.banana-pi.org/Banana_Pi_BPI-F3 ；比价参考：CNX 报道（2024-05）https://www.cnx-software.com/2024/05/10/banana-pi-bpi-f3-sbc-spacemit-k1-octa-core-risc-v-ai-soc/
- Orange Pi RV2 发布报道（$30–49.9）：https://www.cnx-software.com/2025/03/08/orange-pi-rv2-low-cost-risc-v-sbc-ky-x1-octa-core-soc-2-tops-ai-accelerator/
- Milk-V Jupiter 官方购买入口（淘宝/京东/ARACE）：https://milkv.io/jupiter ；规格（256-bit RVV1.0、K1/M1 频率差异）：https://milkv.io/docs/jupiter/overview
- Milk-V Pioneer 规格：https://milkv.io/pioneer

---

## 4. 三档推荐（直接照这个做）

### 档 A：最快能开始实验（0 元，今天申请、1–2 天可用）——**首选**

1. 打开 https://gdriscv.com/ ，手机号/微信注册；
2. 「申请实例」→ 镜像选 **Bianbu v2.0rc2**、区域「珠海-生态场」、设备 **K1**、时长选 **48 小时**（先短后长）；
3. 控制台 →「开始远程」→ 终端；或按[连接指南](https://forum.spacemit.com/t/topic/803) 配 Stunnel → `ssh -p 2222 root@localhost`；
4. 用 **§6 的 30 分钟验收清单**确认是真 K1（VLEN=256、cpufreq、8 核）；
5. 验收通过 → 立即跑 `08b` §6 的 R0/R2/R3（环境基线 + C0/C1 主扫描 + P1 参照），当晚出第一版真机加速比；
6. 后续再申请 **7 天 / 1 个月** 实例做 R4/R5（生成代码 + 多线程/LMUL 微矩阵）。

> 注意：这是"生态场"共享资源，可能是**容器/共享宿主**；验收清单第 8 项专门查噪声。若 RSD 压不到 3%，**只把它当正确性环境**，性能结论等买板（档 C）。

### 档 B：最可复现（0 元，1–3 周审批）——与档 A 并行申请

按 [Cloud-V contactus](https://cloud-v.co/contactus) 提交：说明"学术研究、有公开开源仓库、需要 **Banana Pi F3（8 核 K1，RVV 1.0，VLEN=256）** 的 SSH sandbox 做 OpenBLAS/算子 benchmark"。SLA 明确开源仓库可免费。**预期限制**：无 sudo → 不能锁频、不能改 governor；所以**只用于正确性/结构验证 + 交叉核对外部环境**，性能主表用档 A（有 root）或档 C。
优点：物理板、长期 CI 可挂、比共享云稳定；缺点：排队与审批不确定。

### 档 C：兜底买板（¥300–800，2–5 天到货）——如果 A/B 一个月内都不可持续

- **最省钱**：Orange Pi RV2（8GB 版，$30–49.9 ≈ **¥220–370**）+ Type-C PD 电源 + 风扇 + microSD ≈ **¥300–450**；
- **最稳（与语料完全一致）**：BPI-F3 4GB/16GB（$73.69–100 ≈ **¥530–720**）+ 12V/3A 电源 + 5V 风扇 + microSD ≈ **¥620–800**；
- 到货：淘宝/京东 2–5 天；AliExpress/Amazon 7–20 天（✗推断，下单前问客服）。
- 收货后同样先跑 §6 验收清单，再按 `08b` §2–§3 部署（rsync 130MB bin、锁 1.6GHz、跑 R2–R5，合计 3.5–4.5 h）。

### 档 D：明确不推荐

Scaleway RV1（ISA 不对）、Milk-V Pioneer（贵+0.71）、Lichee Pi 4A（0.7.1+贵）、VisionFive 2（无 V）、RISE Runners（CI-only + 0.7.1）、国内三大云（无公开 RVV 1.0 实例）、玄铁云上实验室（0.7.1 可能性大、细节待核实）。

### 预算与时间对照

| 方案 | 现金 | 到"能跑实验"的时间 | 能力边界 |
|---|---|---|---|
| A Bianbu Cloud | **¥0** | 1–2 天 | 可能是共享/容器环境；root 可锁频，但邻居噪声需实测 |
| B Cloud-V | **¥0** | 1–3 周（审批） | 物理板但无 sudo；正确性/结构验证主力 |
| C 买板（RV2/BPI-F3） | **¥300–800** | 2–5 天（国内） | 完整 root、可锁频、可复现，论文数字最可靠 |
| A+B+C 组合 | ¥0 → ¥300–800 | 2 天出第一版真机数字 | 推荐：**A 立刻跑、B 当对照、C 两个月内兜底** |

---

## 5. 排期（与 `04`/`08b` 对齐）

| 时间 | 动作 | 产出 |
|---|---|---|
| **今天（10 分钟）** | 注册 Bianbu Cloud 并申请 48h K1 实例；同时给 Cloud-V 发 contactus 邮件 | 两个申请号 |
| 第 1 天 | 按 §6 验收；跑 `08b` R0（环境基线+STREAM+roofline） | `board_report`、BW/Peak |
| 第 1–2 周 | 不等云资源：QEMU（已有）+ x86 主实验并行推进（`04` E1/E2） | C0/C1/P1 基线 + pilot |
| 第 2–3 周 | 云实例续到 7 天 → 跑 `08b` R2（45 算子 C0/C1 主扫描）+ R3（P1） | 第一版真机 S_comp 表 |
| 第 3–4 周 | 若 RSD>3% 或实例不稳定 → 下单 Orange Pi RV2 / BPI-F3 | 采购单 |
| 第 4–6 周 | 板到 → 锁频重测 R2–R5，替换云端不可靠数据 | 论文用真机表 |
| 第 6–10 周 | 跨架构实验（R4/R5）+ 与 x86 配对分析 | 跨架构主表 |

> **决策规则**：第 3 周末做一次判定——① Bianbu Cloud 验收通过且 RSD≤3% → 继续租，不买板；② 验收通过但噪声大/实例总到期 → 买 BPI-F3 或 RV2；③ 申请不下来 → 买板（$30–100 完全可承受），同时 Cloud-V 继续排队。

---

## 6. 拿到环境后 30 分钟验收清单（8 项，逐项打勾）

| # | 检查项 | 命令（复制即用） | 期望值 | 不符怎么办 |
|---|---|---|---|---|
| 1 | **是真 K1 吗** | `tr -d '\0' < /proc/device-tree/model; echo; nproc` | 含 `SpacemiT K1` / `spacemit,x60`；`nproc=8` | 不是 K1：停，换环境（别浪费时间） |
| 2 | **ISA 含向量** | `grep -m1 '^isa' /proc/cpuinfo` | 含 `v`、`zba`/`zbb`/`zbc`/`zbs` | 无 `v`：该环境只能做标量，判死 |
| 3 | **VLEN=256** | 交叉编译 `vlen_probe.c`（`08b` §1.3）上传运行；或板上 `gcc` 现编 | 打印 `VLEN=256 bits` | 128：需 `-march=rv64gcv_zvl128b` 并重编 OpenBLAS；`08b` §1.4 分支 B |
| 4 | **编译器/工具链** | `gcc --version`；`gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs -mabi=lp64d -c t.c -o t.o && echo OK` | `OK`（无 gcc 就全程交叉编译，上传静态 ELF） | 无 gcc：走 cross-compile（`08b` §2.1） |
| 5 | **OpenBLAS 能跑** | `ldconfig -p \| grep -i openblas`；或链接用户 `.a` 跑 dgemm：`OPENBLAS_NUM_THREADS=1 taskset -c 2 ./dgemm 256` | 有输出且数值正确；256³ 在真机上应 **5–60 ms** 量级 | 若 >500 ms：**疑似 QEMU/严重降频**，回到第 1、3 项复核 |
| 6 | **频率可观测/可锁** | `ls /sys/devices/system/cpu/cpu0/cpufreq/`；`cat .../scaling_cur_freq`；root 下 `echo performance \| tee .../scaling_governor` | 有 cpufreq 文件；`scaling_cur_freq` 稳定在 1.6GHz 档；能写入 governor | 无 cpufreq：不能锁频 → 只做正确性；记录实际频率并说明 |
| 7 | **绑核与扩展性** | `taskset -c 2 ./dgemm 256` vs `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 ./dgemm 512` | 1T/8T 有可解释差异（4 核簇内更好） | 绑核报错/未知核号：先 `lscpu -e` 确认拓扑 |
| 8 | **网络/持久化/噪声** | `rsync -az bin/ ./`（传 ~100MB）；`df -h`；跑同一 benchmark 20 次看 RSD；`cat /proc/1/cgroup`；`ls /.dockerenv` | 传输正常；文件留得住；**RSD<3%**；不是共享容器 | 容器/噪声大：只当正确性环境；性能等买板；所有数据当场 rsync 回本地 |

> 额外记录（写进 `env_report.json`）：`uname -a`、`cat /etc/os-release`、thermal zone 温度、是否 `M1`（`m1` 字样，1.8GHz）、云实例 ID/到期时间。

---

## 7. 云端特有的坑（与买板不同的地方）

1. **"K1 实例"可能是容器**：`nproc=8` 不代表独享 8 核；共享宿主会让计时抖动。**必须测 RSD**，必要时把 `cgroup` 和 `/.dockerenv` 状态写进论文环境描述。
2. **无 sudo（Cloud-V）**：不能 `cpupower frequency-set`、不能装包；计时只能"记录实际频率 + 多次中位数"，结论定位为"正确性/结构泛化"。
3. **实例到期/数据不持久**：48h/7d/1 月到期后环境清空。**所有代码/结果当天 rsync 回本地 + git 提交**；把"环境初始化脚本（`setup_board.sh`）"写好，换实例 10 分钟重建。
4. **SSH 要过 Stunnel 隧道**：直连 `ssh -p 2222 root@localhost` 依赖本地 stunnel 配置；Windows 用 WSL2 或 MobaXterm（按[官方指南](https://forum.spacemit.com/t/topic/803)）。
5. **默认密码是公开的**（`bianbu/bianbu`、root `bianbu`）：仅测试用途；**别放任何隐私数据/API key**，到期即弃。
6. **免费 8000 实例是活动**（2026-02/03 公告）：2026-11 是否仍在进行**◐待核实**；若申请页面无免费额度，就走档 B/C，不要充值赌。
7. **动态连接地址保密**：官方提示地址外泄会被攻击；不要贴到公开群/仓库。
8. **别用云实例的"远程桌面"做计时**：VNC/Web 桌面本身占 CPU，计时一律走 SSH 无头模式 + `taskset`。

---

## 8. 采购清单与参考总价

| 物品 | 型号/规格 | 参考价 | 备注 |
|---|---|---|---|
| 板卡（省钱档） | Orange Pi RV2 8GB | $30–49.9（≈¥220–370，2025-03 报道） | Ky X1=K1 换标，先按 §6 验收 |
| 板卡（稳妥档） | BPI-F3 4GB/16GB | $73.69–100（≈¥530–720，2024-05 报道/2026 Amazon 动态价） | 与现有语料/脚本完全一致 |
| 电源 | 12V/3A DC（BPI-F3）或 Type-C PD 45W（RV2） | ¥30–60 | 别用劣质电源，降频怪它 |
| 风扇 | 5V PWM 风扇 + 散热片 | ¥15–30 | K1 无风扇会降频（`08b` §5 风险"高"） |
| microSD | 32/64GB A1 | ¥25–40 | 或 eMMC/NVMe（更快） |
| 网线/串口 | 千兆网线；3.3V USB-TTL（可选） | ¥10–30 | 无网时串口救砖 |
| **合计** | — | **¥300–800** | 另有运费/关税（AliExpress 待核实） |

---

## 9. 待核实清单（写论文/下单前逐条确认）

1. ◐ **Bianbu Cloud 免费额度是否仍有效**（2026-02/03 公告"8000 个 K1 免费申领"）；超出试用后的计费方式。
2. ◐ Bianbu Cloud 实例是否为**独占 K1 板/虚拟机**，还是共享宿主容器（决定能不能出性能数字）。
3. ◐ Cloud-V SSH sandbox 的实际排队时间与是否给 root（当前 SLA 写 unprivileged，"privileged access will be added soon"）。
4. ◐ Milk-V Jupiter 当前官方售价（本次 ARACE 429、京东 JS、淘宝无法抓取）。
5. ◐ BPI-F3 淘宝/京东当前到手价与现货（本次只拿到 2024 报道价与 2026 Amazon 动态价）。
6. ◐ 玄铁云上实验室（xrvm）当前是否开放、板卡 ISA 版本。
7. ◐ riscfive.cloud 是否可用（本次抓取失败）。
8. ◐ RISC-V International 个人会员是否免费（SLA 允许"开源仓库"路径，可绕过此问）。
9. ◐ K3（16 核，2026）的 RVV 规格与售价（只作 Future Work，不阻塞）。

---

## 10. 参考链接（按证据强度排序）

**✅ 已抓取页面原文**

1. Bianbu Cloud 平台：https://gdriscv.com/ （标题"Bianbu Cloud"）
2. Bianbu Cloud 申请/实例指南（2026-02-13）：https://forum.spacemit.com/t/topic/972
3. Bianbu Cloud 本地 SSH/VNC 指南（2025-10-27）：https://forum.spacemit.com/t/topic/803
4. BPI 官方论坛"8000 个 K1 实例免费"（2026-03-06）：https://forum.banana-pi.org/t/openclaw-lands-on-risc-v-cloud-based-intelligent-assistant-deployed/26870
5. Cloud-V 服务：https://cloud-v.co/services
6. Cloud-V SLA（2023-12-01 生效）：https://cloud-v.co/risc-v-labs-sla
7. Cloud-V 实例规格（含 BPI-F3/Jupiter/RV2 的 ISA 串）：https://10x-engineers.github.io/riscv-ci-partners/compute_instances/
8. Cloud-V SSH 权限说明（无 sudo）：https://10x-engineers.github.io/riscv-ci-partners/sandboxing/
9. Scaleway EM-RV1 规格/价格/FAQ：https://labs.scaleway.com/en/em-rv1/
10. RISE Runners（免费、Scaleway RV1、ephemeral）：https://ci.riseproject.dev/
11. Milk-V Jupiter 规格与购买入口：https://milkv.io/jupiter ；文档：https://milkv.io/docs/jupiter/overview
12. Milk-V Pioneer 规格（RVV 0.71）：https://milkv.io/pioneer
13. 二手 Pioneer 成交记录（2026-10-29 挂牌、2026-11-11 成交 $1,000）：https://bobsummerwill.com/2026/10/29/milk-v-pioneer-box-for-sale/
14. BPI-F3 比价（2024-05：$63.16/$73.69/$89）：https://www.cnx-software.com/2024/05/10/banana-pi-bpi-f3-sbc-spacemit-k1-octa-core-risc-v-ai-soc/
15. Orange Pi RV2 发布与价格（2025-03：$30–49.9）：https://www.cnx-software.com/2025/03/08/orange-pi-rv2-low-cost-risc-v-sbc-ky-x1-octa-core-soc-2-tops-ai-accelerator/
16. Lichee Pi 4A 价格：https://www.newegg.com/p/3C6-0162-001R4 ；https://www.amazon.ca/LicheePi-Linux-RISC-V-16GB-Development/dp/B0CCRMC6V6
17. RISC-V International Labs 页：https://riscv.org/developers/labs/
18. Cloud-V 首个官方 Lab Partner 海报：https://10xengineers.ai/wp-content/uploads/Poster-Cloud-V_-Becoming-the-first-official-risc-v-lab.pdf
19. SpacemiT K3（2026-06 评测，Future Work）：https://www.cnx-software.com/2026/06/21/spacemit-k3-pico-itx-risc-v-chassis-kit-review-part-2-what-works-what-doesnt-in-bianbu-os-4-0/

**◐ 存在但未能抓取内容（JS/拦截）**

20. 进迭云平台使用指南（官方文档，JS）：https://www.spacemit.com/community/document/info?lang=zh&nodepath=cloud/userguide/cloud_user_guide.md
21. 玄铁云上实验室（JS）：https://www.xrvm.cn/overview/cloudlab
22. riscfive.cloud（抓取失败）：https://riscfive.cloud/
23. BPI-F3 官方文档：https://docs.banana-pi.org/en/BPI-F3/BananaPi_BPI-F3
24. spacemiT MUSE Shelf：https://www.spacemit.com/community/development-kit/k1-muse-shelf
25. 阿里云平头哥云上实验室文章（2022）：https://developer.aliyun.com/article/903432

---

## 附录：与 `08b` 的分工

- **本文（11）**：解决"机器从哪来"——租/云/买、价格、门槛、验收、坑。
- **`08b`**：解决"机器到手后怎么测"——交叉编译口径（C0/C1/P1）、板上一键探测脚本、计时骨架、实验矩阵 R0–R5、能说/不能说红线。
- **`04`**：解决"数字怎么进论文"——指标公式、统计、消融与主结论；**RISC-V 真机数据用于跨架构验证；原生 x86 仍是性能主结论**（若最终只有共享云 K1、无法锁频，则 RISC-V 只报正确性 + 结构指标）。
