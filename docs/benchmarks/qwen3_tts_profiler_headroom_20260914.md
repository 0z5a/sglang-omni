<!-- 采集与分析的来源说明(luojiaxuan,2026-09-14 PT) -->
> **来源**:eval-h100 GPU 0,容器 `sglang-omni-jaxan-1`,树 origin/main `1f6b6843`,CustomVoice Ryan,seed-tts-eval,开环泊松 rps 1 / rps 20。
> torch profiler 经 `/start_profile` `/stop_profile`(with_stack=1,record_shapes=1),窗口 r1 10 s、r20 5 s;graph 全关的 mapping 臂 2 s 只用于 kernel 到 Python 的映射。
> nsys:`nsys launch --trace=cuda,nvtx,osrt --cuda-graph-trace=node` 起服务,`nsys start --gpu-metrics-devices=0 --gpu-metrics-frequency=20000 --sample=none --cpuctxsw=none`(这两个开关只能给 `start`,给 `launch` 会直接报错退出)。
> 原始产物在宿主 `/data/jaxan/runs/20260910-qtts-prefill-graph/prof/`,分析脚本与表格在同根 `analysis/<lens>/` 与 `analysis/skeptic-<lens>/`(root 属主,经 docker exec 读)。
> 分析流程:六个视角(talker-decode、code-predictor、vocoder、nsys-sm、critical-path、code-catalog)独立读同一套产物,各出至多 4 个候选;每个视角一个 skeptic 用 formal trace 逐条复算收益并查重(main、在途 PR、本文档第 9 到 17 轮);最后合成。所有收益都是算术推演,无一实测。
> 两个采集教训:(1) 三个 stage 同在一个进程,每轮只有一个 trace 文件,文件名里的 stage 只是先停 profiler 的那个;(2) `/stop_profile` 导出会把 stage 进程卡 14-26 s,那几轮 client 汇总里的尾部不能用,profile 窗口本身在导出之前是干净的。

# Qwen3-TTS 主线余量综合报告(六镜头剖面综合,2026-09-14 PT)

**背景**:这是 #1754 主线在 origin/main 1f6b6843 上的一次"还有没有余量、余量在哪"的剖面综合。证据是 eval-h100(单张 H100 80GB,GPU 0)上采的四组产物:torch profiler(formal-r1 rps 1 窗口 10 s、formal-r20 rps 20 窗口 5 s,均 with_stack=1)、CUDA graph 全关的 mapping2-r1(只用于 kernel 到 Python 的映射)、nsys 生产配置采样(nsys-formal-r1 10 s、nsys-formal-r20 6 s,GPU metrics 每 50 us 一采,graph node 级追踪)、以及请求事件记录器 JSONL。六个镜头(talker-decode、code-predictor、vocoder、nsys-sm、critical-path、code-catalog)各出最多 4 个候选,再由六个 skeptic 逐条复算并查重(在途 PR、study log 第 9 到 17 轮),结果存活 14 条(合并同类后 9 条)、排除 8 条。所有脚本与表格在容器路径 `/data/runs/20260910-qtts-prefill-graph/analysis/<lens>/` 与 `analysis/skeptic-<lens>/`(宿主 `/data/jaxan/runs/20260910-qtts-prefill-graph/analysis/`,root 属主,经 docker exec 读)。

读法约定:下文"剖面"指 torch profiler 或 nsys 下的数字,CPU 段被膨胀(with_stack 约 1.3-3 倍,nsys node 级追踪每步约 +0.5-0.8 ms);"生产"指 study log 第 9/10/11/15 轮未剖面的记录器与探针数字。GPU kernel 时长两边一致,不受影响。所有候选收益都是 skeptic 修正后的算术推演,没有一条在 GPU 上跑过。

## 1. 结论:余量判决

先答用户的问题:torch profiler 看不出东西之后,nsys 的 SM Issue / SM Active 说 GPU 是空转还是发射差?答案是两种都有,而且都不是"没有余量"。

**r1**(nsys-formal-r1,10.21 s,204,231 个 50 us 采样;`analysis/nsys-sm/summary.md`):
- 全窗 kernel 忙 20.6%(GR Active 24.8%)。空闲的 80%(8.10 s 中的 6.50 s)是 4 个 >100 ms 的请求间隙,这是 rps 1 的到达间隔本身,不算余量。
- 忙时条件下:SMs Active 29.8%,SM Issue 5.4%(按活跃 SM 折算 18.1%),Tensor Active 1.2%,DRAM read 25.9%,每 SM 周期 7.5 个 warp 在飞。73.4% 的忙时 SM Issue 低于 5%,92.9% 低于 10%;只有 1.75% 的忙时达到 40% 以上。
- 原因是 kernel 粒度:597,586 个 kernel,中位数 1.95 us,94.9% 短于 10 us(占 kernel 时间 76.6%);96.6% 来自 CUDA graph(1,182 次 cudaGraphLaunch,每次 488 个 kernel);91.4% 的 kernel 时间落在 grid 小于 132 个 block 的 kernel 里,填不满 132 个 SM。
- 按图类分(overlap 加权):talker decode 图 SMs Active 42.0% / SM Issue 4.8% / DRAM read 40.8%(唯一接近带宽受限的阶段);code predictor 图 23.1% / 5.0% / 20.1%;vocoder WARM 图 48.6% / 16.5% / 14.4%;COLD 图 41.2% / 12.4% / 13.2%。
- 判决:请求内部 GPU 是"忙但发射差"(latency/launch-bound),不是算力或带宽饱和。余量三种:(a) kernel 效率(更少更胖的 kernel、全 SM 的 GEMV),主要在 predictor;(b) 每步 1.86 ms 的 CPU 空洞(8.03 ms 步的 23%),在途 PR #2123 已覆盖;(c) 调度(准入要白等一整个 decode 步)。

**r20**(nsys-formal-r20,6.15 s):
- 全窗 kernel 忙 61.6%(GR Active 68.1%)。38.4% 的空闲里没有一个 >10 ms 的间隙:3,349 个 100 us 到 1 ms 的洞共 978 ms(占空闲 41%),320 个 1 到 10 ms 的洞共 593 ms(25%),其余 776 ms 是 <100 us 的气泡。
- 忙时条件下:SMs Active 53.0%,SM Issue 16.5%(按活跃 SM 31.1%),Tensor Active 3.3%,DRAM read 18.2%;62.2% 的忙时 SM Issue 低于 10%;只有 28.1% 的忙时有两个以上 kernel 并发,即 vocoder 流与 talker 流只是部分重叠。
- talker 流每步 12.91 ms(p50)里空 4.55 ms 等 CPU(predictor GPU 结束到下一个 talker cudaGraphLaunch 3.88 ms + 发射到首 kernel 0.67 ms);401 步 x 4.55 ms = 1.83 s,占 6.15 s 窗口的 30%。
- 判决:调度/CPU 空洞与低发射 kernel 并存。前者大头由 #2123 覆盖(见时间账),剩下的是准入等待;后者在 predictor(982 kernel/帧)与 vocoder 的 eager-in-graph 形状(60% 的重放,约 1150 kernel/次)。
- 补一条:GPC 时钟稳定在 1,982 MHz 均值(p1 1,878;5.5% 采样低于 1,900),study log 第 7 轮"cohort 间降频、重放前爬频"的猜想不成立。

一句话:两个速率下 GPU 都不是"又忙又发射满"。Tensor Active 任何阶段不超过 4.2%,DRAM read 最高 40.8%(bs=1 talker 图)。余量存在,但要靠减少 kernel 数与调整发射/调度,不能靠"让 GPU 更满"。

## 2. 时间账

### 2.1 decode 步

**r1,bs=1**(nsys 生产配置,369 步;`analysis/nsys-sm/graphs-r1.json`):
步 8.03 ms p50(p95 9.67)= talker decode 图 2.03 ms(kernel 和 1.82,367-395 kernel,10% 气泡)+ 间隙 0.12 + predictor 图 3.80 ms(kernel 和 3.28,982 kernel,0.52 ms 节点间隙即 15%)+ 到下一 talker 图 1.86 ms(CPU 1.30 + 发射到首 kernel 0.54)。两图之间 45 个 eager kernel 共 0.10 ms。第 11 轮未剖面探针是 7.2 ms/步。

predictor 内部(bs=1,kernel 和 3.36 ms,`analysis/code-predictor/out/formal-r1_predictor_steps.md`):GEMM 432 个 2.10 ms(62.5%,每个 4.9 us,16-96 CTA;权重 2.65 GB/帧按 3.35 TB/s 只需 0.79 ms,即 38% 峰值);`_seeded_top_k_top_p_sample_kernel` 15 个 x 21.7 us = 0.33 ms;rmsnorm 176 个 0.28 ms;attention 80 个 0.26 ms(cudnn sdpa 3.1 us x75,flash_fwd 5.5 us x5);rope+kv-store、qknorm、act_and_mul、splitK reduce 各 80 个,0.11-0.13 ms;526 个 kernel 短于 2 us。

talker decode 图内部(bs=1):85 个 nvjet 64x8 GEMM 约 1.30 ms(GEMM 口径带宽 2.17 TB/s = 65% 峰值),FA3 decode 56 个 0.34 ms,glue 每层 5-6 个共 0.3 ms。

#2123 之后的推演(`analysis/skeptic-critical-path/tail_vs_predictor_r1.txt`,346 步):现在暴露的 CPU 2.04 ms 均值 / 1.89 p50;若尾部改在 predictor 启动时就开始(#2123 的做法),暴露降到 0.03 ms 均值,346 步里只有 1 步大于 0。

**r20,bs 约 24-48**(401 步):
步 12.91 ms p50(p95 20.8)= talker 图 2.32 + 间隙 0.54 + predictor 图 5.03(kernel 和 4.57,节点间隙 0.85)+ 到下一 talker 4.55(p95 10.7)。skeptic 重算 233 步:步 13.7 p50,尾 4.42 p50 / 11.15 p95,暴露 CPU 5.45 p50;#2123 之后暴露 p50 0.0、均值 0.98、p95 5.94(233 步里 103 步仍大于 0);在此之上再做完整异步循环只再省 0.6 ms 均值,p50 不变。第 11 轮生产步长约 12-14 ms。

predictor bs=48 只比 bs=1 多 37%(每个 kernel 都是延迟受限,sdpa 8.5 us、GEMM 6-8 us、采样 26.4 us);talker 图 2.71 ms 忙。

vocoder 在 r20:两个 follow-up 流 union 忙 16.6% + 16.0%,COLD 流 5.4%;385 个重放里 233 个(60%)是 eager-in-graph(1145-1184 kernel,3.1-4.4 ms),编译的 T=8 B=1 是 432 kernel、2.15 ms;B4/B8 的编译图里有 4/2 个 conv 走 `implicit_convolve_sgemm`(每次 440-1080 us),454 次共 302 ms = 全窗 kernel 时间 6.1%、vocoder 流 12.9%。

### 2.2 首帧路径

**r1**(剖面 formal-r1,12 个请求,均值 ms;`analysis/critical-path/timeline_formal-r1.md`;剖面下服务端 admission 到首 pcm 55.1 ms,对生产 32-36 ms):
准入到 preprocessing 0.44 | preprocessing 4.86(GPU 0.16,其余是 9 次 pageable H2D、tokenizer 0.26 等)| hop 0.11 | 引擎收件到 build 开始 2.87(空闲到达 0.45,忙时到达 4.10)| build 0.76 | build 完成到进等待队列 5.87(空闲 0.27,忙时 8.69,即白等一整个 decode 步)| 队列到 prefill 1.68 | prefill 15.58(GPU 5.5:28 个层图 1.66 + eager attention 0.56 + predictor 3.3)| 到首 codes 发出 0.52 | 第二个 talker 帧 10.21(GPU 5.3)| vocoder 第二帧进到首 pcm 11.63(GPU 3.8:COLD T=2 重放 1160 kernel,span 4.35;其中固定 initial_batch_wait 2.19,`_run_initial_batch` 8.76 = 等事件 3.7 + 51 次 zero_ 1.9 + COLD 发射 1.26 + 计划/提交 1.9)| 进程内 hop 0.41 | 协调器到 HTTP 到客户端 1.45。
整条路 GPU 忙 26.5 / 55.1 ms(48%);其余 52% 是 CPU 发射、调度循环等待与线程交接。IPC 与序列化不在账上(各 hop 0.06-0.37 ms)。

生产口径(第 11 轮账,p50 35.6):preprocessing 2.1 + 准入 3.3 + prefill 12 + 首帧 codes 到 vocoder 约 5 + 第二帧等待约 7 + hop/HTTP;#1998 full 图把 prefill forward 6.2 -> 0.79,首帧 35.6 -> 32.1(第 15 轮三 seed 复测)。predictor 在 r1 首帧路径上出现两次(prefill 收尾 3.3-3.8 + 第二帧 3.8),共约 7.6 ms,占可闻 TTFA 约 44 ms 的 17%。

**r20**(生产口径来自第 9/10/11/15 轮;剖面 formal-r20 是过载运行,在飞 33-51、队列等待 p95 413 ms,只取结构与比例):
收件到 build 0.62 个循环周期 + build 完成到进队列 0.93 个周期(生产 13.8-13.9 ms,剖面 23.0)+ prefill 12(剖面 18.9)+ 第二帧一步 12-14 + vocoder COLD T=1 span 4.42 + initial wait 2。predictor 出现三次(prefill 后 bucket-1 步 3.84、准入时在飞的批步、第二帧 5.3),9-14 ms。生产首帧 p50:breakable 52.5-53.0 / full 49.6-50.0,p95 69-72 / 64-67,可闻 55.5-58.6 / 52.2(第 15 轮三 seed)。

### 2.3 在途 PR 已占的账(不是新候选,叠加时先扣掉)
- #1998(full prefill 图):r1 首帧 -3.5,r20 p50 -3 / p95 -5 / 可闻 -4,已实测。
- #2123(token id 在 predictor 前 staging):每步 CPU 尾巴藏到 predictor 重放下,PR 自测 1 行步 8.04 -> 6.25 ms、16 行 8.57 -> 6.78;按 nsys 推演 r1 每步 -1.9,r20 每步 -5.1 均值、p50 步长 13.7 -> 约 8.3。它在 r20 首帧上的收益 PR 没测,按约 2.5 个周期比例段(收件 0.62 + 准入 0.93 + 第二帧 1.0)推算约 -10 到 -13 ms p50,是本报告之外最大的一笔,先落它再谈别的。
- #2126(pinned 非阻塞 restage):churn 步的 p95 约 1.2 ms API 等待。

## 3. 候选优化(按 r1 首帧收益排序,同分按 r20 p95)

### 3.1 首帧路径上的候选

**C1 准入同轮进队**(critical-path:admit-in-same-iteration)
- 证据:`analysis/skeptic-critical-path/admission_recompute.txt`。r1 13 个请求里 8 个到达时调度器忙,build_end->queue_enter 均值 8.69 / p50 8.67 / max 11.44 ms,空闲到达的 5 个只有 0.27;引擎收件到 prefill 开始 15.34 对 3.71。r20 99 个请求全部忙时到达,build_end->queue_enter 均值 23.61 / p50 23.04,正好一个剖面周期(24.8 ms)。生产口径(第 9/10 轮记录器)同段 r1 均值 3.4 / p95 8.1,r20 13.8-13.9 / p95 19.6。
- 机制:`omni_scheduler.py` process_input_requests(:868-941)在 :912-927 提交 build,唯一的 drain 点 `_drain_request_build_results`(:1089)只在下一轮的 :871/:940 被调,所以 run_batch 期间完成的 build 要白等一整轮;`_sleep_during_idle`(:972)只在空闲时 0.1 ms 轮询,这正是空闲到达只等 0.27 ms 的原因。改法:本轮提交的 future 在 get_next_batch_to_run 前有界等待(上限 1 ms,build 本身剖面 0.65-0.79 ms p50),或在本来要起 decode 步时内联 build。
- 预期:r1 首帧与可闻 TTFA 均值 -3.4 ms、p95 -7 到 -8;r20 p50/p95 -13.8 ms(当前 main),#2123 落地后步长缩到约 8.3 ms,收益变约 -8 ms。镜头原提的"builder 线程直接消费收件箱"第二步只再省 0.2-0.8 ms(build 本身),不是 -8:收件到 build 的 0.62 周期是同步循环下 U(0, step) 的固有等待。
- 工作量:中。风险:中,`_request_admission_lock` 语义、abort 竞态、准入顺序;带 arrival 的轮次多至多 1 ms;要过第 9 轮教训的 r1 100% 完成检查与 3 seed r20 underrun。
- 关联:第 11 轮已点名"新请求平均要等一整个 decode step"并提议"在 resolve 等待期间做准入",未实现;#1754 正文表格的"admission wait 13.9 ms";#1891 的 prefill coalescing 方向相反(攒批);混合 chunked prefill 作为同一 14 ms 的解法在第 11 轮作废(r20 首帧 3.7 s、完成率 63%)。

**C2 vocoder 编译 COLD 与 ramp 宽度**(vocoder:compile-cold-and-ramp-widths 与 code-catalog:vocoder-compile-ramp-widths 合并)
- 证据:formal-r1 的 COLD T=2 B=1 重放(13 次)1160 kernel、忙 3.72 / span 4.35 ms,同步落在首帧路径上(launch 到 gpu_end 4.99 p50,gpu_end 到发布 0.34),49/54 请求的可闻起点在 chunk 0;编译的 WARM T=8 B=1 是 432 kernel、忙 2.15 / span 2.40。eager WARM B=1 的忙时几乎不随 T 变(T=1 3.15,T=2..6 3.65-3.80,每 kernel 约 3.2 us),即节点数受限。r20:384 个重放里 233 个(61%)eager-in-graph,忙时和 928 ms;其中 B<=2 的 212 个跑 3.2-4.4 ms 对编译 1.9-2.1 ms。nsys:eager-in-graph 重放 SM Issue 12-16%,发射受限。每个 eager 重放带 202 个未融合的 SnakeBeta 链 elementwise kernel(exp/sin/pow/reciprocal/add),编译重放为 0。
- 机制:`streaming_vocoder.py:974-976` 与 `:1019-1021` 只给 WARM/WINDOW 传 `compile_fresh_frames=(stride,)`,COLD 不传;runner 已按 (B,T) 键预编译(`incremental_codec_cuda_graph.py:96-101, :284, :331-347`),重放不进 Dynamo(:477+)。把 WARM 扩到 (1,2,4,8)、COLD 加 (1,2),限定实际会命中的形状(COLD {1,2}x{1,2,4,8},WARM {1,2,4}x{1,2,4,8})。
- 预期:r1 首帧与可闻 -1.5(忙)到 -1.9(span)ms,36 -> 约 34;r20 首帧约 -2(COLD T=1 span 4.42 -> 约 2.4);r20 vocoder 重放 GPU 时间 -400 ms / 1484 ms(约 27%),只编 ramp (1,2,4) 则约 -300 ms(18%),尾部/抖动宽度 3,5,6,7 仍 eager。
- 工作量:低(配置 + 形状集),但启动多 20-40 个静态形状 x 约 15 s 编译,除非用 Inductor 缓存或动态 T。风险:第 7 轮 COLD 编译臂实测更差(342f36e5 -> 5f25e155 回滚,ramp (2,4) underrun 2.73 -> 4.26%、首帧 p50 110 -> 136 ms),但那是单 seed、第 9 轮 arena 入图与流解耦之前,且"运行时 guard 开销"论据对捕获后的图不成立;所以 3 seed r1+r20 复测是前提不是形式。编译对 eager 的数值差已在 T=8 接受(rel-L2 约批噪声的 1.5-2 倍,无接缝伪影),宽度 1/2/4 要重验。B>=3 的编译形状会继承 C9 的 cudnn 选算法问题,两者一起做。
- 关联:第 6 轮编译原型表,第 7 轮 COLD 编译作废,#1997(只编稳态 WARM),#1853(桶剪枝,正交)。

**C3 自适应 initial_batch_wait**(critical-path:vocoder-first-chunk-cpu 的 (a) 半;与 C2 在 r1 首帧上并列,按 r20 排在其后)
- 证据:`analysis/skeptic-critical-path/vocoder_first_chunk_r1.txt`:13/13 请求在 `_run_initial_batch` 前都有一个 2.19 ms 均值(p50 2.16,max 2.38)的 queue.get 超时;第 11 轮生产账里就是"initial 批等待 2"。第 11 轮 A/B:`initial_batch_wait_ms` 0 对 2 在 r20 首帧 p50 都是 55-57 ms,underrun 1.67% 对 0.71%,即这 2 ms 只在有同批 bootstrap 时值回来;r1 下是纯延迟。
- 机制:vocoder 已持有判断所需的状态(`streaming_vocoder.py:115-119` 的 decoded_chunks/initial_pending,:1205 已用于 suppress_bootstrap_max_streams);当没有同一 prefill 批的兄弟流 chunk 0 待处理时跳过 2 ms 超时,否则保留。规则要以"同 prefill 批有兄弟 chunk 0 挂起"为键,不能按时间窗(r20 下 20/s x 2.7 ms 窗按时间算只有约 5% 命中,会退化成 0 ms 臂)。
- 预期:r1 首帧与可闻 -2.0 ms;r20 约 0。镜头原提的 (b) 51 次 zero_ 融合与 (c) 减 COLD 图节点被 skeptic 否掉:GPU 侧 53 个 fill kernel 只 0.091 ms,第 10 轮未剖面探针整个 plan 段含 acquire 清零 0.38 ms,COLD 发射未剖面 0.28 ms,两项合计不超过 0.3-0.5 ms。
- 工作量:低。风险:低到中,r20 cohort 碎片化要 3 seed underrun 守住 0.6-1.2% 地板。关联:第 11 轮 A/B,#1997 自审第 3 条(arena release event 顺序)。

**C4 predictor 小 M GEMM 全 SM 化**(code-predictor:small-m-gemm-path;talker-decode:predictor-latency-floor 的 GEMM 半)
- 证据(`analysis/skeptic-code-predictor/v2/out/formal-r1_sk2.md`,224 个 bucket-1 步):GEMM 1969 us + splitK reduce 133 us = 2102 / 3365 us 忙(62.5%),GEMM 节点前另有 209 us 间隙;单 kernel qkv 5.27 us(grid 64 CTA,8.39 MB,1.6 TB/s)、o_proj badd 5.50 us(16 CTA,4.19 MB,0.76 TB/s)、gate_up 6.58 us(96 CTA,12.6 MB,1.9 TB/s)、down splitK 5.18 + reduce 1.66(6.29 MB,0.92 TB/s)、project_input 5.89、lm_head 4.43。bucket 48(96 步):2585 / 4568 us,1.03 TB/s = 31% 峰值,Tensor Active 2.9%。nsys 独立核对(`stats-r1_cuda_gpu_kern_sum.csv`:badd 5.58 us、splitK 5.22 us,各 29,600 次)一致。predictor 在 r1 首帧路径上两次、r20 三次。
- 机制:对 M<=64 的每子步 27 个 predictor GEMM 换成 split-K GEMV 或小 M Triton/CUTLASS kernel,每个权重矩阵铺满 132 SM,epilogue 折进 splitK reduce、bias、残差(o_proj+residual、project_input+bias、down+residual);仍在现有 predictor 图内捕获,不改子步数与采样。
- 预期:按数据时间(3.35 TB/s)加约 1.3 us 固定开销定价:qkv 3.7 / o_proj 2.6 / gate_up 5.0 / down 3.2 / project 与 lm_head 2.6 us,GEMM 2.10 -> 1.24-1.6 ms,即 bucket 1 每步 -0.5 到 -0.86 ms(镜头原估 -0.7 到 -1.0);bucket 48 -1.0 到 -1.3 ms(占 tts_engine 每步约 7.4 ms GPU 的 14-18%)。r1 首帧 -1.0 到 -1.7 ms;r20 首帧约 -3 ms p50(三次曝光)。
- 工作量:高(N 取 {1024,2048,4096,6144} 的 M 特化 GEMV 调优,M=48 需要 tensor core)。风险:bf16 累加顺序变,seeded 输出在第 15 轮记录的 7-21% 重启带内漂移,需重启控制平行对比与 TTS CI stage 2/3;要过 `_predictor_o_proj_add_residual`(`sglang_model.py:1751-1754`)的 batch-invariant / cutedsl 守卫;不适用于 #1790 的 FP8 路径。
- 关联:第 12 轮外审断言"GEMM 2.1 ms 已是地板"但没量 grid 与带宽;#1754 T-PR18 列了 norm/attention/MLP 融合但没列 GEMV;#1790(FP8 MLP,被 predictor 图 runaway 卡住)是唯一另一条 predictor GEMM 改动;容器内 SGLang 0.5.19 没有非量化路径的 bf16 小 M GEMV(只有 sm100 cutedsl 与 MoE gate GEMV 旋钮)。

**C5 predictor glue 融合 + 短上下文 attention**(code-predictor:glue-fusion-and-tiny-attention 与 code-catalog:predictor-step-fusion 合并;predictor-latency-floor 的 glue 半)
- 证据(formal-r1_sk2.md,bucket 1):rmsnorm 96 x 1.56 us = 150 us(+102 前置间隙),fused_add_rmsnorm 80 x 1.58 = 126(+21),qknorm 80 x 1.42 = 113(+18),rope_store 80 x 1.64 = 131(+75),act_and_mul 80 x 1.32 = 106(+81),splitK reduce 80 x 1.66 = 133(+21):glue 759 us kernel + 316 us 间隙;attention cudnn sdpa 75 x 3.12 = 234 us(+36)、flash_fwd 5 x 5.40 = 27 us。bucket 48:sdpa 75 x 8.54 = 640 us + flash 5 x 18.5 = 93 us,glue 945 + 425 间隙。
- 机制修正(两位 skeptic 一致):predictor 路径上 qknorm+rope+store 已是 2 个 kernel(#2057),融合是 2 -> 1,值 80 x (1.42 + 0.22 + 0.94 间隙) = 0.21 ms(b1)/ 0.27(b48);act、rmsnorm、splitK-reduce 折进 GEMM 前后序只有拥有 GEMM kernel 时才成立,属于 C4,不能重复计;torch.compile 跨不过 cuBLAS/cuDNN,trace 里 glue 被 352 段不透明节点切开(175 单、173 对、3 三、1 五),完美段内融合最多去 183 个节点(535 -> 352)即不超过 0.28 ms,镜头"490 -> 150"不可达。attention:一个 (row, head) 一 CTA、cache_len <= 16 的 decode attention kernel(1 个 key 那次输出就是 v,可跳),b1 75 x (3.12 + 0.48 - 约 2.0) = 0.12 ms,b48 75 x (8.54 + 0.83 - 约 3) + 5 x 约 15 = 0.56 ms。
- 预期:独立于 C4 时 bucket 1 每步 -0.33 到 -0.38 ms,r1 首帧 -0.66 到 -0.75;bucket 48 每步 -0.6 到 -0.83 ms(tts_engine GPU 的 11%,predictor GPU 的约 15%),r20 首帧 -1.2 到 -2.5 ms p50。与 C4 合起来 predictor 3.9 -> 约 2.0-2.3 ms(b1)、5.4 -> 约 2.5-3.0(b48)。
- 工作量:高(2-3 个手写 kernel,须可捕获)。风险:attention 数值须与 SDPA 足够接近以过 seeded 平行;Inductor 单独做不到(glue 已是 custom op)。
- 关联:#1754 T-PR18 未实现项("short-context Code Predictor attention"、"QK norm + RoPE fusion"、"RMSNorm"、"SwiGLU/MLP");第 11/12 轮候选 A;第 12 轮外审要求按约 3 ms 规划、别花一周;#2057、#2108、#1641、#1164/#871;#2114 是 AuK 的 DiT kernel,不适用。

**C6 seeded top-k/top-p 采样 kernel 加速**(code-predictor:seeded-sampling-kernel = nsys-sm:predictor-sampler-single-block = code-catalog:predictor-seeded-sampling-kernel)
- 证据:15 次/帧(不是 16,codebook 0 由 talker 的 SGLang sampler 采)x 21.7 us = 326 us,占 predictor 忙时 9.7%,grid (1,1,1) 8 warp;bucket 48 15 x 26.4 = 396 us;nsys 独立给 22.7 us 均值 / 20.2 中位数(5,550 次 r1),r1 全部 kernel 时间的 5.5%、r20 3.2%。是图里单次最贵的 kernel 类型(一次 8-12 MB 权重的 GEMV 才 5-6 us)。
- 机制:签名 ('sampled', 50, no top_p) 走 block_k=64,一个程序对 2048 个打包 uint64 键做 tl.topk(k=64),再 64 宽 softmax、float64 gumbel argmax(`sampling_kernels.py:293-437`)。换两阶段阈值/radix select 或 warp 协作部分排序,保留 (score desc, index asc)、+0 先于 -0、murmur3、float64 gumbel 的契约;因打包键唯一,任何精确 top-64 选择返回同一有序集合。
- 预期:21.7 -> 6-9 us 现实(镜头说 5-6),bucket 1 每步 -0.19 到 -0.27 ms,r1 首帧 -0.38 到 -0.53(正好在 0.5 ms 门槛);bucket 48 -0.26 到 -0.32 ms(tts_engine GPU 的 3.5-5%),r20 首帧 -0.8 到 -1 ms p50。上限约 7% 的 predictor。
- 工作量:中。风险:中,bit-parity 契约(#1641 的 torch.topk gatherTopK 顺序),k=50/64 的平行测试要扩,用第 15 轮重启控制协议验。关联:#1641(kernel 本体)、#871、#1239、#1726、#1971;第 11 轮账列了 0.31 ms 但候选 A 没碰它。

**C7 predictor 双 token prologue 合并**(code-predictor:two-token-prologue)
- 证据:每步 segment 0 = 124 kernel,忙 394 / span 456 us(b1),726 us(b48);其中 cache_len 0 的 forward#1 55 kernel、192 us 忙 / 220 span,attention 5 x flash_fwd 5.40 us 只对 1 个 key(输出就是 v);forward#2 170 / 199。`sglang_model.py:1507-1518` 两次调 `_predictor_forward_one_token`,两个输入(talker_predictor_embed、layer0_predictor_embed)在循环前都已有;`_predictor_cached_self_attention` 在 seq_len != 1 时 raise(:1818),位置/槽按 [cache_len, :batch] 索引(:893-918)。
- 机制:一次 M=2 forward 预填两个 token(M=2 GEMM 与 M=1 同价,2q x 2k 因果 SDPA),从 cache_len 2 起继续 14 个单 token 子步;重捕获 predictor 图,不改采样。
- 预期:b1 每步约 -0.21 ms,r1 首帧约 -0.43(低于 0.5 门槛);b48 约 -0.30(M=96 需第二个 M-tile,+15-40 us),占 tts_engine GPU 4%,r20 首帧约 -0.8 ms p50。
- 工作量:低。风险:低,M=2 的 cuBLAS 选核只影响前两位置的 KV,在重启带内。关联:无先例(#1134/#1947/#1971/#2057 都没动 prologue);建议作为任何 predictor 图重捕获的搭车项。

**C8 vocoder arena 扁平化,单次 gather/scatter**(vocoder:flat-arena-single-gather-scatter)
- 证据:r1 vocoder 流 9692 个 index_select/index_copy kernel,33.8 ms = 10.3%;编译 T=8 B=1 重放 76 个 0.28 ms(13.0%),eager 重放 92 个 0.30 ms;单 kernel p50 3.3 us(index_copy 4.0,index_select 3.1),节点间隙 0.45 us。r20 32,896 个,135.6 ms = 9.1%。arena 是 2 x 38 个缓冲(frame_positions + conv/transconv histories + 8 K + 8 V)各一次 index_select 与 index_copy_(`codec_state_arena.py:187-258`)。
- 机制:所有 bf16 状态缓冲放进一个 [num_slots+1, total_elems] 存储,按层暴露视图;gather 变一次 index_select 出 [B, total],scatter 变一次 index_copy_。skeptic 修正:gather 半(38 -> 2,约 -0.13 ms)要求解码器接受 flat 行的跨步视图(histories 经 cat 消费,可行);scatter 半只有解码器就地写进 flat staging 视图才成立,否则每层一次 copy 换一次 index_copy。
- 预期:每次重放 -0.13 到 -0.27 ms;r1 首帧 -0.13 到 -0.27(低于门槛);r20 vocoder GPU -50 到 -104 ms / 1496(3.4-7%)。工作量:中。风险:中(视图连续性、release/zero 路径)。关联:第 9 轮 arena 入图(5bd83547 / 2aa92f9b)、第 5/6 轮。

### 3.2 只影响稳态产能、不动首帧的候选

**C9 修 B4/B8 vocoder 图的 cudnn 算法选择**(vocoder:cudnn-algo-wide-buckets = nsys-sm:vocoder-legacy-conv-b4b8 = code-catalog:vocoder-cudnn-implicit-convolve,三镜头独立发现)
- 证据:nsys r20 454 次 `implicit_convolve_sgemm<__nv_bfloat16,__nv_bfloat16,128,6,7,3,3,5,1>`,302 ms = 全部 kernel 时间 6.12%、vocoder 流 12.85%;B4 图 26/122(91 次)每次 4 个 x 552-569 us = 2.21-2.28 ms,占 5.29-5.45 ms span 的 42%;B8 图 98/2(45 次)每次 2 个 x 1.08 ms = 2.15-2.19 ms,占 6.36-6.43 ms 的 34%;torch trace(`analysis/code-catalog/conv_probe_r20.md`)290 次 203.1 ms = 12.1%,graph 14(eager T=4 B=8)也有 4 个。同层在 B1/B2 图走 cutlass/xmma tensor-op kernel 13-46 us;每次发射恒定 4/2 个,说明算法在捕获时定死。r1 0 次,COLD 0 次。这些重放 SM Active 80-82%、SM Issue 37-40%:是在 CUDA core 上算满的,不是发射问题。
- 机制:捕获 warmup 阶段(`incremental_codec_cuda_graph.py:_warmup_capture_shape`)在默认启发式下把引擎冻进图;改为在 precompile + capture 范围内开 `torch.backends.cudnn.benchmark=True`(容器内 torch 2.13.0+cu130 默认 False,sglang 只在 minimax_h3 强制 False,`sglang_omni/models/qwen3_tts` 没有任何 cudnn 设置),或对这几层固定 channels_last;用 `graph_nodes.py` 验证图里不再有 implicit_convolve_sgemm 节点。
- 预期:B4 重放 4.7 -> 约 2.9 ms(-1.8)、B8 5.8 -> 约 3.9(-1.85);r20 vocoder 重放 GPU -160 到 -285 ms(-10 到 -12%),B=4 每行成本 1.16 -> 约 0.65 ms;首帧 0,r1 0;underrun 已在地板,收益是产能与宽 cohort 的 follow-up 延迟。
- 工作量:低。风险:低;唯一未验证的是 cudnnFind 是否真会给这两个形状选到 tensor-core 引擎(需要 GPU);第 6 轮"cudnn.benchmark 无影响"是在 B2 eager 测的,那里本就不发生这个选择,不构成反证。关联:第 6 轮、第 7 轮 channels-last 原型(只针对转置)、#1853(正交)。

### 3.3 叠加账(算术上限,不含交互,没有一条实测)
- r1 首帧 p50:main breakable 35.6 -> #1998 32.1 -> #2123 约 -1.9(链上一步的空洞)-> C1 -3.4 -> C2 -1.5 到 -1.9 -> C3 -2.0 -> C4+C5+C6+C7 合计 predictor 每帧 -1.2 到 -1.7、两次曝光 -2.5 到 -3.4 -> 约 19-21 ms。参照实现约 26 ms。即便只落 #2123 + C1 + C3 三条低风险项也到约 25 ms。
- r20 首帧 p50:full 49.6-50 -> #2123 约 -10 到 -13(未测,按周期比例)-> C1 -8 -> predictor 四项 -4 到 -5 -> C2 -2 -> 约 22-26 ms;p95 从 64-67 起按同比例。参照约 26 / 38。
- 含义:主线离参照实现的差距不是"没有余量",而是差 3-4 个各自 2-14 ms 的已定位项;其中最大两笔(#2123、C1)都是 CPU/调度,不是 kernel。

## 4. 已排除
- talker-decode:r20-step-cpu-work(r20 步约 46% CPU,向量化每请求 Python + 异步循环):(b) 半就是 #2123(尾巴藏到 predictor 下,1 行步 8.04 -> 6.25,且把 lookahead_eligible 设 False 关掉了异步循环路线),restage 项是 #2126;nsys 推演 #2123 之后暴露 CPU p50 为 0(r1 358/358 步、r20 p50 0 / p95 3.9),剩的只有 r20 p95。
- talker-decode:prepare-decode-buffers-pageable-h2d(六个 pageable H2D + 隐式同步):与 #2126 逐字相同;拆开算 API 只占 p50 0.09 ms/变更步(r20)、r1 首帧路径 0.06 ms,-0.5 ms 的估计把 Python 建列表也算进去了,而那部分 #2126 也保留。
- talker-decode:breakable-prefill-eager-attention:就是 #1998,收益已实测(r1 35.6 -> 32.1),零新增。
- vocoder:wake-on-completion-publish(完成事件触发提交而不是等 4 ms 收集超时):延迟真实(r1 gpu_end->finish p50 2.0 ms,100% 走超时;r20 p95 7.8),但 COLD 首块同步,首帧 0;GPU 时间 0;underrun 已在地板;"68/80 ms"来自过载的剖面运行。只是 follow-up 块延迟的打磨,属 #1754 T-PR10。
- nsys-sm:talker-step-cpu-hole(把每步 CPU 与下一步重叠):就是 #2123;按 predictor 启动而非结束起算尾巴,残余空洞 r1 p50 0.0(99.7% 步为 0)、r20 p50 0.0(均值 0.8,p95 4.8,剖面膨胀);"把 eager kernel 折进图"只值 0.03-0.05 ms。
- nsys-sm:talker-gemv-bandwidth(bs=1 talker 图 41% 带宽):GEMM 口径 2.17 TB/s = 65% 峰值,40.8% 是整图含 attention/glue/气泡的平均;85% 峰值上限每步 0.31 ms(r1)、0.38(r20),低于门槛;"合并 QKV 与 gate_up"已是现状;SGLang 0.5.19 自带 --bf16-gemm-backend gemv 但只对 m==1、不覆盖 n=12288,自报 5-15%,即 0.07-0.2 ms,且未接到 Qwen3-TTS;这些 GEMM 上的带宽杠杆是在途 #1790(FP8 MLP,runaway 阻塞)。
- critical-path:async-decode-loop(为 Qwen3-TTS 实现 post_decode_launch/resolve 开异步循环):T-PR12/#1643 已关闭(无收益 + 无安全的逐行 penalty 交接);main 上 lookahead_eligible 对带 penalty 的批强制同步而默认 repetition_penalty 1.05;#2123 把 lookahead_eligible 设 False 并用一次拷贝重排拿到同样的重叠;nsys 推演在 #2123 之上再做异步只多 0.6 ms 均值、p50 0。
- critical-path:batched-chunk-emit(每步一条 [B,Q] 消息代替逐请求 emit):3.3 倍慢化真实(0.049 -> 0.162 ms/块,GIL),但生产尾巴总共 4.4 ms p50,emit 最多约 2 ms,且整段在 #2123 藏进 predictor 的尾巴里;增量 p50 0、均值 -0.5 ms;r1 小于 0.1。
- 量化了但未列为候选(收益小或已有原型):talker 的 MRoPE 三 kernel 阶梯(mrope_section=[24,20,20] 让 get_rope 返回 MRotaryEmbedding 从而关掉融合,84 节点占 decode 图 6-7%,但三行位置相等);talker 层 0 的 eager SGLang sampler(25.6 kernel/步,86 us);vocoder nchw<->nhwc 转置(r1 19.7%、r20 15.2% 的 vocoder kernel 时间,第 7 轮 channels-last 原型 62 -> 19 个、1.90 -> 1.66 ms,不在 main、无 PR);PDL(图内间隙只 10-15%,天花板小);preprocessing 的 9 次 pageable H2D(剖面 4.86 ms,GPU 0.16)。

## 5. 下一步实验(前三)

**E1 准入同轮进队 A/B(C1)**。改 `omni_scheduler.py` process_input_requests:本轮提交的 build future 在 get_next_batch_to_run 前有界等待(上限 1 ms),超时则照旧下轮 drain。四臂 x 3 seed:main、main+#2123、各自 +C1;harness 同第 15 轮(eval-h100 单 H100,CustomVoice Ryan,seed-tts-eval,开环泊松,--rps 1 与 --rps 20,--warmup 30s --duration 60s,seed 0/1/2),开请求事件记录器。测:build_end->queue_enter 与 eng_in->prefill_start(期望 r1 忙时到达 8.7 -> <1 ms 剖面 / 3.4 -> <0.5 生产,r20 13.8 -> <1),first playable p50/p95、可闻 TTFA、underrun、100% 完成、E2E p95/p99(第 11 轮混合 prefill 的尾部教训)。判据:r1 均值 -3 ms 且尾部不变;r20 p50 -8 ms 以上(在 #2123 之上)。

**E2 vocoder 编译集与 cudnn 算法(C2 + C9 + C3)**。臂:(a) 基线;(b) WARM `compile_fresh_frames=(1,2,4,8)`;(c) b + COLD (1,2);(d) c + 捕获期 `torch.backends.cudnn.benchmark=True`;(e) d + 自适应 initial_batch_wait(同 prefill 批有兄弟 chunk 0 挂起才等 2 ms)。每臂先用 `graph_nodes.py` 在短 torch profile 上核对:COLD/ramp 图节点数约 430、B4/B8 图内 implicit_convolve_sgemm 为 0、启动捕获耗时;然后 r1 + r20 各 3 seed。测:r1 首帧与可闻(期望 b 不变,c -1.5 到 -1.9,e 再 -2),r20 首帧、underrun(必须留在 0-0.4%)、vocoder 流 union 忙(期望 33.7% -> 约 26%),按第 15 轮重启控制协议做接缝一致性(同配置重启对照 + 编译对 eager)。第 7 轮 COLD 编译作废是单 seed,这里 3 seed 说了算。

**E3 predictor 小 M GEMV 离线微基准(C4 的 go/no-go,不碰服务)**。对六个形状(qkv 1024x4096、o_proj 1024x1024 加残差、gate_up 1024x6144、down 3072x1024 折 reduce、project_input 与 lm_head 1024x2048)在 M in {1,2,4,8,48} 用 Triton split-K GEMV 对 cuBLAS 现状,CUDA graph 内重放测 us/次与 TB/s。判据:o_proj 类 4.19 MB 形状不超过 3 us、gate_up 不超过 5.5 us、M=48 不劣于 cuBLAS,并且 bf16 输出与 cuBLAS 的差异同 torch 参考量级;达标再在 `sglang_model.py` 接入并做 E1 同规格的 A/B 加 TTS CI stage 2/3 与重启控制平行。不达标就把 C4 从清单划掉,C5/C6/C7 独立推进(C7 可搭任何 predictor 图重捕获的车)。

## 6. 数据局限
- CPU 侧膨胀:with_stack=1 让 Python 密集段膨胀(r1 decode 周期剖面 9.1 对生产 7.2 ms,prefill 16.0 对 12.0,r20 每步 Python 约 3 倍);CUPTI graph node 级追踪把 cudaGraphLaunch 从未剖面的 0.1-0.3 ms 抬到 1.0-1.5 ms;nsys 的 node 级追踪 + osrt 每步约 +0.5-0.8 ms(r1 步 8.03 对 7.2),其客户端首帧 38.7 ms(r1)/ 60.4 与 p95 87.4 ms、underrun 2.63%(r20)都劣于生产。所有"launch-bound"分类与 CPU 空洞都是上界;kernel 时长与 GPU metrics 不受影响。
- formal-r20 是过载运行:在飞 33-51、等待队列 p95 413 ms、超过 suppress_bootstrap_max_streams=24 后首块只带 1 帧(剖面可闻 TTFA p50 180 ms 对 TTFB 94),43% 请求可闻起点落到 chunk 1。其绝对数只用于结构与比例,生产 r20 尺寸一律引第 9/10/11/15 轮记录器数字。
- 样本:单 seed 单窗口;r1 12-13 个请求 / 369 步,r20 401 步(skeptic 重算 233 步);vocoder r1 117 次重放、r20 385 次;nsys r20 GPU metrics 只覆盖 63 s 捕获的最后 6.1 s;nsys 与 torch 剖面是不同的服务进程。
- trace 里没有 cpu_op 事件(record_shapes 没产出),kernel 到 Python 的归属靠发射线程的最内层 Python 帧(图重放归到调 cudaGraphLaunch 的帧);B4/B8 那几层 conv 靠 launch 几何与相邻 Inductor kernel 名识别,不是记录的输入形状;编译 vocoder 形状靠每次重放的 kernel 数(416/427/432/433)区分,B4 对 B8 是推断;vocoder 桶靠 nhwcToNchw 的 grid 比(617/1234/2371/4647)推断;每步 batch 靠采样 kernel 的 grid 或 _peek_next_decode_inputs 计数推断。
- nsys GPU metrics 是 50 us 的设备级采样,中位 kernel 才 2-2.5 us,忙时条件化用逐样本重叠加权;SM Issue 是 132 个 SM 的总量,低值混合了"grid 小"与"SM 内停顿"(已另报按活跃 SM 折算值);r20 predictor 51% 的时间与 vocoder 流重叠,其 SM Active 50% 是上界(r1 逐次 p50 18% 是干净的)。
- 未测量:每步 talker batch(nsys 不带);哪几层 conv 命中 implicit_convolve_sgemm(需 cuDNN 日志);CPU 空洞里 Python/API 与 GIL 争用的拆分(osrt 数据可答但未分析,critical-path 的 3.3 倍 emit 慢化是唯一直接的争用证据);r20 export 有 427 条 r1 时段的陈旧记录已剔除;talker 权重 2.82 GB 按架构算不是读 checkpoint。
- BBuf 的 profiler-analysis skill 在 formal-r1 上跑通(EXIT 0)但只做了交叉检查:它按 cpu_op/kernel 配对设计,把整个三阶段进程当一个 extend 阶段,顶行"FP8 scaled MM"是对 bf16 nvjet 的名字误配,没有识别 vocoder 项;所有表都是自写解析。
- 所有候选收益是逐 kernel 账的算术推演,没有重跑;C4 的 GEMV 目标速率假设小 N GEMV 能到 HBM 峰值的 60-75%,C2 假设编译 T<8 图与编译 T=8 B=1 同价,C9 假设 cudnnFind 能选到 tensor-core 引擎;#2123 在 r20 首帧的收益按周期比例推算,PR 本身只测了 1 行与 16 行单步。