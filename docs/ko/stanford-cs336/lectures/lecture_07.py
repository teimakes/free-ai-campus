import torch
import time
import math
import sys
import os
from inspect import isfunction
from typing import Callable
from torch import nn, tensor
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from edtrace import text, image, link
from gpu_util import cuda_if_available
from lecture_util import article_link

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda: None  # CUDA를 사용할 수 없으면 아무 연산도 하지 않는다.

def main():
    text("# 7강. 병렬 처리")
    text("지난주: GPU 한 대 안에서의 병렬 처리")
    text("이번 주: 여러 GPU에 걸친 병렬 처리")
    image("images/gpu-node-overview.png", width=700)

    text("두 경우 모두 **연산**을 수행하는 산술 논리 장치와 입출력 **데이터**가 멀리 떨어져 있다.")
    text("공통 과제는 데이터 전송이 병목이 되지 않도록 연산을 배치하는 것이다.")

    text("하드웨어 계층을 구분하면 다음과 같다.")
    text("- 단일 노드, 단일 GPU: L1 캐시·공유 메모리(가장 빠름)")
    text("- 단일 노드, 단일 GPU: HBM")
    text("- 단일 노드, 여러 GPU: NVLink/NVSwitch")
    text("- 여러 노드, 여러 GPU: InfiniBand/이더넷(가장 느림)")

    text("지난주에는 융합과 타일링으로 메모리 접근을 줄였다.")
    text("이번 주에는 복제와 분할로 GPU 사이, 노드 사이의 통신을 줄인다.")

    text("GPU를 여러 대 사용하는 이유는?")
    text("1. 매개변수와 옵티마이저 상태, 기울기, 활성값이 GPU 한 대에 들어가지 않는다.")
    text("2. GPU를 늘려 연산량을 분담하고 더 빠르게 학습하고 싶다.")

    # 이 강의를 직접 실행하면(python lecture_07.py) 다중 프로세스를 사용한다. 각 프로세스의 출력은 아래에서 볼 수 있다.
    # 실행을 추적할 때는(python -m edtrace.execute -m lecture_07) 다중 프로세스를 끈다.
    link(title="이 강의의 표준 출력", url="var/traces/lecture_07_stdout.txt")

    text("### 1부. 분산 통신과 연산의 기본 요소")
    collective_operations()    # 프로그래밍 모델
    hardware()                 # GPU를 연결하는 하드웨어
    torch_distributed()        # NCCL과 PyTorch에서의 구현
    benchmarking()             # 실제 NCCL 대역폭 측정

    text("### 2부. 분산 학습")
    text("여러 층으로 이루어진 MLP에서 각 전략을 최소한으로 구현한 예를 살펴본다.")
    text("Transformer에서는 MLP가 연산 병목이므로 이 예가 대표성을 갖는다.")

    data_parallelism()         # 배치 차원을 따라 나눈다.
    tensor_parallelism()       # 너비 차원을 따라 나눈다.
    pipeline_parallelism()     # 깊이 차원을 따라 나눈다.

    text("여기에서 다루지 않은 것은?")
    text("- 통신과 연산을 겹쳐 실행하기")
    text("- 어텐션 등이 있는 더 일반적인 모델")
    text("- 그 밖의 병렬화 방식(시퀀스 병렬화, 전문가 병렬화, 여러 방식의 조합)")
    text("- JAX/TPU에서는 모델과 분할 전략만 정의하면 나머지는 JAX 컴파일러가 처리한다. "), link(title="levanter", url="https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html")
    text("- 여기서는 PyTorch를 사용해 기본 연산만으로 구현하는 과정을 살펴본다.")

    text("### 정리")
    text("- 병렬화할 수 있는 차원은 데이터(배치), 텐서·전문가(너비), 파이프라인(깊이), 시퀀스(길이) 등 다양하다.")
    text("- 데이터 병렬화: DDP(all-reduce), FSDP/ZeRO(all-gather + reduce-scatter).")
    text("- 텐서 병렬화에는 NVLink처럼 매우 빠른 연결이 필요하다.")
    text("- 파이프라인 병렬화는 느린 연결에서도 사용할 수 있지만 장치가 쉬는 구간(파이프라인 버블)을 줄여야 한다.")
    text("- 값을 **다시 계산**하거나, **메모리에 저장**하거나, 다른 GPU에 저장한 뒤 **통신**으로 가져올 수 있다.")
    text("- 하드웨어가 빨라져도 더 큰 모델을 사용하려는 수요가 있어 이러한 계층 구조는 계속 필요하다.")


def collective_operations():
    text("**집합 통신 연산**은 분산 프로그래밍을 구성하는 기본 연산이다. "), article_link("https://en.wikipedia.org/wiki/Collective_operation")
    text("- 1980년대 병렬 프로그래밍 문헌부터 다뤄 온 개념이다.")
    text("- *집합*이라는 말은 여러 장치가 따를 공통 통신 패턴을 지정한다는 뜻이다.")
    text("- 장치 사이의 일대일 통신을 직접 관리할 때보다 간편하고 빠를 수 있다.")

    text("**설정**")
    image("images/ranks.png", width=500)
    text("- **랭크(rank)**: 장치 또는 GPU 하나를 가리키는 번호(예: 0, 1, 2, 3).")
    text("- **월드 크기(world size)**: 전체 장치 수(예: 4).")

    text("연산의 종류")
    text("- broadcast, scatter, gather, reduce: 기본 연산.")
    text("- all-gather, reduce-scatter, all-reduce: 자주 쓰는 연산.")
    text("- all-to-all: 전문가 혼합(MoE)에서 사용.")

    text("**Broadcast**: 랭크 0의 값을 모든 랭크로 복사한다.")
    # 입력
    rank0 = tensor([0., 1, 2, 3])

    # 출력
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([0., 1, 2, 3])
    rank2 = tensor([0., 1, 2, 3])
    rank3 = tensor([0., 1, 2, 3])

    text("활용 예: 랭크 0이 초기 체크포인트를 읽고 모든 랭크에 보낸다.")

    text("**Scatter**: 랭크 0의 텐서를 나누어 모든 랭크에 보낸다.")
    # 입력
    rank0 = tensor([0., 1, 2, 3])

    # 출력
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    text("이 연산을 이해하면 reduce-scatter를 이해하기 쉽다.")

    text("**Gather**: 모든 랭크의 부분 텐서를 랭크 0으로 모은다. scatter와 반대다.")
    # 입력
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # 출력
    rank0 = tensor([0., 1, 2, 3])

    text("이 연산을 이해하면 all-gather를 이해하기 쉽다.")

    text("**Reduce**: 모든 랭크의 값을 합, 최솟값, 최댓값 등의 연산으로 집계해 랭크 0에 둔다.")
    # 입력
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # 출력
    rank0 = tensor([6.])  # 모든 랭크의 값 합계(0 + 1 + 2 + 3)

    text("이 연산을 이해하면 all-reduce를 이해하기 쉽다.")

    text("**All-gather**: 모든 랭크가 gather의 결과 전체를 받는다.")
    # 입력
    rank0 = tensor([0.])
    rank1 = tensor([1.])
    rank2 = tensor([2.])
    rank3 = tensor([3.])

    # 출력
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([0., 1, 2, 3])
    rank2 = tensor([0., 1, 2, 3])
    rank3 = tensor([0., 1, 2, 3])

    text("활용 예: 각 랭크에 매개변수를 나눠 저장한 뒤 순전파에 필요한 전체 매개변수를 모은다.")

    text("**Reduce-scatter**: 각 차원에 대해 reduce를 수행하고 결과를 나누어 보낸다.")
    # 입력
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([1., 2, 3, 4])
    rank2 = tensor([2., 3, 4, 5])
    rank3 = tensor([3., 4, 5, 6])

    # 출력
    rank0 = tensor([6.])  # 0번 차원을 따라 합산(0 + 1 + 2 + 3)
    rank1 = tensor([10.]) # 1번 차원을 따라 합산(1 + 2 + 3 + 4)
    rank2 = tensor([14.]) # 2번 차원을 따라 합산(2 + 3 + 4 + 5)
    rank3 = tensor([18.]) # 3번 차원을 따라 합산(3 + 4 + 5 + 6)

    text("활용 예: 역전파 후 각 데이터 분할에서 계산한 기울기를 합하고, 저장 공간은 여러 장치에 분산한다.")

    text("**All-reduce** = reduce-scatter + all-gather")
    # 입력
    rank0 = tensor([0., 1, 2, 3])
    rank1 = tensor([1., 2, 3, 4])
    rank2 = tensor([2., 3, 4, 5])
    rank3 = tensor([3., 4, 5, 6])

    # 출력
    rank0 = tensor([6., 10, 14, 18])
    rank1 = tensor([6., 10, 14, 18])
    rank2 = tensor([6., 10, 14, 18])
    rank3 = tensor([6., 10, 14, 18])

    text("활용 예: 역전파 후 각 데이터 분할의 기울기를 합하고 전체 매개변수는 각 장치에 복제한다.")
    text("all-reduce를 reduce-scatter와 all-gather로 나누면 ZeRO/FSDP처럼 연산 배치를 바꿀 수 있다.")

    text("**All-to-all**: 각 랭크가 다른 각 랭크에 텐서를 보낸다. 가장 일반적인 통신 방식이다.")
    # 입력
    rank0 = tensor([0., 1, 2, 3])      # 랭크 0에 0, 랭크 1에 1, 랭크 2에 2, 랭크 3에 3을 보낸다.
    rank1 = tensor([4., 5, 6, 7])      # 랭크 0에 4, 랭크 1에 5, 랭크 2에 6, 랭크 3에 7을 보낸다.
    rank2 = tensor([8., 9, 10, 11])    # 랭크 0에 8, 랭크 1에 9, 랭크 2에 10, 랭크 3에 11을 보낸다.
    rank3 = tensor([12., 13, 14, 15])  # 랭크 0에 12, 랭크 1에 13, 랭크 2에 14, 랭크 3에 15를 보낸다.

    # 출력
    rank0 = tensor([0, 4, 8, 12])
    rank1 = tensor([1, 5, 9, 13])
    rank2 = tensor([2, 6, 10, 14])
    rank3 = tensor([3, 7, 11, 15])

    text("참고")
    text("- MoE에서 각 랭크에는 일부 데이터와 일부 전문가가 있으므로 데이터를 담당 전문가에게 보내야 한다.")
    text("- 균등하게 나누면 all-to-all은 전치 연산처럼 보인다.")
    text("- 불균등한 분할도 처리할 수 있지만 되도록 균등하게 나누는 것이 좋다.")

    text("용어를 기억하는 방법")
    text("- Reduce: 합, 최솟값, 최댓값처럼 결합법칙과 교환법칙이 성립하는 연산으로 값을 집계한다.")
    text("- Scatter와 gather는 반대 방향의 연산이다.")
    text("- All: 결과의 목적지가 모든 장치라는 뜻이다.")


def hardware():
    text("전통적인 구성(가정용)")
    image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs42774-021-00098-3/MediaObjects/42774_2021_98_Fig1_HTML.png?as=webp", width=500)
    text("- 같은 노드에 있는 GPU는 PCI(e) 버스로 통신한다(v7.0, 16레인에서 242 GB/s). "), article_link("https://en.wikipedia.org/wiki/PCI_Express")
    text("- 다른 노드에 있는 GPU는 이더넷으로 통신한다(약 200 MB/s).")
    
    text("현대적인 구성(데이터센터)")
    image("images/gpu-node-overview.png", width=700)

    text("일반적인 예")
    text("- 노드당 GPU 8개를 NVLink로 NVSwitch에 연결한다. B200의 NVLink 5.0은 1.8 TB/s이며 HBM 대역폭은 8 TB/s였다.")
    text("- 포드당 노드 256개를 InfiniBand로 연결한다(PCIe → HCA 또는 InfiniBand NIC → InfiniBand 케이블, 약 0.05 TB/s).")
    text("- 클러스터 또는 데이터센터에는 포드 N개가 있으며 이더넷으로 연결한다(PCIe → CPU).")

    text("CPU를 거치지 않는 통신")
    text("- 이더넷은 CPU를 거친다. 데이터를 커널 소켓 버퍼에 복사하고 TCP 패킷을 만든 뒤 NIC 링 버퍼에 복사한다.")
    text("- RDMA(원격 직접 메모리 접근)는 GPU가 CPU를 거치지 않고 다른 GPU의 메모리를 직접 읽고 쓰게 한다.")
    text("- InfiniBand는 RDMA를 지원한다. 일반적인 이더넷은 이를 지원하지 않는다.")

    text("이후의 발전")
    text("- GB200/GB300 NVL72: 트레이당 GPU 8개, 랙당 트레이 9개로 NVLink 도메인 하나에 GPU 72개가 들어간다.")
    text("- RoCE(RDMA over Converged Ethernet): 이더넷에서 CPU를 거치지 않는다. InfiniBand와 비슷하고 가격이 낮지만 성능도 낮다. Meta에서 사용한다.")

    text("### NVIDIA 집합 통신 라이브러리(NCCL)")
    text("NCCL은 집합 통신 연산을 GPU 사이에 전송할 저수준 패킷으로 변환한다. "), link(title="발표", url="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/")
    text("- 노드와 스위치 수, NVLink·PCIe 연결 등 하드웨어의 연결 구조를 파악한다.")
    text("- GPU 사이의 통신 경로를 최적화한다.")
    text("- 송수신을 수행할 GPU 커널을 실행한다.")


def torch_distributed():
    text("PyTorch 분산 라이브러리(`torch.distributed`) "), link(title="문서", url="https://pytorch.org/docs/stable/distributed.html")
    text("- `all_gather_into_tensor` 등 집합 통신 연산을 호출할 수 있는 인터페이스를 제공한다.")
    text("- 하드웨어에 맞춰 여러 백엔드를 지원한다. CPU에서는 gloo, GPU에서는 nccl을 쓴다.")
    text("- `FullyShardedDataParallel` 같은 상위 수준의 알고리즘도 지원한다. 이 강의에서는 사용하지 않는다.")

    text("예제를 살펴보자.")
    spawn(collective_operations_main, world_size=4)


def collective_operations_main(rank: int, world_size: int):  # @inspect rank world_size
    "이 함수는 각 프로세스(rank = 0, ..., world_size - 1)에서 비동기적으로 실행된다."
    setup(rank, world_size)

    ### All-reduce (dist = torch.distributed)
    dist.barrier()  # 모든 프로세스가 이 지점에 도달할 때까지 기다린다(여기서는 출력을 위해서다).

    data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank  # 입력이자 출력

    print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # 텐서를 제자리에서 수정한다.
    print(f"Rank {rank} [after all-reduce]: {data}", flush=True)

    ### Reduce-scatter
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=cuda_if_available(rank)) + rank  # 입력
    output = torch.empty(1, device=cuda_if_available(rank))  # 출력을 할당한다.

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    ### All-gather
    dist.barrier()

    input = output  # reduce-scatter의 출력을 입력으로 쓴다.
    output = torch.empty(world_size, device=cuda_if_available(rank))  # 출력을 할당한다.

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    text("실제로 all-reduce = reduce-scatter + all-gather가 성립한다.")

    cleanup()


def benchmarking():
    text("실제 통신 속도는 어느 정도인가?")

    # All-reduce
    spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)

    # Reduce-scatter
    spawn(reduce_scatter, world_size=4, num_elements=100 * 1024**2)

    text("참고 자료")
    link(title="집합 통신 연산을 분석하는 방법", url="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce")
    link(title="벤치마크 예제 코드", url="https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py")


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)  # @stepover

    # 텐서 생성
    data = torch.randn(num_elements, device=cuda_if_available(rank))

    # 준비 실행
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # CUDA 커널이 끝날 때까지 기다린다.
    dist.barrier()            # 모든 프로세스가 여기에 도달할 때까지 기다린다.

    # all-reduce 수행
    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # CUDA 커널이 끝날 때까지 기다린다.
    dist.barrier()            # 모든 프로세스가 여기에 도달할 때까지 기다린다.
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)  # @stepover

    # 유효 대역폭 측정
    dist.barrier()
    size_bytes = data.element_size() * data.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 송신과 수신을 모두 세므로 2를 곱한다. all-reduce에는 world_size-1단계가 있다.
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    # 참고
    # - 유효 대역폭 ≈ 2 * size_bytes / total_duration
    # - world_size에 무관하다.
    # - 링 또는 트리 등 연결 구조에 무관하다.

    cleanup()  # @stepover


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)  # @stepover

    # 입력과 출력 생성
    input = torch.randn(world_size, num_elements, device=cuda_if_available(rank))  # 랭크마다 행렬 하나를 갖는다.
    output = torch.empty(num_elements, device=cuda_if_available(rank))

    # 준비 실행
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # CUDA 커널이 끝날 때까지 기다린다.
    dist.barrier()            # 모든 프로세스가 여기에 도달할 때까지 기다린다.

    # reduce-scatter 수행
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # CUDA 커널이 끝날 때까지 기다린다.
    dist.barrier()            # 모든 프로세스가 여기에 도달할 때까지 기다린다.
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)  # @stepover

    # 유효 대역폭 측정
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # 입력 데이터의 양
    sent_bytes = data_bytes * (world_size - 1)  # 전송해야 하는 양(여기서는 2를 곱하지 않는다)
    total_duration = world_size * duration  # 전체 전송 시간
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    # 참고
    # - all-reduce = reduce-scatter + all-gather
    # - all-reduce는 reduce-scatter의 두 배에 해당하는 데이터를 두 배의 시간에 전송하므로 대역폭은 비슷하다.

    cleanup()  # @stepover


def data_parallelism():
    image("images/data-parallelism.png", width=300)
    text("분할 전략: 각 랭크가 데이터 일부를 받는다.")

    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)

    text("참고")
    text("- 각 랭크가 자기 데이터로 계산하므로 손실은 랭크마다 다르다.")
    text("- 기울기에는 all-reduce를 적용해 모든 랭크에서 같은 값을 갖게 한다.")
    text("- 따라서 매개변수도 모든 랭크에서 같은 값을 유지한다.")

    text("다음 강의의 FSDP/ZeRO: all-gather와 reduce-scatter를 사용해 전체 매개변수를 늘 메모리에 보관하는 비용을 줄인다.")


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


def data_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)  # @stepover

    # 이 랭크의 데이터를 선택한다. 실제로는 각 랭크가 자기 데이터만 읽으면 된다.
    # --- B0 ---
    # --- B1 ---
    # --- B2 ---
    # --- B3 ---
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size @stepover
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(cuda_if_available(rank))

    # MLP 매개변수 params[0], ..., params[num_layers - 1]을 생성한다. 모든 랭크가 매개변수 전체를 갖는다.
    params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # 각 랭크는 자체 옵티마이저 상태를 갖는다.

    for step in range(num_steps):
        # 순전파
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # 손실 함수는 크기의 제곱을 평균한 값이다.

        # 역전파
        loss.backward()

        # 작업자 사이에서 기울기를 동기화한다. 일반 학습과 DDP의 유일한 차이다.
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # 매개변수 갱신
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[layer]) for layer in range(num_layers)]}", flush=True)  # @stepover

    cleanup()  # @stepover


def tensor_parallelism():
    image("images/tensor-parallelism.png", width=300)
    text("분할 전략: 각 랭크가 각 층의 일부를 맡으며 전체 데이터와 활성값을 전송한다.")

    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int):
    setup(rank, world_size)  # @stepover

    data = data.to(cuda_if_available(rank))  # 모든 랭크가 데이터를 받는다(batch_size × num_dim).
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # `num_dim`을 분할한다.  @inspect local_num_dim @stepover

    # 모델을 생성한다. 각 랭크는 매개변수의 1/world_size를 갖는다.
    #  |  |  |  |
    # W0 W1 W2 W3
    #  |  |  |  |
    params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]

    # 순전파
    x = data
    for layer in range(num_layers):
        # 활성값 계산(batch_size × local_num_dim)
        x = x @ params[layer]  # 매개변수의 일부에 대해서만 계산한 결과다.
        x = F.gelu(x)

        # 활성값 메모리 할당(world_size × batch_size × local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank)) for _ in range(world_size)]

        # all-gather로 활성값을 보낸다.
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # 이어 붙이면 batch_size × num_dim이 된다.
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)  # @stepover

    # 역전파는 연습문제다.

    cleanup()  # @stepover


def pipeline_parallelism():
    image("images/pipeline-parallelism.png", width=300)
    text("분할 전략: 각 랭크가 여러 층 중 일부를 맡으며 전체 데이터와 활성값을 전송한다.")

    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)  # @stepover

    # 전체 데이터를 사용한다.
    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # 층을 나눈다.
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers @stepover

    # 각 랭크는 일부 층을 맡는다.
    local_params = [get_init_params(num_dim, num_dim, rank) for layer in range(local_num_layers)]  # @stepover

    # 순전파

    # 파이프라인의 유휴 시간을 줄이기 위해 마이크로배치로 나눈다.
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size @stepover
    if rank == 0:
        # 입력 데이터
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # 활성값 메모리 할당
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=cuda_if_available(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # 이전 랭크에서 활성값을 받는다.
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # 이 랭크에 배정된 층을 계산한다.
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # 다음 랭크로 보낸다.
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)  # @stepover
            dist.send(tensor=x, dst=rank + 1)

    text("여기서는 통신과 연산을 겹쳐 파이프라인의 유휴 시간을 없애는 방법까지 다루지 않는다.")

    # 역전파는 연습문제다.

    cleanup()  # @stepover

############################################################

def setup(rank: int, world_size: int):
    "분산 실행 환경을 초기화한다. 프로세스 시작 시 호출한다."
    # 조정을 맡는 마스터(랭크 0)의 위치를 지정한다. 실제 데이터는 NCCL로 전송한다.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    "분산 실행 환경을 정리한다. 프로세스 종료 시 호출한다."
    torch.distributed.destroy_process_group()


class DisableDistributed:
    "분산 함수를 일시적으로 아무 연산도 하지 않는 함수로 교체하는 컨텍스트 관리자다. 강의 실행을 추적할 때는 다중 프로세스 내부를 추적할 수 없으므로 분산 통신 없이 함수를 직접 실행한다. "
    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                setattr(dist, name, lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])


def spawn(func: Callable, world_size: int, *args, **kwargs):
    "`world_size`개의 프로세스를 실행하고, 각 프로세스에서 world_size, args, kwargs로 `func`를 호출한다. edtrace로 추적할 때는 다중 프로세스 없이 함수를 직접 실행하고 분산 함수를 끈다. "
    # kwargs의 순서가 main에서 필요로 하는 순서와 같다고 가정한다.
    if not sys.gettrace():
        # 다중 프로세스를 사용하는 일반적인 실행 경로다.
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)
    else:
        # edtrace로 추적 중이라면 함수를 직접 실행한다.
        with DisableDistributed():  # @stepover
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    "매개변수를 생성해 `rank`번째 GPU에 둔다."
    torch.random.manual_seed(0)  # 결과를 재현하기 위한 설정
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=cuda_if_available(rank)) / math.sqrt(num_outputs))


def int_divide(a: int, b: int):
    "a / b의 몫을 반환한다. 나머지가 있으면 오류를 발생시킨다."
    assert a % b == 0
    return a // b


def summarize_tensor(tensor: tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"


if __name__ == "__main__":
    main()
