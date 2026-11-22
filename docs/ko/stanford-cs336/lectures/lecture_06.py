import os
import time
from typing import Callable
import torch
from torch.profiler import ProfilerActivity
import triton
import triton.language as tl
from edtrace import text, link, image
from lecture_util import get_local_url
from gpu_util import cuda_if_available


def main():
    text("지난 강의: GPU의 구조와 성능 개요")
    text("이번 강의: 벤치마킹, 프로파일링, 커널 작성")

    review_of_gpus()
    benchmarking_and_profiling()           # 병목은 어디에 있는가?
    naive_vs_builtin_vs_compiled_gelu()    # GeLU 예제에 적용한다

    # Triton 커널 작성
    triton_introduction()
    triton_gelu_example()      # 원소별 연산
    triton_softmax_example()   # 축소 연산(행 하나가 블록에 들어가는 경우)
    triton_row_sum_example()   # 축소 연산(행 하나가 블록에 들어가지 않는 경우)
    triton_matmul_relu_example()    # 타일링: 공유 메모리 사용

    text("정리")
    text("- 정확한 구현을 위해 프로그래밍 모델(PyTorch, Triton, PTX)을 이해한다.")
    text("- 성능을 최적화하려면 하드웨어(SM, 워프, 점유율, 뱅크 충돌 등)를 이해해야 한다.")
    text("- 문제 크기에 따라 성능이 어떻게 달라지는지 벤치마크한다.")
    text("- 프로파일러로 어떤 연산이 얼마나 오래 실행되는지 확인한다.")
    text("- Triton은 스레드 블록 단위로 생각한다. 데이터를 공유 메모리로 읽고, 연산을 융합해 수행한 뒤 HBM에 쓴다.")
    text("- 예제: GeLU(원소별), 소프트맥스(행별), 행의 합(간단한 타일링), 행렬 곱셈(타일링).")

    text("다음 시간에는 GPU를 여러 대 사용한다!")


def review_of_gpus():
    text("## 하드웨어")
    image("images/gpu-hardware.png", width=800)
    text("| 가속기                              | A100      | H100      | B200      |", verbatim=True)
    text("+------------------------------------+-----------+-----------+-----------+", verbatim=True)
    text("| SM 수                               |       108 |       132 |       148 |", verbatim=True)
    text("+------------------------------------+-----------+-----------+-----------+", verbatim=True)
    text("| 레지스터 용량(SM당)                 |    256 KB |    256 KB |    256 KB |", verbatim=True)
    text("| L1 캐시 + 공유 메모리(SM당)         |    192 KB |    256 KB |    256 KB |", verbatim=True)
    text("| L2 캐시 용량                        |     40 MB |     50 MB | 96-126 MB |", verbatim=True)
    text("| HBM 용량                            |     80 GB |     80 GB |    192 GB |", verbatim=True)
    text("+------------------------------------+-----------+-----------+-----------+", verbatim=True)
    text("| 레지스터 대역폭                     | ~116 TB/s | ~401 TB/s | ~447 TB/s |", verbatim=True)
    text("| L1 캐시 + 공유 메모리 대역폭        |  ~19 TB/s |  ~33 TB/s |  ~40 TB/s |", verbatim=True)
    text("| L2 캐시 대역폭                      | ~5-8 TB/s |  ~12 TB/s |  ~17 TB/s |", verbatim=True)
    text("| HBM 대역폭                          |    2 TB/s | 3.35 TB/s |    8 TB/s |", verbatim=True)

    text("L1/L2 대역폭은 대략적인 추정치다. 온라인 자료마다 수치가 다르다.")
    text("B200에는 레지스터와 공유 메모리 사이에 텐서 코어용 텐서 메모리(TMEM)도 있다. 프로그래머에게 직접 보이지 않는다.")

    text("## 프로그래밍 모델")
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)
    text("- *스레드*: 데이터 일부에 대해 코드를 실행한다.")
    text("- *스레드 블록* 또는 CTA(concurrent thread array): 스레드 묶음.")
    text("- *그리드*: 스레드 블록의 집합.")

    text("H100과 B200에는 분산 공유 메모리를 사용할 수 있는 스레드 블록 클러스터도 있다.")

    text("스레드 블록이 필요한 이유는?")
    text("GeLU 같은 원소별 연산에는 스레드 하나가 원소 하나를 처리하는 방식이 자연스럽다.")
    text("- i = 0, ..., N-1 각각에 대해 f(i)를 계산한다.")
    text("소프트맥스나 행렬 곱셈처럼 여러 원소를 함께 처리하는 연산에서는 스레드 사이에 통신이 필요하다.")
    text("HBM 읽기·쓰기는 느리므로 각 SM의 공유 메모리를 쓴다.")
    text("스레드 블록은 같은 공유 메모리에 접근하는 스레드의 집합이다.")
    text("따라서 스레드 블록 하나는 SM 하나에 배정된다.")
    text("뒤에서 볼 Triton에서는 처음부터 스레드 블록 단위로 연산을 정의한다.")

    text("## 프로그래밍 모델과 하드웨어의 관계")
    text("프로그래밍 모델은 하드웨어를 추상화한다.")
    text("원칙적으로 연산 결과만 정확하면 될 때는 이 모델만 고려하면 된다.")
    text("하지만 성능은 하드웨어 특성에 민감하다. 높은 성능을 얻으려면 하드웨어도 알아야 한다.")

    text("몇 가지 사항을 살펴보자.")

    text("**워프**")
    text("- 블록 안의 스레드는 워프로 나뉜다. 워프 하나에는 스레드 32개가 있다.")
    text("- 예를 들어 스레드 64개인 블록에는 워프 2개가 있다.")
    text("| TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT | TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT |", verbatim=True)
    text("- 워프의 모든 스레드는 한 SM에서 같은 명령을 같은 단계에 실행한다.")
    text("- 제어 흐름 분기: 한 워프의 스레드가 서로 다른 명령(if A, else B)을 실행해야 하면 각 경로를 순서대로 실행해 성능이 떨어진다.")
    text("| AAAAAAAAA....................... |", verbatim=True)
    text("| .........BBBBBBBBBBBBBBBBBBBBBBB |", verbatim=True)
    text("- SM은 워프 여러 개를 실행하며, 한 워프가 HBM 읽기·쓰기를 기다릴 때 다른 워프로 전환할 수 있다. 전환 자체에는 비용이 들지 않는다.")

    text("**워프 점유율**")
    text("- 스레드 하나는 레지스터를 0개부터 255개까지 사용할 수 있다.")
    text("- 각 스레드가 쓰는 레지스터가 많을수록 한 SM에 배정할 수 있는 스레드가 줄어 점유율이 낮아진다.")
    text("- 스레드 하나가 더 많은 일을 처리한다면 낮은 점유율이 반드시 불리하지는 않다.")
    text("- 예: 스레드 하나에서 여러 원소를 처리하는 스레드 조대화(thread coarsening).")
    text("- 예: 블록당 스레드가 64개이고 각 스레드가 레지스터 160개를 사용하며 SM에는 레지스터 65,536개가 있다.")
    
    # 실행할 작업
    num_threads_per_block = 128
    num_registers_per_thread = 160

    # 하드웨어 자원
    max_registers = 65536  # SM당 사용할 수 있는 레지스터 수
    max_warps = 64         # SM당 동시에 실행할 수 있는 워프 수

    # 한 번에 실행할 수 있는 작업
    assert num_registers_per_thread <= 255
    num_registers_per_block = num_threads_per_block * num_registers_per_thread  # @inspect num_registers_per_block
    num_blocks = max_registers // num_registers_per_block  # 레지스터 수에 따른 제약 @inspect num_blocks
    num_warps = num_blocks * num_threads_per_block / 32  # @inspect num_warps
    occupancy = num_warps / max_warps  # @inspect occupancy

    text("**공유 메모리의 뱅크 충돌**")
    text("- 공유 메모리는 폭이 4바이트인 뱅크 32개로 나뉜다.")
    text("B00 B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16 B17 B18 B19 B20 B21 B22 B23 B24 B25 B26 B27 B28 B29 B30 B31", verbatim=True)
    text("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", verbatim=True)
    text("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", verbatim=True)
    text("... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...", verbatim=True)
    text("- 한 사이클에 한 뱅크의 서로 다른 위치에는 스레드 하나만 접근할 수 있다.")
    text("- 여러 스레드가 같은 뱅크의 서로 다른 위치에 접근하면 순서대로 처리한다(뱅크 충돌).")
    text("- 최악의 예: 행 하나가 뱅크 전체에 걸치는 행렬에서 스레드 32개가 첫 번째 열을 읽으면 32중 뱅크 충돌이 발생한다.")
    text("- A @ B를 계산할 때 A의 행과 B의 열을 읽어야 한다.")
    text("- 스위즐링(swizzling)으로 공유 메모리 배치를 바꿔 뱅크 충돌을 피한다. 예를 들어 행과 열 인덱스에 XOR를 적용한다.")

    text("**HBM 접근 병합**")
    text("- 한 워프의 스레드 32개가 HBM에 접근할 때 요청을 128바이트 트랜잭션(캐시 라인) 단위로 합친다.")
    text("M00 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19 M20 M21 M22 M23 M24 M25 M26 M27 M28 M29 M30 M31", verbatim=True)
    text("M32 M33 M34 M35 M36 M37 M38 M39 M40 M41 M42 M43 M44 M45 M46 M47 M48 M49 M50 M51 M52 M53 M54 M55 M56 M57 M58 M59 M60 M61 M62 M63", verbatim=True)
    text("- 최상의 경우는 모든 스레드가 같은 캐시 라인을 읽어 접근이 완전히 병합되는 것이다(32스레드 × 4바이트 = 128바이트).")

    text("**블록 점유율**")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/06/pasted-image-0.png", width=400)
    text("- 스레드 블록은 여러 차례(웨이브)에 걸쳐 SM에 배정된다.")
    text("- B200에는 SM이 148개다. 블록 160개를 실행하면 첫 번째 웨이브에서 148개, 두 번째 웨이브에서 12개를 실행한다.")
    text("- 웨이브 양자화 문제: 마지막 웨이브에는 블록이 적어서 일부 SM이 쉰다. 블록 점유율이 낮아진다.")
    text("- 해결 방법: 스레드 블록 수와 SM 수가 나누어떨어지도록 조정한다.")

    text("정리")
    text("- 프로그래밍 모델: 그리드(HBM) → 스레드 블록(공유 메모리) → 스레드(레지스터).")
    text("- 워프, 뱅크 충돌, 메모리 접근 병합, 점유율 등의 하드웨어 특성이 성능에 영향을 준다.")


def benchmarking_and_profiling():
    text("성능을 개선하는 절차")
    text("1. 코드의 성능을 측정하고 프로파일링한다.")
    text("2. 코드를 수정한다.")
    text("3. 다시 성능을 측정하고 프로파일링한다.")

    benchmarking()   # 얼마나 오래 걸리는가?
    profiling()      # 어디에서 시간을 쓰는가?

    text("코드를 벤치마크하고 프로파일링하자.")


def benchmarking():
    text("벤치마킹은 어떤 연산을 실행하는 데 걸리는 실제 경과 시간을 측정한다.")
    text("전체 소요 시간을 알려 준다. 어느 부분에 시간이 쓰이는지는 프로파일링으로 알아본다.")

    text("전체 실행 시간만으로도 다음을 알아볼 수 있다.")
    text("- 여러 구현을 비교해 어느 쪽이 빠른지 본다.")
    text("- 예를 들어 차원을 바꿔 가며 성능이 어떻게 달라지는지 본다.")

    text("[`torch.utils.benchmark`](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)를 사용할 수 있다.")
    text("여기서는 측정 방식을 직접 살펴보기 위해 벤치마크를 작성한다.")

    # 행렬 곱셈 벤치마크
    matmul = run_operation2(dim=1024, operation=lambda a, b: a @ b)
    result = benchmark(matmul)  # @inspect result

    # 차원에 따라 실행 시간이 어떻게 변하는지 확인한다
    results = {}
    for dim in [256, 512, 1024, 2048, 4096, 8192]:
         results[dim] = benchmark(run_operation2(dim=dim, operation=lambda a, b: a @ b))  # @inspect results @stepover

    text("차원이 작을 때는 시간이 거의 일정하다가, 이후에는 차원의 세제곱에 비례해 증가한다.")


def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    "`func`를 `num_trials`번 실행해 평균 시간을 반환한다."
    # 준비 실행: 컴파일 등의 영향으로 처음 몇 번은 더 느릴 수 있다.
    # 커널을 여러 번 실행하므로 안정된 상태에서 걸리는 시간을 측정한다.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # CUDA 스레드가 끝나기를 기다린다(중요).

    # 이제 실제 시간을 측정한다.
    times: list[float] = [] # @inspect times
    for trial in range(num_trials):  # 여러 번 실행해 변동을 반영한다.
        # CPU 오버헤드를 제외하고 GPU 시간을 정확하게 측정하기 위해 CUDA 이벤트를 쓴다.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # 측정 시작
        run()  # 실제 계산 실행
        end_event.record()  # 측정 끝

        torch.cuda.synchronize()  # CUDA 스레드가 끝날 때까지 기다린다.

        times.append((start_event.elapsed_time(end_event)))  # @inspect times

    mean_time = mean(times)   # @inspect mean_time @stepover
    return mean_time


def profiling():
    text("벤치마킹으로 전체 실행 시간을 측정했다면 프로파일링으로는 어디에서 시간을 쓰는지 살펴본다.")
    text("프로파일링은 시간 측정 외에도 내부에서 어떤 연산이 실행되는지 이해하는 데 도움이 된다.")

    text("PyTorch에는 [프로파일러](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)가 내장되어 있다.")
    text("과제에서는 더 자세한 정보를 얻기 위해 Nsight를 사용한다.")

    text("## add(dim=2048)")
    add_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a + b))
    text(add_profile, verbatim=True)

    text("## matmul(dim=2048)")
    matmul_profile = profile(run_operation2(dim=2048, operation=lambda a, b: a @ b)) # @stepover
    text(matmul_profile, verbatim=True)

    text("## matmul(dim=128)")
    matmul_profile = profile(run_operation2(dim=128, operation=lambda a, b: a @ b)) # @stepover
    text(matmul_profile, verbatim=True)

    text("관찰할 점")
    text("- 실제로 호출한 CUDA 커널을 볼 수 있다. 이름이 길게 표시된다.")
    text("- 텐서 차원에 따라 호출하는 CUDA 커널이 달라진다.")

    text("CUDA 커널 이름에는 구현 정보가 담겨 있다.")
    text("예: cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bi...")
    text("- cutlass: NVIDIA의 CUDA 선형대수 라이브러리")
    text("- sm100: NVIDIA Blackwell 아키텍처(B200)")
    text("- f32: float32")
    text("- 64x64x16: 타일 모양(뒤에서 다룬다).")


def profile(run: Callable, num_warmups: int = 1):
    # 준비 실행
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    # 프로파일러와 함께 코드를 실행한다.
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        torch.cuda.synchronize()

    # 표를 출력한다.
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=100,
                                      row_limit=10)

    # profiles.txt 끝에 추가한다.
    with open("var/profiles.txt", "a") as f:
        f.write(f"Profile at {time.ctime()}:\n")
        f.write(table)
        f.write("\n\n")
    return table


def naive_vs_builtin_vs_compiled_gelu():
    text("[GeLU 활성화 함수](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)를 벤치마크하고 프로파일링해 보자.")

    x = torch.tensor([1.])  # @inspect x

    # 1. PyTorch의 개별 연산으로 직접 구현한다(융합 없음).
    y1 = naive_gelu(x)  # @inspect y1

    # 2. PyTorch 내장 구현을 사용한다(융합).
    y2 = builtin_gelu(x)  # @inspect y2
    check_equal_1d(naive_gelu, builtin_gelu)  # 올바르게 동작하는지 확인한다.

    # 3. 직접 작성한 구현에 PyTorch 컴파일러를 적용한다.
    compiled_gelu = torch.compile(naive_gelu)  # @stepover
    y3 = compiled_gelu(x)  # @inspect y3 @stepover
    check_equal_1d(naive_gelu, compiled_gelu)  # 컴파일 전후의 연산 결과가 같은지 확인한다. @stepover

    # 벤치마킹
    naive_time = benchmark(run_operation1(dim=16384, operation=naive_gelu)) # @inspect naive_time @stepover
    builtin_time = benchmark(run_operation1(dim=16384, operation=builtin_gelu)) # @inspect builtin_time @stepover
    compiled_time = benchmark(run_operation1(dim=16384, operation=compiled_gelu)) # @inspect compiled_time @stepover
    text("내장 버전과 컴파일한 버전은 훨씬 빠르다.")

    text("이유를 파악하려면 프로파일러에서 어느 부분이 오래 걸리는지 보자.")

    text("## naive_gelu")
    naive_gelu_profile = profile(run_operation1(dim=16384, operation=naive_gelu))  # @stepover
    text(naive_gelu_profile, verbatim=True)

    text("## builtin_gelu")
    builtin_gelu_profile = profile(run_operation1(dim=16384, operation=builtin_gelu))  # @stepover
    text(builtin_gelu_profile, verbatim=True)

    text("## compiled_gelu")
    compiled_gelu_profile = profile(run_operation1(dim=16384, operation=compiled_gelu))  # @stepover
    text(compiled_gelu_profile, verbatim=True)

    text("관찰할 점")
    text("- 직접 작성한 구현은 커널 여러 개를 호출해 HBM 읽기·쓰기가 많다(**융합 없음**).")
    text("- 내장 구현과 컴파일한 구현은 **커널 융합**으로 한 커널에서 HBM을 한 번 읽고 한 번 쓴다.")
    text("- 컴파일해서 얻은 커널은 Triton 커널이다.")


def triton_introduction():
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)

    text("NVIDIA에서 개발한 CUDA에서는 각 스레드가 하는 일을 지정한다.")
    text("- 장점: 세밀하게 제어할 수 있다.")
    text("- 단점: 공유 메모리 등 직접 관리할 것이 많다.")

    text("OpenAI에서 개발한 Triton에서는 각 스레드 블록이 수행할 일을 지정한다.")
    text("- 일반적으로 충분히 표현력이 있으며 특히 처음 사용할 때 편하다.")
    text("- 데이터를 공유 메모리에 적재하고 연산한 뒤 전역 메모리에 다시 쓰는 방식으로 생각하면 된다.")


def triton_gelu_example():
    text("GeLU의 Triton 커널을 작성해 보자.")

    x = torch.randn(8192, device=cuda_if_available())
    y = triton_gelu(x)

    check_equal_1d(triton_gelu, naive_gelu)  # 계산 결과가 정확한지 확인한다. @stepover

    text("Triton은 GPU용 어셈블리 언어인 PTX(Parallel Thread Execution)로 컴파일된다.")

    text("Triton이 생성한 PTX 코드를 확인할 수 있다.")
    link(get_local_url("var/triton_gelu-ptx.txt"))

    text("관찰할 점")
    text("- ld.global.*과 st.global.*은 전역 메모리 읽기와 쓰기 명령이다.")
    text("- %ctaid.x는 블록 인덱스, %tid.x는 스레드 인덱스다.")
    text("- %f*는 부동소수점 레지스터, %r*는 정수 레지스터다.")
    text("- 한 스레드가 동시에 원소 8개를 처리한다(스레드 조대화).")


def triton_gelu(x: torch.Tensor):
    # 입력 확인
    assert x.is_cuda
    assert x.is_contiguous()

    # 출력 텐서 할당
    y = torch.empty_like(x)

    # 그리드를 정한다(원소를 블록으로 나눈다).
    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    블록 0       |    블록 1       |     블록 2       |    블록 3      |
    num_elements = x.numel()  # @inspect num_elements
    BLOCK_SIZE = 1024  # 스레드 수
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)  # @inspect num_blocks

    # 커널 실행
    kernel = triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    # 생성한 PTX를 출력한다(뒤에서 살펴본다).
    output_ptx("triton_gelu", kernel)  # @stepover

    return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # 입력의 시작 주소는 `x_ptr`이다.
    # 출력의 시작 주소는 `y_ptr`이다.

    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    블록 0       |    블록 1       |     블록 2       |    블록 3      |

    pid = tl.program_id(axis=0)      # 블록 식별자
    start = pid * BLOCK_SIZE         # 이 블록의 시작 인덱스

    # 이 스레드 블록이 처리할 인덱스
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # 텐서의 끝을 넘어 읽거나 쓰지 않는다.
    mask = offsets < num_elements

    # 읽기
    x = tl.load(x_ptr + offsets, mask=mask)

    # GeLU 근사식은 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))이다.
    # 계산한다. tl.tanh가 없으므로 tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)을 이용한다.
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # 저장
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_softmax_example():
    text("지금까지 Triton에서 GeLU 같은 원소별 연산을 살펴봤다.")
    text("이제 여러 값을 모아 계산하는 연산을 보자.")

    text("다음 Triton의 융합 소프트맥스 튜토리얼을 대략적으로 따른다. "), link("https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html")

    text("소프트맥스는 어텐션과 확률 계산에 사용한다.")
    text("행렬의 각 행에서 원소별로 지수를 취하고 정규화한다.")
    text("[0 0 0]      =>   [1/3 1/3 1/3]", verbatim=True)
    text("[1 1 -inf]        [1/2 1/2 0  ]", verbatim=True)

    text("먼저 직접 구현하면서 읽기와 쓰기 횟수를 세어 보자.")
    x = torch.tensor([
        [5., 5, 5],
        [0, 0, 100],
    ], device=cuda_if_available())
    y1 = naive_softmax(x) # @inspect y1

    text("이제 Triton 커널을 작성한다.")
    image("images/triton-softmax.png", width=600)
    y2 = triton_softmax(x)  # @inspect y2

    # 구현이 올바른지 확인한다.
    check_equal_2d(pytorch_softmax, naive_softmax) # @stepover
    check_equal_2d(pytorch_softmax, triton_softmax) # @stepover


def naive_softmax(x: torch.Tensor):
    # M은 행 수, N은 열 수다.
    M, N = x.shape

    # 행마다 최댓값을 구한다(읽기 MN회, 쓰기 M회).
    x_max = x.max(dim=1)[0]

    # 최댓값을 뺀다(읽기 MN + M회, 쓰기 MN회).
    x = x - x_max[:, None]

    # 지수를 취한다(읽기 MN회, 쓰기 MN회).
    numerator = torch.exp(x)

    # 정규화 상수를 구한다(읽기 MN회, 쓰기 M회).
    denominator = numerator.sum(dim=1)

    # 정규화한다(읽기 MN회, 쓰기 MN회).
    y = numerator / denominator[:, None]

    # 합계: 읽기 5MN + M회, 쓰기 3MN + 2M회.
    # 원칙적으로는 읽기 MN회, 쓰기 MN회면 된다(4배 가속).
    return y


def triton_softmax(x: torch.Tensor):
    # 출력 텐서 할당
    y = torch.empty_like(x)

    # 그리드 결정
    M, N = x.shape                          # 행 수 × 열 수
    block_size = triton.next_power_of_2(N)  # 블록 하나에 한 행의 모든 열이 들어간다.
    num_blocks = M                          # 블록 하나가 행 하나다.

    # 커널 실행
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y


@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # 각 행을 독립적으로 처리한다.
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 전역 메모리에서 읽는다.
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # 연산
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # 전역 메모리에 다시 쓴다.
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)


def triton_row_sum_example():
    text("소프트맥스 예제에서는 한 행이 한 블록에 들어가므로 축소 연산은 블록 안에서 이루어진다. 이 부분은 Triton이 처리한다.")
    text("행 하나가 블록에 들어가지 않으면 어떻게 할까?")
    text("예를 들어 열은 4,096개인데 블록 크기가 1,024라면?")

    text("계산 방법")
    text("- 행을 타일로 나눈다(위 예에서는 4개).")
    text("- 각 스레드는 타일을 순회하며 합을 누적한다.")
    text("- 각 스레드의 누적값을 공유 메모리나 워프 셔플로 합해 마지막 축소 연산을 수행한다.")

    text("소프트맥스 대신 행의 합을 구하는 더 간단한 예를 보자.")
    x = torch.tensor([[1., 2, 3, 4], [5, 6, 7, 8]], device=cuda_if_available())  # @inspect x
    y1 = builtin_row_sum(x)  # @inspect y1

    image("images/triton-row-sum.png", width=600)

    y2 = triton_row_sum(x)  # @inspect y2


def builtin_row_sum(x: torch.Tensor):
    return x.sum(dim=1)


def triton_row_sum(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # 어느 행을 처리하는가?

    # 각 스레드의 누산기
    # 한 행: T1 T2 T3 T4 | T1 T2 T3 T4 | T1 T2 T3 T4 (N = 12, BLOCK_SIZE = 4)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 타일을 순회한다.
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    # 스레드 전체의 값 BLOCK_SIZE개를 스칼라 하나로 합한다.
    result = tl.sum(acc, axis=0)

    tl.store(out_ptr + row, result)


def triton_matmul_relu_example():
    text("행렬 곱셈은 딥러닝에서 기본이 되는 연산이다.")
    a = torch.randn(1024, 1024, device=cuda_if_available())
    b = torch.randn(1024, 1024, device=cuda_if_available())
    c = naive_matmul_relu(a, b)

    text("행렬 곱셈 커널을 어떻게 구현할까?")

    text("|        k                  n                     ", verbatim=True)
    text("|   [ A1 A2 A3 ]       [ B1 B2 B3 ]   [ C1 C2 C3 ]", verbatim=True)
    text("| m [ A4 A5 A6 ]  *  k [ B4 B5 B6 ] = [ C4 C5 C6 ]", verbatim=True)
    text("|   [ A7 A8 A9 ]       [ B7 B8 B9 ]   [ C7 C8 C9 ]", verbatim=True)

    text("**단순한 방법**")
    text("임의의 (m, n)을 고정한다.")
    text("각 k에 대해 다음을 반복한다.")
    text("- HBM에서 A[m, k]와 B[k, n]을 읽는다.")
    text("- 두 값을 곱해 누적한다.")
    text("결과를 HBM의 C[m, n]에 쓴다.")

    text("병목: 읽기 M K N회, 쓰기 M N회.")
    text("산술 집약도: O(1).")

    text("C4와 C5를 구할 때 모두 A4, A5, A6이 필요하다.")
    text("HBM에서 A4, A5, A6을 한 번만 읽고 두 값을 모두 계산할 수 있을까?")
    text("공유 메모리를 사용하면 가능하다.")

    text("**이상적인 방법**")
    text("- A와 B 전체를 공유 메모리에 적재한 다음 C를 계산한다.")
    text("- 읽기 M K + K N회, 쓰기 M N회면 된다.")
    text("- 앞서 본 이상적인 산술 집약도 O(N)을 얻는다.")
    text("- 하지만 일반적으로 A와 B는 너무 커서 공유 메모리에 들어가지 않는다.")

    text("**타일링**")

    image("images/gemm_tiled.png", width=600)
    text("행렬 C를 출력 타일(스레드 블록)로 나눈다.")
    text("C의 출력 타일 하나를 고정한다.")
    text("A의 행 타일과 B의 열 타일로 이루어진 각 쌍에 대해 다음을 반복한다.")
    text("- 해당하는 A와 B의 타일을 HBM에서 공유 메모리로 읽는다.")
    text("- 두 타일을 행렬 곱셈한다.")
    text("- 결과를 공유 메모리의 부분합에 누적한다.")
    text("출력 타일을 HBM에 쓴다.")

    text("산술 집약도: O(tile_size).")

    text("추가로 고려할 연산")
    text("- 행렬 곱셈 뒤에 원소별 활성화 함수를 적용하는 경우가 많다.")
    text("- 예: GeLU(A @ B).")
    text("- 커널 융합으로 처리한다.")

    text("**구현**")

    text("복습: 각 행렬은 메모리에 일렬로 저장된다.")
    x = torch.tensor([[0., 1, 2, 3], [4, 5, 6, 7]])  # @inspect x
    stride_row, stride_col = x.stride()  # @inspect stride_row stride_col
    row = 1
    col = 2
    index = row * stride_row + col * stride_col  # @inspect index

    # c = a @ b 계산
    c = triton_matmul_relu(a, b)


def naive_matmul_relu(x: torch.Tensor, y: torch.Tensor):
    # 행렬 곱셈 후 ReLU 적용
    return torch.nn.functional.relu(x @ y)


def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    # A는 M × K, B는 K × N 행렬이다.
    M, K = a.shape
    K, N = b.shape

    # 출력 텐서 할당
    c = torch.empty((M, N), device=a.device)

    # 그리드 결정
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return c


@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,    # c = a @ b 계산
    M, N, K,                # a는 M × K, b는 K × N, c는 M × N이다.
    stride_am, stride_ak,   # a의 원소에 접근하는 방법
    stride_bk, stride_bn,   # b의 원소에 접근하는 방법
    stride_cm, stride_cn,   # c의 원소에 접근하는 방법
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # (m, n)번째 타일을 처리한다.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 인덱스
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # a의 행 인덱스 [BLOCK_M]
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # b의 열 인덱스 [BLOCK_N]
    indices_k = tl.arange(0, BLOCK_K)                    # a의 행 인덱스 = b의 열 인덱스 [BLOCK_K]

    # a와 b의 포인터로 만든 초기 행렬
    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak  # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # a의 행 타일과 b의 열 타일을 따라 이동한다.
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak  # a의 다음 행 타일로 이동한다.
        b_ptrs += BLOCK_K * stride_bk  # b의 다음 열 타일로 이동한다.

    # 활성화 함수(예: ReLU)를 적용한다.
    acc = tl.maximum(acc, 0.0)

    # 출력 타일을 쓴다.
    c_ptrs = c_ptr + indices_m[:, None] * stride_cm + indices_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(indices_m[:, None] < M) & (indices_n[None, :] < N))


############################################################

def run_operation1(dim: int, operation: Callable) -> Callable:
    # 준비: dim × dim인 임의의 행렬 하나를 생성한다.
    x = torch.randn(dim, dim, device=cuda_if_available())
    # 해당 연산을 수행할 함수를 반환한다.
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # 준비: dim × dim인 임의의 행렬 두 개를 생성한다.
    x = torch.randn(dim, dim, device=cuda_if_available())
    y = torch.randn(dim, dim, device=cuda_if_available())
    # 해당 연산을 수행할 함수를 반환한다.
    return lambda : operation(x, y)


def naive_gelu(x: torch.Tensor):
    # tanh를 이용한 GeLU 활성화 함수의 근사
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


def builtin_gelu(x: torch.Tensor):
    # tanh 근사를 적용한 PyTorch 내장 GeLU
    return torch.nn.functional.gelu(x, approximate="tanh")


def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


def check_equal_1d(f1, f2):
    x = torch.randn(2048, device=cuda_if_available())
    y1 = f1(x)  # @stepover
    y2 = f2(x)  # @stepover
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal_2d(f1, f2):
    x = torch.randn(2048, 2048, device=cuda_if_available())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal_2d_2d(f1, f2):
    x1 = torch.randn(2048, 2048, device=cuda_if_available())
    x2 = torch.randn(2048, 2048, device=cuda_if_available())
    y1 = f1(x1, x2)
    y2 = f2(x1, x2)
    assert torch.allclose(y1, y2, atol=1e-6)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)

    
def output_ptx(name: str, kernel):
    "주어진 `kernel`에 대해 Triton이 생성한 PTX 코드를 출력한다."
    ptx_path = f"var/{name}-ptx.txt"
    with open(ptx_path, "w") as f:
        ptx = kernel.asm["ptx"]
        f.write(ptx)


if __name__ == "__main__":
    main()
