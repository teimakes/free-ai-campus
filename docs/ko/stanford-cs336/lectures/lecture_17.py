from edtrace import text, image, link
from lecture_util import article_link, post_link

def main():
    text("## 17강. 멀티모달 모델")
    text("지금까지 다룬 대상: 언어 모델")
    text("> 텍스트 ⇒ 텍스트")
    text("현실의 정보는 여러 양식으로 표현된다.")
    image("images/multimodality.png", width=600)

    text("최종 목표: **옴니 모델**")
    text("- 어떤 조합의 양식이든 입력받아 이해한다.")
    text("- 어떤 조합의 양식이든 출력으로 생성한다.")

    text("현재의 접근법")
    text("- Transformer는 성능이 좋아 여기서도 사용한다.")
    text("- Transformer는 이산 또는 연속 토큰을 처리한다. 토큰은 대략적으로 의미를 갖는 정보 단위다.")
    text("- 따라서 모든 양식의 데이터를 토큰으로 변환해야 한다.")
    text("- 토큰화 강의에서 본 것처럼 텍스트에도 같은 작업이 필요했다.")
    text("- 텍스트 이외의 양식은 더 어렵다.")

    text("질문")
    text("1. 텍스트 이외의 데이터를 어떻게 입력할까? 예를 들어 이미지를 어떻게 이해할까?")
    text("2. 텍스트 이외의 데이터를 어떻게 출력할까? 예를 들어 음성을 어떻게 생성할까?")

    # 이미지 인코딩
    clip()
    siglip()

    # 언어 모델에 이미지 인코딩 전달하기
    llava()
    llava_onevision()
    qwen_vl()
    qwen2_vl()
    qwen3_vl()

    # 옴니 모델로 확장하기
    chameleon()

    text("정리")
    text("- 최첨단 모델에는 여러 양식을 처음부터 통합해 다루는 능력이 기대된다.")
    text("- 텍스트 이외의 양식을 어떻게 인코딩할지가 기본 문제다.")
    text("- 이해에는 의미 정보가, 생성에는 더 세밀한 정보가 필요할 수 있다.")
    text("- 안정적으로 학습하려면 정보 밀도가 낮은 이미지·비디오와 텍스트의 비중을 조절해야 한다.")
    text("- 생성에 연속 인코더, Transformer, 확산 모델을 함께 사용할 수 있다.")


def clip():
    text("CLIP(대조적 언어·이미지 사전학습) "), link("https://arxiv.org/abs/2103.00020")

    text("배경")
    text("- 컴퓨터 비전 모델은 사람이 주석을 단 이미지로 학습했다.")
    text("- 훨씬 많은 이미지·캡션 쌍을 활용할 수 있을까?")
    image("images/clip.png", width=800)

    text("방법")
    text("- 이미지와 텍스트가 짝지어진 예제를 한 배치 가져온다(예: 32,768개).")
    text("- 각 이미지와 각 텍스트를 인코딩한다.")
    text("- 각 이미지에 대해서는 다른 텍스트보다 원래 짝인 텍스트에 높은 점수를 주도록 한다.")
    text("- 각 텍스트에 대해서는 다른 이미지보다 원래 짝인 이미지에 높은 점수를 주도록 한다.")
    image("images/clip-code.png", width=400)

    text("데이터")
    text("- 검색어 50만 개로 검색해 검색어당 이미지·텍스트 쌍을 약 2만 개 수집했다.")
    text("- 이미지·텍스트 쌍 4억 개로 학습했다.")
    text("- 데이터셋은 공개하지 않았다.")
    text("- OpenCLIP이 LAION-5B를 사용해 재현했다. LAION-5B는 자료를 선별할 때 CLIP을 사용했다. "), link("https://arxiv.org/abs/2212.07143")

    text("데이터 처리 "), link(title="코드", url="https://github.com/openai/CLIP/blob/main/clip/clip.py#L79")
    text("- 입력 이미지는 가로 W, 세로 H의 임의 해상도다.")
    text("- 짧은 변을 336픽셀로 맞추도록 바이큐빅 보간으로 크기를 조정한다.")
    text("- 가운데를 기준으로 가장자리를 잘라 336 x 336 이미지를 얻는다.")
    
    text("시각 인코더")
    text("- ResNet-50과 Vision Transformer를 실험했다. "), link("https://arxiv.org/pdf/2010.11929")
    image("images/vit.png", width=600)
    text("- 어텐션 풀링은 활성값 전체의 평균을 쿼리로 사용해 QKV 어텐션을 계산한다.")
    text("- 가장 좋은 모델은 ViT-L/14@336px다. L은 대형 모델, 패치는 14 x 14, 채널은 3개이며 학습 이미지 해상도는 336 x 336이다.")

    text("텍스트 인코더")
    text("- GPT-2 Transformer(매개변수 6,300만 개, 12개 층).")
    text("- [BOS] ... [EOS]를 인코딩한 뒤 최상위 층의 [EOS] 위치 활성값을 반환한다.")

    text("주요 결과")
    text("- ImageNet에서 제로샷 CLIP이 ImageNet 이미지 120만 장으로 학습한 ResNet-50보다 높은 성능을 냈다.")

    text("비교 실험")
    text("- 다른 방법: 이미지에서 텍스트를 직접 예측한다.")
    text("- CLIP처럼 짝의 순위를 학습하는 방법보다 연산 효율이 훨씬 낮았다.")
    image("images/clip-efficiency.png", width=400)

    text("정리")
    text("- 이미지의 인코딩은 잡음이 섞인 텍스트가 제공하는 의미 정보를 담는다.")
    text("- 구조는 이미지 분류 성능을 기준으로 선택했으므로 매우 세밀한 정보까지 따지지는 않았다.")
    text("- 기술적으로 큰 배치와 전체 배치에 걸친 소프트맥스가 필요하다.")


def siglip():
    text("SigLIP(언어·이미지 사전학습을 위한 시그모이드 손실) "), link("https://arxiv.org/abs/2303.15343")

    text("목적함수")
    text("- CLIP: (텍스트, 이미지) 쌍을 가능한 다른 모든 (텍스트, 이미지') 쌍과 비교하는 다중 클래스 분류.")
    text("- SigLIP: (텍스트, 이미지)가 실제로 짝인지 판별하는 이진 분류.")
    image("images/siglip-code.png", width=500)

    text("데이터")
    text("- WebLI에는 수십억 규모의 이미지·텍스트 쌍이 있다. "), link("https://arxiv.org/pdf/2209.06794")
    text("- 인터넷에서 수집했다.")
    text("- OCR로 이미지 속의 텍스트를 자동 추출했다.")
    text("- 품질 점수 상위 10%를 남겼다.")
    text("- 100개 언어를 지원한다.")

    text("효율")
    text("- CLIP: TPUv3 256개로 10일.")
    text("- SigLIP: TPUv3보다 FLOP/s가 낮은 TPUv4 32개로 5일. 훨씬 빨랐다.")
    image("images/siglip-parallelism.png", width=800)

    text("배치 크기")
    text("- 배치 크기와 손실 계산을 분리한다.")
    text("- 배치 크기 16K 미만에서 CLIP보다 좋은 성능을 냈다.")
    text("- 최대 1M까지 시험했지만 32K면 충분했다.")


def llava():
    text("LLaVA(대규모 언어·시각 비서) "), link("https://arxiv.org/abs/2304.08485")

    text("시각 인코더: CLIP")
    text("텍스트 디코더: Vicuna. ShareGPT 대화로 LLaMA를 미세조정한 모델이다. "), post_link("https://www.lmsys.org/blog/2023-03-30-vicuna/")

    text("데이터")
    text("- MS COCO에는 경계 상자와 Mechanical Turk 작업자가 작성한 캡션이 달린 이미지가 있다.")
    text("- GPT-4에 캡션이나 검출된 물체를 주고 질문 또는 대화를 생성한다.")
    text("- 생성한 결과를 원본 이미지와 짝짓는다.")
    text("- 예제 15만 8,000개다.")
    image("images/llava-gen.png", width=600)

    text("모델")
    text("- CLIP(ViT-L/14)으로 이미지를 인코딩한다.")
    text("- 선형 투영 W로 언어 모델의 임베딩 공간에 옮긴다. Flamingo와 Q-former는 더 복잡하다.")
    image("images/llava-architecture.png", width=600)

    text("학습")
    text("- 1단계(정렬): 시각 인코더와 언어 모델은 고정하고 W만 학습한다.")
    text("- 2단계(미세조정): 시각 인코더를 고정하고 W와 언어 모델을 학습한다.")
    image("images/llava-example.png", width=600)


def llava_onevision():
    text("LLaVA OneVision "), link("https://arxiv.org/pdf/2408.03326")
    text("- LLaVA 1.5와 LLaVA-NeXT에 이은 LLaVA 계열의 최신 버전이다.")
    text("- 여러 이미지와 비디오를 처리한다.")

    image("images/llava-onevision.png", width=600)
    text("- 시각 인코더: SigLIP. 마지막 Transformer 층 전후의 격자 특징을 사용한다.")
    text("- 텍스트 디코더: Qwen-2 72B.")
    text("- 투영기: 2층 MLP.")

    text("데이터 처리")
    text("- OCR 등에서는 높은 해상도를 보존하는 것이 중요하다.")
    text("- CLIP은 336 x 336으로 크기를 바꾸고 잘라내므로 정보가 사라진다.")
    text("- 해결 방법: LLaVA 1.5에서 도입한 AnyRes. "), link(title="논문", url="https://static.hliu.cc/files/llava/improved_llava.pdf")
    text("- 이미지를 시각 인코더의 해상도에 맞게 가로 a개, 세로 b개로 나눠 인코딩한 뒤 이어 붙인다.")
    text("- 원본 해상도가 높아 토큰이 너무 많아지면 쌍선형 보간을 사용한다.")
    image("images/llava-onevision-anyres.png", width=600)
    text("단일 이미지, 복수 이미지, 비디오의 세 입력 유형")
    text("- 서로 다른 양식에서 얻는 시퀀스 길이를 대략 비슷하게 맞춘다.")
    image("images/llava-onevision-modalities.png", width=600)
    text("- 단일 이미지에는 높은 해상도를 사용한다.")
    text("- 복수 이미지에서는 각각 기본 해상도를 사용한다.")
    text("- 비디오에서는 프레임마다 더 낮은 해상도를 사용한다.")

    text("데이터")
    text("- 양보다 품질을 우선했다.")
    image("images/llava-onevision-data-1.png", width=700)
    image("images/llava-onevision-data-2.png", width=700)

    text("학습")
    text("- 쉬운 것부터 어려운 순서로 학습한다.")
    image("images/llava-onevision-training.png", width=700)

    text("양식 간 전이")
    text("- 단일 이미지에 담긴 도식과 차트로 학습해 여러 이미지로 일반화한다.")
    image("images/llava-onevision-transfer-s1.png", width=600)
    text("- 단일 이미지의 OCR과 복수 이미지에서의 관계 추론을 학습해 그래픽 인터페이스를 조작하는 에이전트에 적용한다.")
    image("images/llava-onevision-transfer-s2.png", width=600)
    text("- 단일 이미지에 원을 그려 대상을 지정하는 시각 프롬프팅이 비디오로도 일반화된다.")
    image("images/llava-onevision-transfer-s8.png", width=600)

    text("정리")
    text("- 일반적인 시각 언어 모델 구조: 시각 인코더 + 투영기 + 언어 모델.")
    text("- 작업 대부분이 데이터 선별에 들어가며, 과제에 맞게 합성한 데이터를 많이 쓴다.")
    text("- 모델 가중치와 데이터를 공개했다.")


def qwen_vl():
    text("Qwen-VL "), link("https://arxiv.org/abs/2308.12966")

    text("구조")
    text("- 시각 인코더: OpenCLIP의 ViT-bigC(14 x 14 패치). "), link("https://arxiv.org/abs/2212.07143")
    text("- 어댑터: 2차원 위치 인코딩을 포함하는 한 층의 교차 어텐션으로 길이를 256으로 고정한다.")
    text("- 특수 토큰: <img>, <box>, <ref>")

    text("학습")
    image("images/qwen-vl-stages.png", width=700)
    text("- 1단계: 대규모 저품질 데이터로 학습한다. 언어 모델을 고정하고 시각 인코더와 어댑터를 학습한다.")
    image("images/qwen-vl-stage1.png", width=400)
    text("- 2단계: 품질이 높은 과제별 데이터로 해상도를 높여 모든 매개변수를 학습한다.")
    image("images/qwen-vl-stage2.png", width=400)
    text("- 3단계: 지시 학습 데이터로 시각 인코더를 고정하고 어댑터와 언어 모델을 학습한다.")

    image("images/qwen-vl-examples.png", width=600)


def qwen2_vl():
    text("Qwen2-VL "), link("https://arxiv.org/abs/2409.12191")

    text("시각 인코더: 더 큰 ViT(매개변수 6억 7,500만 개).")
    image("images/qwen2-vl-architecture.png", width=700)
    text("- 동적 해상도로 서로 다른 입력 해상도를 처리한다.")
    text("- 224 x 224 패치를 ViT/14로 인코딩하고 2 x 2마다 압축해 토큰 66개를 얻는다.")
    text("- 비디오는 초당 2프레임을 샘플링하고 최대 16,384토큰을 사용한다.")

    text("멀티모달 회전 위치 임베딩(MRoPE)")
    image("images/qwen2-vl-mrope.png", width=600)
    
    text("언어 모델은 Qwen2로, 시각 인코더는 DFN으로 초기화한다. "), link("https://arxiv.org/abs/2309.17425")
    text("학습(Qwen-VL과 유사)")
    text("- 1단계: 시각 인코더만 학습한다.")
    text("- 2단계: 모든 매개변수를 학습한다.")
    text("- 3단계: 지시 수행 데이터셋으로 언어 모델을 학습한다.")

    text("다양한 기능")
    image("images/qwen2-vl-capabilities.png", width=700)


def qwen3_vl():
    text("Qwen3-VL "), link("https://arxiv.org/abs/2511.21631")
    image("images/qwen3-vl.png", width=700)

    text("언어 모델")
    text("- Qwen-3 계열 모델이다. 밀집 모델과 최대 235B-A22B의 MoE 모델이 있다.")
    text("- 긴 문맥(256K)을 이해한다.")

    text("시각 인코더")
    text("- SigLIP-2. SigLIP과 구조가 같다. "), link("https://arxiv.org/pdf/2502.14786")
    text("- 교차 배치 MRoPE: 시간·너비·높이 축을 저주파와 고주파 대역 전체에 분산한다.")
    text("... [t t t t w w w w h h h h] 대신 [t w h t w h t w h t w h] 순서다.")
    text("- 위치 임베딩 대신 별도 토큰으로 비디오 시각 정보를 명시한다.")
    text("- 토큰별 손실을 제곱근으로 정규화해 텍스트와 멀티모달 데이터의 비중을 맞춘다. 긴 비디오 예제가 손실을 지배하지 않게 한다.")

    text("어댑터")
    text("- DeepStack: 층 사이의 특징을 결합해 여러 층에 시각 정보를 넣는다. "), link("https://arxiv.org/abs/2406.04334")

    text("학습")
    text("- 사전학습은 4단계다. 어댑터를 학습한 뒤 길이 8K, 32K, 256K에서 모든 매개변수를 학습한다.")
    image("images/qwen3-vl-pretraining.png", width=600)
    text("- 후학습: 긴 사고 과정 데이터로 SFT한 뒤 지식 증류와 강화학습을 한다.")

    image("images/qwen3-vl-results.png", width=600)

    text("정리")
    text("- 최고 수준의 성능.")
    text("- 데이터 작업이 많았지만 공개된 세부 사항은 적다.")
    text("- 구조 변경은 작아 보여도 효과는 클 수 있다.")
    text("- 모델과 학습 규모를 키웠다.")


def chameleon():
    text("Chameleon "), link("https://arxiv.org/pdf/2405.09818")

    text("지금까지의 시각 언어 모델은 CLIP이나 SigLIP으로 인코딩한 이미지를 언어 모델에 넣었다.")
    text("단점: 이미지를 생성할 수 없어서 확산 모델이 따로 필요하다.")

    text("Chameleon은 모든 것을 이산 토큰으로 변환한다.")
    text("장점: 같은 방식으로 이미지를 분석하고 생성할 수 있다.")
    image("images/chameleon.png", width=600)
    image("images/chameleon-example.png", width=600)

    text("시각 인코더 "), link("https://arxiv.org/pdf/2203.13131")
    text("- 생성할 수 있으려면 인코더가 입력을 이산 토큰으로 변환해야 한다.")
    text("- VQ-VAE(벡터 양자화 변분 오토인코더). "), link("https://arxiv.org/pdf/1711.00937")
    text("- 이미지를 이산 코드북에 대응시키고 다시 이미지로 디코딩해 재구성 손실을 최소화한다.")
    image("images/vq-vae.png", width=600)
    text("- 512 x 512 이미지를 토큰 1,024개로 인코딩한다. 코드북 크기는 8,192다.")
    text("- 새 BPE 토크나이저를 학습한다.")

    text("학습")
    text("- 1단계(80%): 대규모 비지도 학습. 텍스트 토큰 2조 9,000억 개, 텍스트·이미지 토큰 1조 5,000억 개, 텍스트와 이미지가 교차하는 토큰 4,000억 개다.")
    text("- 2단계(20%): 1단계 데이터 50%, 고품질 데이터 50%.")
    
    text("학습 안정성")
    text("- 텍스트 토큰은 엔트로피가 낮고 이미지 토큰은 높다. 이 차이로 노름이 커지고 로짓이 변하는 문제가 생긴다.")
    text("- QK 정규화와 z-loss 정규화를 사용한다.")

    text("정리")
    text("- 이산 토큰의 자기회귀 모델링만 사용해 구성이 단순하다.")
    text("- 성능은 상대적으로 낮다. 예를 들어 OCR처럼 세밀한 정보가 필요할 때 이산화 과정에서 정보가 사라진다.")
    text("- 여러 양식을 함께 학습하는 데는 어려움이 있다.")
    

if __name__ == "__main__":
    main()
