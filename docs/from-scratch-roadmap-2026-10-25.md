# From scratch 학습 경로 · 2026-10-25

## 선정 기준

README의 기존 대학 강의 목록에서 세 경로를 골랐다. 추천 순서는 이 저장소의 편집 판단이며 대학의 공식 연계 과정은 아니다. 강좌 전체가 저수준 구현만을 가르친다는 뜻으로 쓰지 않았다. 이론을 배운 뒤 공개 과제와 코드로 직접 구현하는 흐름이다. Python·선형대수·미분·확률을 공통 준비로 두고, LLM 과정의 PyTorch·신경망 선행 지식을 따로 적었다.

## 대학 강의 공식 출처

| 경로 | 공식 출처 | 확인한 내용 |
| --- | --- | --- |
| ML/DL | [서울대 ML/DL I](https://vip.snu.ac.kr/viplab/courses/mldl1_2026_1/index.html) | 2026 Spring. 회귀·경사하강법·정규화·신경망·역전파 및 Python 실습 |
| ML/DL | [MIT 6.S191](https://introtodeeplearning.com/) | 2026 강의·슬라이드·소프트웨어 실습. 신경망·생성 모델. FAQ에서 학습 자료 무료 공개 확인 |
| ML/DL | [Stanford CS231n](https://cs231n.github.io/) | 2026 과제: NumPy, 이미지 분류, fully connected networks, CNN, Transformer 등. 기존 목록의 공개 영상은 2025 Spring으로 구분 |
| LLM | [Stanford CS224N](https://web.stanford.edu/class/cs224n/) | 2026 Winter 자료·과제, Attention·Transformer·모델 평가. 공식 안내의 무료 공개 영상은 2024판이며 2026 녹화는 수강생용 |
| LLM | [Stanford CS336](https://cs336.stanford.edu/) | 2026 Spring. 토크나이저·모델·옵티마이저 구현, 학습, 시스템 최적화, 데이터, 평가·정렬 과제. GPU 비용은 별도임도 확인 |
| LLM | [CMU 11-711](https://cmu-l3.github.io/anlp-spring2026/) | 2026 Spring Advanced NLP. 생성 모델과 LLM의 모델링·학습·추론, 연구 프로젝트 |
| Agent | [Stanford CS221](https://stanford-cs221.github.io/autumn2025/) | 2025 Fall. 탐색·MDP·게임·확률 모델 등의 기반과 구현 과제 |
| Agent | [CMU 11-768](https://www.cmu-agents.com/) · [일정](https://www.cmu-agents.com/#/schedule) · [과제](https://www.cmu-agents.com/#/assignments) | 2026 Fall 진행 중. 브라우저에서 도구 사용·문맥·메모리·계획·코딩 에이전트의 슬라이드와 일부 녹화 링크 확인. Harness·Eval 과제 공개, Training 과제는 예고 단계 |
| Agent | [Stanford CS329A](https://cs329a.stanford.edu/) | 2025 Fall. 도구·코드 피드백, 다단계 계획, 검증기, 자기 개선과 평가 |

CMU 공개 과제는 [assignment-1](https://github.com/cmu-agents/assignment-1) README도 확인했다. 위 목록은 공개 강의와 과제의 구성을 확인한 기록이며 전체 영상 시청·모든 과제 실행·최종 학습 성과를 검증한 기록이 아니다.

## 크리에이터 보충 자료

- [Karpathy Zero to Hero](https://github.com/karpathy/nn-zero-to-hero): 작성자의 README에서 micrograd·makemore·GPT·BPE 강의와 노트북을 확인했다. 유튜브 표에는 한 행만 두었다.
- [Umar Jamil 채널](https://www.youtube.com/@umarjamilai/videos): 실제 동영상 목록 확인. [분산 학습 구현 영상](https://www.youtube.com/watch?v=XoGvCBRnwLs)은 확장한 설명에서 **2026-08-18** 게시일·19시간 34분 분량·DDP/FSDP 및 여러 병렬화 기법·[TorchFeather](https://github.com/hkproj/torchfeather) 코드 링크를 확인했다. 해당 코드 README도 읽었다.
- [Transformer 구현 영상](https://www.youtube.com/watch?v=ISNdQcPhsts)은 채널에서 3년 전 영상으로 표시됐으며, [작성자 코드 README](https://github.com/hkproj/pytorch-transformer)가 같은 영상을 연결한다. 최신 대학 강의를 대신하는 항목이 아니라 구현 보충 자료로 분리했다.
- Aladdin Persson의 과거 강의는 이번 추가 목록과 로드맵에서 제외했다.

## 읽고 화면도 확인한 참고 저장소

- [mlabonne/llm-course](https://github.com/mlabonne/llm-course): 실제 README와 Scientist 로드맵 이미지를 확인했다. 학습 목적별 경로를 먼저 나누는 방식만 참고했다.
- [ossu/computer-science](https://github.com/ossu/computer-science): 실제 README의 Curriculum 화면과 표를 확인했다. 선행 지식과 전체 목록을 구분하는 방식을 참고했다.
- [dair-ai/ML-YouTube-Courses](https://github.com/dair-ai/ML-YouTube-Courses): README를 읽고 주제별 강좌·구체적 학습 내용·영상 링크의 구성을 참고했다. 과거 강의 연도를 그대로 최신 강의로 옮기지 않았다.

그래픽은 이 저장소용 SVG로 새로 작성했다. 세 경로에 신경망·Attention 행렬·에이전트 루프를 각각 배치하고, 강좌 코드를 세로 순서로 정렬했다. 참고 저장소의 이미지·배너를 복제하지 않았다.

## 무료 범위

무료로 열람할 수 있는 강의·노트·공개 코드를 모은다. 일부 서비스는 무료 가입이 필요하다. API 호출·GPU/클라우드 사용·인증 시험·수료증·학점 과정의 비용은 이 문서의 무료 범위에 포함하지 않는다.
