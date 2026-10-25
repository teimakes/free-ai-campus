# ML / DL University Lectures

국내외 대학의 머신러닝·딥러닝·자연어처리·컴퓨터 비전 공개 강의와 교수·연구실 채널을 모읍니다. 강의를 바로 시작할 수 있도록 **공개 영상의 학기**, **공식 자료의 학기**, **주요 주제**를 함께 정리합니다.

A curated collection of university lectures and channels on machine learning, deep learning, NLP, computer vision, and reinforcement learning.

**공개 강의 표 확인일: 2026-10-25** · [확인 근거와 범위](docs/verification-2026-10-25.md) · [강의 제안하기](https://github.com/teimakes/ml-dl-university/issues/new?template=lecture.yml)

## Table of Contents

- [공개 강의 업데이트](#공개-강의-업데이트)
- [학습 경로 예시](#학습-경로-예시)
- [ML/DL-Korea](#mldl-korea)
- [ML/DL-Overseas](#mldl-overseas)
- [NLP-Korea](#nlp-korea)
- [NLP-Overseas](#nlp-overseas)
- [Vision-Korea](#vision-korea)
- [Vision-Overseas](#vision-overseas)
- [Reinforcement Learning](#reinforcement-learning)
- [Seminar](#seminar)
- [Contributing](#contributing)

## 공개 강의 업데이트

대학·강사 공식 페이지 또는 공식 영상에서 강좌와 공개 영상판을 확인한 목록입니다. **영상판과 자료판이 다른 경우 각각의 연도를 표시했습니다.** `1강`은 첫 강의 링크이고, `재생목록`은 여러 강의를 모은 링크입니다. 아래 강의는 모두 영어로 진행됩니다.

| 강좌 | 공개 영상판·링크 | 공식 자료·강좌 안내 | 주요 주제 |
| --- | --- | --- | --- |
| Stanford CS229 — Machine Learning | 2026 Spring · [1강](https://www.youtube.com/watch?v=DATnpGoGhM8) | [Stanford Online의 2강·강좌 소개](https://www.youtube.com/watch?v=cmNIMjPYdgM) | 머신러닝 기초, 지도학습 |
| MIT 6.S191 — Introduction to Deep Learning | 2026 · [재생목록](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) | [2026 강좌·슬라이드·실습](https://introtodeeplearning.com/) | 딥러닝 입문, 생성 모델, LLM, 강화학습 |
| NYU — Introduction to Deep Learning Research / Alfredo Canziani | 2025 Fall · [1강](https://www.youtube.com/watch?v=rg4QyMFONNQ) | [2025 강좌·강의별 영상·코드](https://atcold.github.io/NYU-DLFL25U/) | 신경망의 수학적 이해와 구현 |
| KAIST AI605 — Deep Learning for NLP / 서민준 | 2022 Spring · [재생목록](https://www.youtube.com/playlist?list=PLN9tAT2r9TDwlfX18uHYVigflooo8DMgq) | [2022 강좌·과제](https://seominjoon.github.io/kaist-ai605/) | Transformer, 언어 모델, 사전학습, NLP 실습 |
| Stanford CS224N — Natural Language Processing with Deep Learning | 2024 Spring · [재생목록](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) | [2026 Winter 강좌·슬라이드·과제](https://web.stanford.edu/class/cs224n/) | 신경망 기반 자연어처리, 언어 모델 |
| Stanford CS336 — Language Modeling from Scratch / Percy Liang·Tatsunori Hashimoto | 2026 Spring · [공식 강좌가 연결한 재생목록](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV) | [2026 강좌·코드·과제](https://cs336.stanford.edu/) | 토크나이저, Transformer 구현, GPU 최적화, 데이터, 평가, 후학습 |
| CMU 11-711 — Advanced NLP / Sean Welleck | 2026 Spring · [1강](https://www.youtube.com/watch?v=dnnBCi-qd0Q) | [2026 강좌·슬라이드·코드](https://cmu-l3.github.io/anlp-spring2026/) | LLM의 모델링·학습·추론, 평가, 강화학습·에이전트 |
| Stanford CS231n — Deep Learning for Computer Vision | 2025 Spring · [1강](https://www.youtube.com/watch?v=2fq9wYslV0A) | [공식 노트·2026 과제](https://cs231n.github.io/) | 신경망, CNN, 컴퓨터 비전, 실습 |
| Stanford CS224R — Deep Reinforcement Learning / Chelsea Finn | 2025 Spring · [재생목록](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL) | [2026 Spring 강좌·자료](https://cs224r.stanford.edu/) | 모방학습, 딥 강화학습, 로보틱스·언어 모델 응용 |
| UC Berkeley CS285 — Deep Reinforcement Learning / Sergey Levine | 2023 Fall · [재생목록](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps) | [2026 Spring 강좌·자료](https://rail.eecs.berkeley.edu/deeprlcourse/) | 정책 경사, actor-critic, 모델 기반·오프라인 강화학습 |

CS224N은 공식 사이트가 2026 자료와 2024 공개 영상을 안내합니다. CS224R과 CS285도 현재 자료의 학기와 공식 사이트가 연결하는 공개 영상의 학기가 다릅니다. 서로 다른 학기의 과제·영상은 내용과 순서가 다를 수 있습니다.

## 학습 경로 예시

아래 순서는 목록을 활용하기 위한 편집 제안입니다. 선수지식과 과제 요구사항은 각 강좌의 공식 안내를 확인하세요.

- **ML/DL 기초:** MIT 6.S191로 전체 흐름을 살펴보고 CS229로 머신러닝 기초를 학습합니다.
- **NLP·LLM 구현:** KAIST AI605 또는 CS224N을 학습한 뒤, CS336으로 언어 모델 구현과 학습 과정을 다룹니다.
- **컴퓨터 비전:** CS231n의 공개 영상과 공식 노트·실습 자료를 함께 활용합니다.
- **강화학습:** CS224R 또는 CS285의 공개 영상을 선택하고 해당 학기의 자료를 함께 확인합니다.

## ML/DL-Korea

기존 목록의 교수·연구실 채널입니다. 채널에는 여러 학기의 영상과 세미나가 함께 있으므로, 개별 강의의 연도는 영상 제목·설명에서 확인하세요.

- [서울대학교 윤성로](https://www.youtube.com/channel/UCEkAV8kTc87PogbVLq-nd6A)
- [KAIST 문일철 / AAILab](https://www.youtube.com/channel/UC9caTTXVw19PtY07es58NDg)
- [고려대학교 김성범](https://www.youtube.com/channel/UCueLU1pCvFlM8Y8sth7a6RQ/)
- [고려대학교 김중헌](https://www.youtube.com/user/joongheon/)

## ML/DL-Overseas

- [NYU Yann LeCun & Alfredo Canziani — 기존 등록 재생목록](https://www.youtube.com/playlist?list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq)
- MIT 6.S191, NYU 2025, Stanford CS229의 영상·자료는 [공개 강의 업데이트](#공개-강의-업데이트)를 참고하세요.

## NLP-Korea

- [KAIST 서민준](https://www.youtube.com/channel/UCriDmMe_Ieb8hnox1CBiFRA) · [AI605 2022 공개 강좌](https://seominjoon.github.io/kaist-ai605/) — 영어 강의
- [서울대학교 DSBA 연구실 / 강필성](https://www.youtube.com/channel/UCPq01cgCcEwhXl7BvcwIQyg) · [서울대학교 공식 연구실 소개](https://ie.snu.ac.kr/dsba_lab/)
- [성균관대학교 구자환](https://www.youtube.com/c/BigDataKoo/)
- [국민대학교 강승식](https://www.youtube.com/channel/UC9BQ1zJdpPBit9V3IFx8GDw/)

강필성 교수님의 소속 표기는 서울대학교 공식 연구실 소개에 맞춰 갱신했습니다. 기존 목록에 등록한 채널 URL은 유지했습니다.

## NLP-Overseas

- [CMU Graham Neubig](https://www.youtube.com/user/neubig)
- [UMass Mohit Iyyer](https://www.youtube.com/channel/UCKccxTTLm4_IjJdu02ZQL-Q)
- CS224N, CS336, CMU Advanced NLP의 영상·자료는 [공개 강의 업데이트](#공개-강의-업데이트)를 참고하세요.

## Vision-Korea

- [김승룡 — 기존 등록 강의 채널](https://www.youtube.com/channel/UCro6IqfE5NOspKCDusilfmw) · [KAIST CVLAB 공식 홈페이지](https://cvlab.kaist.ac.kr/)

김승룡 교수님의 현재 소속은 KAIST 김재철AI대학원입니다. 기존에 등록한 고려대학교 시절의 채널 링크는 유지했습니다.

## Vision-Overseas

- [University of Michigan Justin Johnson — 기존 등록 재생목록](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)
- Stanford CS231n의 2025 공개 영상과 2026 과제는 [공개 강의 업데이트](#공개-강의-업데이트)를 참고하세요.

## Reinforcement Learning

- **Stanford CS224R:** 2025 공개 영상과 2026 공식 강좌 자료를 제공합니다.
- **UC Berkeley CS285:** 공식 사이트가 연결한 2023 공개 영상과 2026 공식 강좌 자료를 제공합니다.

영상과 공식 자료 링크는 [공개 강의 업데이트](#공개-강의-업데이트)에 정리했습니다.

## Seminar

- [서울대학교 AI연구원](https://www.youtube.com/c/%EC%84%9C%EC%9A%B8%EB%8C%80%ED%95%99%EA%B5%90AI%EC%97%B0%EA%B5%AC%EC%9B%90)
- [KAIST 스타트업 — 기존 등록 딥러닝 세미나 목록](https://www.youtube.com/channel/UC7Xy68lSL9SHVgJ14bIORrQ/playlists)
- [서울대학교 DSBA 연구실](https://www.youtube.com/channel/UCPq01cgCcEwhXl7BvcwIQyg) — 자연어처리·비전·시계열 관련 연구실 채널

## Contributing

새 학기 공개 강의, 공식 재생목록, 변경된 교수 소속, 끊어진 링크를 [Issue](https://github.com/teimakes/ml-dl-university/issues/new?template=lecture.yml) 또는 PR로 제안해 주세요.

강의명, 대학·강사, 영상 학기, 자료 학기, 공식 출처를 함께 적어 주시면 확인하기 쉽습니다. 자세한 기준과 형식은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
