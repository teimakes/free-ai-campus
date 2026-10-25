# From Scratch 학습 경로·무료 자료 확인 기록

확인일: **2026-10-25 (한국 시간)**.

상단 로드맵은 Python 기초를 아는 학습자가 직접 구현하는 순서로 편집한 추천 경로입니다. 제공자가 공동으로 설계한 단일 과정은 아닙니다. 강의·코드가 공개되어 있는지, 각 자료가 어떤 구현을 다루는지 확인했으며, 전체 영상 시청이나 코드 실행은 수행하지 않았습니다.

## 무료의 범위

- 수록 대상은 무료로 열람 가능한 학습 영상·강의 노트·코드입니다. 무료 회원가입이 필요한 강좌를 포함합니다.
- 공개된 코드와 강의 자료를 안내하며, 유료 도서 본문·별도 판매 영상·유료 인증서의 제공을 뜻하지 않습니다.
- API 호출, 클라우드 GPU, 모델 호스팅, 외부 검색 등 실습 서비스 이용료는 별도입니다. 특히 에이전트 예제의 API 설정 안내는 코드 열람 비용과 구분했습니다.

## ML/DL from scratch

1. **Aladdin Persson — Linear Regression with Gradient Descent**: [작성자 강의·코드 목록](https://github.com/aladdinpersson/Machine-Learning-Collection)에서 [영상](https://youtu.be/pCCUnoes1Po)과 [경사하강법 코드](https://github.com/AladdinPersson/Machine-Learning-Collection/blob/master/ML/algorithms/linearregression/linear_regression_gradient_descent.py)의 대응을 확인했습니다.
2. **Vizuara — Building Neural Networks from Scratch**: [기존 공개 재생목록](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSj6tNyn_UadmUeU3Q3oR-hu)을 신경망 단계에 배치했습니다. 기존 채널 확인 기록의 뉴런·레이어·활성함수 구현 자료를 이용합니다.
3. **Andrej Karpathy — micrograd**: [공식 강의 계획](https://karpathy.ai/zero-to-hero.html)과 [강사 저장소](https://github.com/karpathy/nn-zero-to-hero)가 [강의 영상](https://www.youtube.com/watch?v=VMj-3S1tku0) 및 micrograd 코드·노트북을 연결합니다. 역전파와 자동미분 엔진 구현을 다루는 단계입니다.

## LLM from scratch

1. **Andrej Karpathy — GPT Tokenizer**: [공식 강사 저장소](https://github.com/karpathy/nn-zero-to-hero)의 Lecture 8에서 [강의](https://www.youtube.com/watch?v=zduSFxRajkE)와 [minbpe](https://github.com/karpathy/minbpe)의 연결을 확인했습니다. BPE 학습·인코딩·디코딩 구현을 다룹니다.
2. **Andrej Karpathy — Let's build GPT**: 같은 공식 강의 목록의 [GPT 구현 영상](https://www.youtube.com/watch?v=kCc8FmEb1nY)과 [강의 코드 저장소](https://github.com/karpathy/ng-video-lecture)를 확인했습니다. Transformer 언어 모델 구현과 소규모 학습을 다룹니다. 대규모 모델 학습에 필요한 연산 비용까지 무료라는 의미는 아닙니다.
3. **Sebastian Raschka — Instruction tuning**: 기존에 확인한 [책 공식 코드 저장소](https://github.com/rasbt/LLMs-from-scratch)의 [Chapter 7](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07)을 연결했습니다. 무료 공개 코드·설명이며 도서나 Manning 판매 영상의 무료 제공을 뜻하지 않습니다.

## AI Agent from scratch

1. **Hugging Face — Dummy Agent Library**: [실제 튜토리얼](https://huggingface.co/learn/agents-course/unit1/dummy-agent-library)과 페이지가 연결한 [노트북](https://huggingface.co/agents-course/notebooks/blob/main/unit1/dummy_agent_library.ipynb)을 확인했습니다. Python 함수와 LLM을 이용해 도구 요청·실행·관찰 결과를 연결합니다. 모델 호출에는 외부 추론 API와 토큰 설정이 필요합니다.
2. **LangChain — Agents from Scratch**: [공식 저장소](https://github.com/langchain-ai/agents-from-scratch)는 에이전트 구현, 평가, 사람의 검토, 메모리 노트북을 공개합니다. LangGraph를 사용하며, 실습에는 모델 API와 선택적인 추적 서비스 설정이 안내됩니다.
3. **LangChain — Deep Agents from Scratch**: [공식 저장소](https://github.com/langchain-ai/deep-agents-from-scratch)는 계획, 파일 시스템, 하위 에이전트, 통합 리서치 에이전트로 이어지는 노트북을 공개합니다. LangGraph 기반 확장 과정이며, 모델·검색 API 이용 비용은 별도입니다.

에이전트의 'from scratch'는 기존 LLM을 이용해 에이전트의 루프·도구·상태를 구성하는 뜻입니다. 이 경로에서 LLM의 사전학습까지 처음부터 수행하는 것은 아닙니다.

## 표시·구조

- GitHub가 지원하는 Mermaid 흐름도와 파랑·보라·초록 세 트랙을 사용했습니다. 각 트랙 안에서는 왼쪽에서 오른쪽으로 진행합니다.
- 강의·자료 링크는 접이식 영역에도 제공하여 다이어그램과 별도로 열 수 있게 했습니다.
- 기존 다섯 분류와 사용자가 정리한 기존 행은 보존하고, 새로 확인한 강의·자료는 기존 표의 형식에 맞춰 개별 행으로 추가했습니다.
