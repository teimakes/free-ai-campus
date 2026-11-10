# CMU 11-785 — 딥러닝 원리와 구현

[한국어 가이드 목차](../README.md) · [전체 자료 목록](../../../README.md)

## 자료 정보

- 성격: 한국어 학습 가이드. 전체 강의 번역이 아닙니다.
- 제공자: Carnegie Mellon University / Bhiksha Raj·Rita Singh.
- 원문: [2026 Fall 공식 강좌](https://deeplearning.cs.cmu.edu/F26/) · [차시별 영상·슬라이드](https://deeplearning.cs.cmu.edu/F26/pages/tables/lectures_table.html).
- 언어·버전: 영어 / 2026 Fall. 확인일: 2026-11-10.
- 공개 범위: 진행 중인 강좌의 일부 녹화·슬라이드. 과제·채점에는 Canvas·Gradescope 등 수강생용 서비스가 포함되므로 전체 과제 접근을 보장하지 않습니다.

## 무엇을 배우나요?

신경망의 표현 능력과 학습에서 시작해 CNN, 순환 신경망, Attention·Transformer로 이어집니다. 모델을 사용하는 것에서 한 단계 나아가 계산과 학습 과정을 이해하려는 학습자에게 추천합니다.

## 시작 전 준비

공식 안내는 Python 3, NumPy·PyTorch 사용과 미분·연쇄법칙, 선형대수, 기초 확률 지식을 요구합니다. [강좌 소개·선수지식](https://deeplearning.cs.cmu.edu/F26/pages/general_logistics.html)을 확인하세요.

## 추천 학습 순서

1. 신경망 표현과 학습 강의를 읽으며 입력·가중치·손실의 관계를 정리합니다.
2. CNN과 순환망에서 데이터 모양과 파라미터 공유 방식을 비교합니다.
3. Attention·Transformer로 넘어가 각 연산의 입력·출력 차원을 기록합니다.

## 직접 해볼 과제

작은 다층 퍼셉트론의 순전파와 역전파를 NumPy로 작성하고, 유한차분으로 계산한 기울기와 비교해 보세요. 같은 입력으로 PyTorch 결과도 대조하면 구현 오류를 찾는 연습이 됩니다. 이는 편집자가 제안한 별도 과제입니다.

확인 범위: 강좌 소개·선수지식·공개 영상 연결·진행 일정. 전체 영상 재생과 과제 실행은 검증하지 않았습니다. [확인 기록](../../content-update-2026-11-10.md)
