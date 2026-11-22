import os

import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
from edtrace import link, text, image
from lecture_util import article_link, post_link, video_link, get_local_url
from references import shannon_1950, lstm_1997, brants_2007, bengio_2003, glorot_2010, seq2seq_2014
from references import bahdanau_2015_attention, transformer_2017, gpt2_2019, t5_2019, kaplan_scaling_laws_2020, mup_2022
from references import dpo_2023, adamw_2017, adam_2014, grpo, ppo_2017, muon_2024
from references import large_batch_training_2018, wsd_2024, cosine_learning_rate_2017, moe_2017, switch_transformers_2021, auxfree_2024, mtp_2024
from references import megatron_lm_2019, shazeer_2020, elmo_2018, bert_2018
from references import rms_norm_2019, layernorm_2016, pre_post_norm_2020, qk_norm_2023
from references import rope_2021, soap_2024, sparse_transformer_2019, gqa_2023, mla_2024
from references import linear_attention_2020, mamba_2_2024, gdn_2024, mamba_3_2026
from references import megabyte_2023, byt5_2021, blt_2024, tfree_2024, hnet_2025, sennrich_2016, zero_2019, gpipe_2018
from references import regmix_2025, olmix_2026, wrap_2024

from references import gpt_3_2020, gpt_4_2023, instruct_gpt_2022
from references import the_pile_2020, gpt_j_2021, opt_175b_2022, bloom_2022, palm_2022, chinchilla_2022
from references import llama_2023, llama_2_2023, llama_3_2024
from references import mistral_7b_2023, mixtral_2024
from references import deepseek_67b_2024, deepseek_v2_2024, deepseek_v3_2024, deepseek_r1_2025, deepseek_v3_2_2025
from references import qwen_2_5_2024, qwen_3_2025
from references import kimi_1_5_2025, kimi_k2_5_2026
from references import glm_4_5_2025, glm_5_2026
from references import minimax_m2_5_2026
from references import xiaomi_mimo_v2_2026

from references import marin_8b_2025, marin_32b_2025
from references import olmo_7b_2024, olmo_2_2025, olmo_3_2025
from references import nemotron_15b_2024, nemotron_3_2025

import tiktoken

def main():
    welcome()
    why_this_course_exists()
    current_lm_landscape()

    what_is_this_program()

    course_logistics()
    course_syllabus()

    tokenization()  # 첫 단원

    text("다음 시간: 자원 소요량 계산")


def welcome():
    text("## CS336: 처음부터 만드는 언어 모델(2026년 봄)"),

    image("images/course-staff.png", width=600)
    text("CS336을 세 번째로 개설합니다.")

    text("두 번째로 개설한 2025년 봄 학기 강의는 [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)에서 볼 수 있습니다.")
    text("무엇이 달라졌을까요?")
    text("- 처음부터 직접 만든다는 원칙은 같습니다.")
    text("- 학습 시간에 비해 얻는 것이 많은 개념을 우선하고, 세부 사항에 집중하더라도 전체 내용을 놓치지 않습니다.")
    text("- 현대 언어 모델의 구성 요소인 전문가 혼합, 긴 문맥, 에이전트를 더 많이 다룹니다.")


def why_this_course_exists():
    text("## 이 강좌를 만든 이유")

    text("문제는 연구자들이 **기반 기술에서 점점 멀어지고 있다**는 것입니다.")
    text("- 2016년에는 연구자가 직접 모델을 구현하고 학습했습니다.")
    text("- 2018년에는 연구자가 BERT 같은 모델을 내려받아 미세조정했습니다.")
    text("- 오늘날 연구자는 GPT, Claude, Gemini 같은 API 모델에 프롬프트를 입력합니다.")

    text("한 단계 더 추상화된 도구를 쓰면 생산성이 높아집니다. 다만 다음과 같은 문제가 있습니다.")
    text("- 프로그래밍 언어나 운영체제와 달리, 이런 추상화에서는 내부 구현의 세부 사항이 드러납니다.")
    text("- 소프트웨어의 여러 계층을 뜯어고쳐야 하는 기초 연구가 여전히 남아 있습니다.")

    text("**기초 연구**를 하려면 이 기술을 **온전히 이해**해야 합니다.")

    text("이 강좌의 원칙은 **직접 만들면서 이해하는 것**입니다.")
    text("그런데 한 가지 문제가 있습니다.")

    text("## 언어 모델의 산업화")
    image("https://upload.wikimedia.org/wikipedia/commons/c/cc/Industrialisation.jpg", width=400)

    text("최첨단 모델의 학습에는 아주 큰 비용이 듭니다.")
    text("- 2023년 GPT-4의 학습 비용은 1억 달러였다고 알려져 있습니다. "), article_link("https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/")
    text("- 2025년 xAI는 Grok의 학습을 위해 GPU 23만 개로 클러스터를 구축했습니다. "), article_link("https://x.com/elonmusk/status/1947701807389515912")

    text("최첨단 모델을 만드는 구체적인 방법은 공개되어 있지 않습니다.")
    text("GPT-4 기술 보고서에서 발췌 "), link(gpt_4_2023), text(":")
    image("images/gpt4-no-details.png", width=600)

    text("우리가 직접 최첨단 모델을 만드는 것은 현재 여건으로 어렵습니다.")
    text("파라미터가 10억 개 미만인 작은 언어 모델은 만들 수 있지만, 여기서 얻은 결과가 대형 모델의 특성을 대표하지 않을 수 있습니다.")

    text("예 1. 전체 FLOP 중 어텐션과 MLP가 각각 차지하는 비율은 모델 규모에 따라 달라집니다. "), post_link("https://x.com/stephenroller/status/1579993017234382849")
    image("images/roller-flops.png", width=400)
    text("예 2. 규모가 커지면서 새로운 행동이 나타납니다. "), link("https://arxiv.org/pdf/2206.07682")
    image("images/wei-emergence-plot.png", width=600)

    text("## 이 수업에서 배워 최첨단 모델에도 적용할 수 있는 것은 무엇일까요?")
    text("지식을 세 종류로 나눠 봅시다.")
    text("- **작동 원리**: Transformer가 무엇이며 모델 병렬화가 어떻게 작동하는지 이해합니다.")
    text("- **접근 방식**: 하드웨어를 최대한 활용하고 규모의 영향을 중요하게 고려합니다.")
    text("- **경험적 판단**: 어떤 데이터와 모델 설계가 높은 정확도로 이어지는지 판단합니다.")

    text("작동 원리와 접근 방식은 가르칠 수 있으며, 여기서 배운 것을 다른 규모에도 적용할 수 있습니다.")
    text("경험적 판단은 규모가 달라지면 그대로 통하지 않을 수 있어 일부만 가르칠 수 있습니다.")

    text("## 경험적으로 내린 판단? 🤷")
    text("일부 설계 결정은 아직 근거를 설명하기 어려우며, 실험을 해 보면서 내린 것입니다.")
    text("예를 들어 SwiGLU를 소개한 Noam Shazeer의 논문을 봅시다. "), link(shazeer_2020)
    image("images/divine-benevolence.png", width=600)

    text("## 뼈아픈 교훈(The Bitter Lesson)")
    text("잘못된 해석: 규모만 중요하고 알고리즘은 중요하지 않다.")
    text("올바른 해석: 규모를 키울 수 있는 알고리즘이 중요하다.")
    text("### 정확도 = 효율 × 자원")
    text("실제로 규모가 클수록 낭비를 감당하기 어려워 효율이 훨씬 중요해집니다.")
    link("https://arxiv.org/abs/2005.04305"), text("에서는 2012년부터 2019년 사이 ImageNet에서 알고리즘 효율이 44배 높아졌음을 보였습니다.")

    text("질문을 이렇게 바꿔 봅시다. 연산량과 데이터 예산이 정해져 있을 때, 어느 정도로 좋은 모델을 만들 수 있을까요?")
    text("다시 말해 **효율을 최대화하는 문제**입니다.")


def current_lm_landscape():
    text("## 신경망을 쓰기 전(2010년대 이전)")
    text("- 영어의 엔트로피를 측정하기 위한 언어 모델 "), link(shannon_1950)
    text("- 기계 번역과 음성 인식 시스템에 사용된 N-그램 언어 모델 "), link(brants_2007)

    text("## 신경망의 구성 요소(2010년대)")
    text("- 장단기 기억 신경망(LSTM) "), link(lstm_1997)
    text("- 최초의 신경망 언어 모델 "), link(bengio_2003)
    text("- 시퀀스-투-시퀀스 모델링(기계 번역) "), link(seq2seq_2014)
    text("- Adam 옵티마이저 "), link(adam_2014)
    text("- 어텐션 메커니즘(기계 번역) "), link(bahdanau_2015_attention)
    text("- Transformer 구조(기계 번역) "), link(transformer_2017)
    text("- 전문가 혼합 "), link(moe_2017)
    text("- 모델 병렬화 "), link(gpipe_2018), link(zero_2019), link(megatron_lm_2019)

    text("## 초기 기반 모델(2010년대 후반)")
    text("- ELMo: LSTM으로 사전학습하고 미세조정해 후속 과제 성능을 높였습니다. "), link(elmo_2018)
    text("- BERT: Transformer로 사전학습하고 미세조정해 후속 과제 성능을 높였습니다. "), link(bert_2018)
    text("- Google의 T5(110억): 모든 과제를 텍스트 입력에서 텍스트 출력을 만드는 문제로 표현했습니다. "), link(t5_2019)

    text("## 모델 규모 키우기")
    text("- OpenAI의 GPT-2(15억): 자연스러운 글을 생성하고 제로샷 능력의 초기 징후를 보였습니다. "), link(gpt2_2019)
    text("- 스케일링 법칙: 규모를 키울 때의 성능 향상을 기대하고 예측할 근거를 제시했습니다. "), link(kaplan_scaling_laws_2020)
    text("- OpenAI의 GPT-3(1,750억): 문맥 내 학습 "), link(gpt_3_2020)
    text("- Google의 PaLM(5,400억): 규모가 매우 컸지만 학습량이 부족했습니다. "), link(palm_2022)
    text("- DeepMind의 Chinchilla(700억): 주어진 연산량에서 최적인 스케일링 법칙 "), link(chinchilla_2022)

    text("## 공개 모델")
    text("초기에는 GPT-3를 재현하려 했습니다.")
    text("- EleutherAI의 공개 데이터셋(The Pile)과 모델(GPT-J) "), link(the_pile_2020), link(gpt_j_2021)
    text("- Meta의 OPT(1,750억): GPT-3 재현 과정에서 하드웨어 문제가 많이 발생했습니다. "), link(opt_175b_2022)
    text("- Hugging Face와 BigScience의 BLOOM(1,760억): 데이터 확보에 중점을 뒀습니다. "), link(bloom_2022)

    text("주목할 만한 가중치 공개 모델(가중치와 논문):")
    text("- Meta의 Llama 모델 "), link(llama_2023), link(llama_2_2023), link(llama_3_2024)
    text("- Mistral의 모델 "), link(mistral_7b_2023), link(mixtral_2024)
    text("- DeepSeek의 모델 "), link(deepseek_67b_2024), link(deepseek_v2_2024), link(deepseek_v3_2024), link(deepseek_r1_2025), link(deepseek_v3_2_2025)
    text("- Alibaba의 Qwen 모델 "), link(qwen_2_5_2024), link(qwen_3_2025)
    text("- Moonshot의 Kimi 모델 "), link(kimi_1_5_2025), link(kimi_k2_5_2026)
    text("- Z.ai의 GLM 모델 "), link(glm_4_5_2025), link(glm_5_2026)
    text("- Minimax의 모델 "), link(minimax_m2_5_2026)
    text("- Xiaomi의 MIMO 모델 "), link(xiaomi_mimo_v2_2026)
    text("이 모델들은 GPT, Claude, Gemini 등 비공개 모델의 성능에 가까워지고 있습니다.")

    text("오픈소스 모델(가중치, 논문, 코드, 데이터):")
    text("- AI2의 Olmo 모델 "), link(olmo_7b_2024), link(olmo_2_2025), link(olmo_3_2025)
    text("- NVIDIA의 Nemotron 모델 "), link(nemotron_15b_2024), link(nemotron_3_2025)
    text("- 개발 과정도 공개하는 Marin의 모델 "), link(marin_8b_2025), link(marin_32b_2025)

    text("신뢰와 혁신을 위해서는 개방성이 중요합니다. "), link("https://arxiv.org/abs/2403.07918")
    text("공개 모델에 담긴 아이디어 덕분에 CS336을 가르칠 수 있습니다.")

    text("언어 모델이란 무엇일까요?")
    text("- 2018년(BERT): 미세조정하는 모델")
    text("- 2020년(GPT-3): 프롬프트를 입력하는 모델")
    text("- 2022년(ChatGPT): 대화를 나누는 모델 "), link(title="대화 예시", url="https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/all/train?row=72&conversation-viewer=72")
    text("- 2026년(에이전트): 자율적으로 행동하는 모델 "), link(title="실행 과정 예시", url="https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/viewer/default/train?conversation-viewer=1")

    text("기본 원리는 같습니다. 어텐션, 커널, 최적화가 필요합니다.")
    text("요구 사항은 달라졌습니다. 문맥이 길어졌고 추론 효율이 더욱 중요해졌습니다.")


def what_is_this_program():
    text("이 강의는 *실행 가능한 강의*입니다. 프로그램을 실행하면 강의 내용이 표시됩니다.")
    text("실행 가능한 강의에서는 다음과 같은 일을 할 수 있습니다.")
    text("- 모든 내용이 코드이므로 코드를 읽고 실행할 수 있습니다.")
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total
    text("- 강의가 어떤 계층 구조로 구성되어 있는지 확인할 수 있습니다.")


def course_logistics():
    text("모든 정보는 온라인에 있습니다. "), link(title="강좌 웹사이트", url="https://stanford-cs336.github.io/spring2026/")

    text("이 수업은 5학점입니다.")
    text("2024년 봄 강의 평가에서 한 학생이 남긴 의견입니다.")
    text("> *과제 하나가 CS 224n의 과제 다섯 개와 기말 프로젝트를 모두 합친 것과 거의 같은 분량이었다. 그런데 그게 고작 첫 과제였다.*")

    text("## 이 수업을 들을 만한 이유")
    text("- 무엇이 어떻게 작동하는지 끝까지 이해하고 싶습니다.")
    text("- 연구에 필요한 구현 역량을 키우고 싶습니다.")

    text("## 이 수업이 맞지 않을 수 있는 경우")
    text("- 이번 분기에 실제 연구 성과를 내야 합니다. 지도교수와 상의하세요.")
    text("- 멀티모달, RAG 등 최근 주목받는 AI 기법을 배우고 싶습니다. 그런 목적이라면 세미나 수업을 들으세요.")
    text("- 자신의 응용 분야에서 좋은 결과를 얻고 싶습니다. 그런 목적이라면 기존 모델에 프롬프트를 입력하거나 미세조정하세요.")

    text("## 집에서 수업을 따라가는 방법")
    text("- 모든 강의 자료와 과제를 온라인에 공개하므로 누구나 스스로 학습할 수 있습니다.")
    text("- 강의는 [CGOE](https://cgoe.stanford.edu/)를 통해 녹화합니다.")

    text("## 과제")
    text("- 과제는 기초, 시스템, 스케일링 법칙, 데이터, 정렬의 다섯 가지입니다.")
    text("- 뼈대 코드는 제공하지 않습니다. 구현이 올바른지 확인할 수 있도록 단위 테스트와 어댑터 인터페이스를 제공합니다.")
    text("- 로컬에서 구현하고 정확성을 검사한 다음 클러스터에서 정확도와 속도를 측정합니다.")
    text("- 일부 과제에는 순위표가 있습니다. 정해진 학습 예산에서 퍼플렉서티를 최소화하는 방식입니다.")

    text("## AI 사용 지침")
    text("- 코딩 에이전트로 모든 과제를 풀 수 있지만 그렇게 하면 배우는 것이 없습니다.")
    text("- AI는 질문에 답하고 학습을 지도하는 데 매우 유용할 수 있습니다.")
    text("- AI가 학습을 돕도록 지시하는 수업 제공 AGENTS.md 파일을 반드시 사용해야 합니다.")
    text("- [AI 사용 지침](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)을 읽으세요.")

    text("## 연산 자원")
    text("- 연산 자원을 제공한 [Modal](https://modal.com/)에 감사드립니다. 🙏")
    text("- 연산 자원에 접근하고 사용하는 방법은 [안내문](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)을 읽으세요.")


def course_syllabus():
    basics()         # 과제 1: 토큰화, 모델 구조, 학습
    systems()        # 과제 2: 커널, 병렬화, 추론
    scaling_laws()   # 과제 3: 스케일링 법칙
    data()           # 과제 4: 평가, 선별, 변환, 필터링, 중복 제거, 혼합
    alignment()      # 과제 5: RLHF, 강화학습 알고리즘, 강화학습 시스템

    text("**효율**이 중요하다는 점을 기억하세요.")
    text("- 자원 = 데이터 + 하드웨어(연산, 메모리, 통신 대역폭)")
    text("- 정해진 자원으로 가장 좋은 모델을 어떻게 학습할까요?")

    text("현재는 연산 자원이 제한되므로 주어진 하드웨어를 최대한 활용하는 방향으로 설계합니다.")
    text("- 시스템: 효율을 직접 다룹니다.")
    text("- 토큰화: 원시 바이트를 직접 처리하는 방식은 단순하지만 현재 모델 구조에서는 연산 효율이 낮습니다.")
    text("- 모델 구조: KV 캐시 공유, 슬라이딩 윈도 어텐션 등 많은 변경은 메모리나 FLOP을 줄이려는 데서 비롯됩니다.")
    text("- 데이터 필터링: 품질이 낮거나 무관한 데이터로 가중치를 갱신하는 데 연산 자원을 낭비하지 않습니다.")
    text("- 스케일링 법칙: 작은 모델에서 적은 연산량으로 하이퍼파라미터를 조정합니다.")

    text("앞으로는 데이터 양이 제약이 될 것입니다.")


class Tokenizer(ABC):
    "토크나이저의 추상 인터페이스."
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


def basics():
    text("목표는 기본적인 언어 모델을 학습할 수 있게 되는 것입니다.")
    text("구성 요소: 토큰화, 모델 구조, 학습")

    text("## 토큰화")
    text("모델이 처리하는 기본 단위는 무엇일까요?")
    text("엄밀히 말해 토크나이저는 원시 입력(바이트)과 정수 시퀀스(토큰) 사이를 양방향으로 변환합니다.")
    image("images/tokenized-example.png", width=600) 
    text("널리 쓰이는 토크나이저: **바이트 쌍 인코딩**(BPE) "), link(sennrich_2016)
    text("기본 생각은 입력을 자주 나타나는 단위로 나누는 것입니다.")
    text("효율의 관점에서 봅시다.")
    text("- 문맥 길이를 줄입니다(1,000바이트 → 약 250토큰).")
    text("- 계산을 적응적으로 배분합니다. 입력에서 주목할 부분에 모델의 표현 용량을 더 많이 씁니다.")

    text("궁극적으로는 바이트를 직접 처리하는, 토크나이저 없는 모델 구조를 만들고 싶습니다. "), link(byt5_2021), link(megabyte_2023), link(blt_2024), link(tfree_2024), link(hnet_2025)
    text("유망한 방식이지만 아직 최첨단 모델 규모로 확대되지는 않았습니다.")
    
    text("## 모델 구조")
    text("먼저 원래의 Transformer를 살펴봅니다. "), link(transformer_2017)
    image("images/transformer-architecture.png", width=500)

    text("이후에 달라진 점:")
    text("- 활성 함수: ReLU, SwiGLU "), link(shazeer_2020)
    text("- 위치 인코딩: 사인파, RoPE "), link(rope_2021)
    text("- 정규화: LayerNorm, RMSNorm, QK 정규화, 사전 정규화와 사후 정규화 "), link(layernorm_2016), link(rms_norm_2019), link(qk_norm_2023), link(pre_post_norm_2020)
    text("- 어텐션: 전체 어텐션, 희소·국소 어텐션, 그룹 쿼리 어텐션(GQA), 다중 헤드 잠재 어텐션(MLA) "), link(sparse_transformer_2019), link(gqa_2023), link(mla_2024)
    text("- 순환 모델, 상태 공간 모델, 선형 어텐션: Mamba, Gated DeltaNet "), link(linear_attention_2020), link(mamba_2_2024), link(gdn_2024), link(mamba_3_2026)
    text("- MLP: 밀집 모델, 전문가 혼합 "), link(moe_2017), link(switch_transformers_2021)
    text("- 모델의 형태(은닉 차원, 깊이, 헤드 수, 전문가 수)")

    text("## 학습")
    text("모델의 파라미터를 어떻게 정할까요?")
    text("- 손실 함수(예: 여러 토큰 예측) "), link(mtp_2024), link(deepseek_v3_2024)
    text("- 옵티마이저(예: AdamW, SOAP, Muon) "), link(adam_2014), link(adamw_2017), link(soap_2024), link(muon_2024)
    text("- 초기화 스케일(예: Xavier 초기화, muP) "), link(glorot_2010), link(mup_2022)
    text("- 학습률 스케줄(예: 코사인, WSD) "), link(cosine_learning_rate_2017), link(wsd_2024)
    text("- 정규화 기법(예: 드롭아웃, 가중치 감쇠)")
    text("- 배치 크기(예: 임계 배치 크기) "), link(large_batch_training_2018)
    text("- MoE에 특유한 문제: 부하 균형(예: 보조 손실을 쓰지 않는 aux-free 방식) "), link(auxfree_2024), link(deepseek_v3_2024)

    text("## 과제 1(기초)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment1-basics"), link(title="PDF", url="https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2026_assignment1_basics.pdf")
    text("- BPE 토크나이저 구현")
    text("- Transformer, 교차 엔트로피 손실, AdamW 옵티마이저, 학습 루프 구현")
    text("- 필요한 자원 계산")
    text("- TinyStories와 OpenWebText로 학습")
    text("- 순위표: B200 한 대에서 45분 동안 학습해 OpenWebText 퍼플렉서티 최소화 "), link(title="지난해 순위표", url="https://github.com/stanford-cs336/spring2025-assignment1-basics-leaderboard")

    text("모델을 설계할 때는 다음 세 가지를 함께 고려합니다.")
    text("- 표현력: 데이터 안의 복잡한 의존 관계를 표현할 수 있는가?")
    text("- 안정성: 파라미터와 기울기의 노름이 적정 범위에 머무르는가?")
    text("- 효율: 학습과 추론 모두에서 하드웨어가 빠르게 실행할 수 있는가?")


def systems():
    text("목표는 GPU나 TPU의 성능을 최대한 활용하는 것입니다.")
    text("구성 요소: 커널, 병렬화, 추론")

    text("## 기초")
    text("- 자원 소요량 계산: 모델의 메모리 사용과 연산 특성")
    total_flops = 6 * 70e9 * 1e12  # 700억 개 파라미터를 1조 토큰으로 학습하는 연산량 = 4.2e23 FLOP @inspect total_flops
    image("images/compute-memory.png", width=300)
    text("- 모델 파라미터를 메모리(HBM)에서 연산 장치(SM)로 옮겨야 합니다.")
    text("- 예: B200의 BF16 연산 성능은 2.25 PFLOP/s이고 메모리 대역폭은 8 TB/s입니다.")
    text("- 루프라인 분석: 처리 속도를 제한하는 요인이 연산인지 메모리인지 파악합니다.")
    text("- 벤치마킹과 프로파일링(nsight): 실제 실행 중에 일어나는 일을 확인합니다.")

    text("[DGX B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html):")
    image("https://docs.nvidia.com/dgx/dgxb200-user-guide/_images/dgx-b200-system-topology.png", width=500)

    text("## 커널")
    text("- 커널은 GPU에서 실행하는 함수입니다.")
    text("- PyTorch에서는 기본 연산마다 표준 커널을 실행합니다.")
    text("- GPU를 더 빠르게 구동하도록 커널을 직접 작성할 수 있습니다.")
    text("- 원칙: 데이터 이동이 최소화되도록 계산을 구성합니다.")
    text("- 단순한 방식: HBM 읽기 → A 계산 → HBM 쓰기 → HBM 읽기 → B 계산 → HBM 쓰기")
    text("- 결합한 방식: HBM 읽기 → A와 B 계산 → HBM 쓰기")
    text("- 기법: 연산자 결합(행렬곱 + 활성 함수), 타일링(FlashAttention)")
    text("- 워프 분기, 메모리 접근 병합, 뱅크 충돌, 점유율, 대량 비동기 메모리 전송")
    text("- CUDA, **Triton**, CUTLASS, ThunderKittens로 커널을 작성합니다.")

    text("## 병렬화")
    text("- GPU가 1,024개 있다면 어떻게 할까요?")
    text("- GPU 사이의 데이터 이동은 더 느립니다. 여기서도 데이터 이동을 최소화한다는 원칙은 같습니다.")
    text("- 기존의 집합 통신 연산(예: gather, reduce, all-reduce)을 사용합니다.")
    text("- 여러 GPU에 걸쳐 메모리를 분할합니다(파라미터, 활성값, 기울기, 옵티마이저 상태).")
    text("- 계산을 나누는 방법: 데이터·텐서·파이프라인·시퀀스·전문가 병렬화")
    
    text("## 추론")
    text("목표는 프롬프트가 주어졌을 때 토큰을 생성하는 것입니다. 실제로 모델을 사용하려면 이 과정이 필요합니다.")
    text("강화학습, 테스트 시점 연산, 평가에도 추론이 필요합니다.")
    text("두 단계: 프리필(prefill)과 디코딩(decode)")
    image("images/prefill-decode.png", width=500)
    text("- 프리필: 학습할 때처럼 토큰이 주어져 있어 한꺼번에 처리할 수 있습니다. 연산량이 병목입니다.")
    text("- 디코딩: 토큰을 한 번에 하나씩 생성해야 합니다. 메모리가 병목입니다.")
    text("디코딩 속도를 높이는 방법:")
    text("- 가지치기, 양자화, 증류로 연산 비용이 낮은 모델 사용")
    text("- 추측 디코딩: 비용이 낮은 초안 모델이 여러 토큰을 제안하면 전체 모델이 이를 병렬로 채점합니다. 정확한 디코딩 결과를 얻을 수 있습니다.")
    text("- 시스템 최적화: 커널 결합, 연속 배치 처리")

    text("## 과제 2(시스템)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment2-systems"), link(title="2025년 봄 학기 PDF", url="https://github.com/stanford-cs336/assignment2-systems/blob/spring2025/cs336_spring2025_assignment2_systems.pdf")
    text("- Triton으로 결합형 RMSNorm 커널 구현")
    text("- 분산 데이터 병렬 학습 구현")
    text("- 옵티마이저 상태 분할 구현")
    text("- 구현한 코드 벤치마킹 및 프로파일링")

    text("추천 도서: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)")
    text("- 언어 모델을 위한 시스템을 어떤 개념으로 접근할지 잘 설명합니다.")
    text("- Google에서 쓴 책이라 TPU를 주로 다루지만 상위 수준의 개념은 비슷합니다.")


def scaling_laws():
    text("연산 예산이 1e25 FLOP이라면 좋은 모델을 학습하기 위해 어떤 하이퍼파라미터를 쓸까요?")
    text("전체 규모에서 하이퍼파라미터를 탐색하는 데는 비용이 너무 많이 듭니다.")

    text("하나의 규모만 생각하는 대신, 연산량에 따라 하이퍼파라미터를 정하는 **규모별 학습 설정**(FLOP → 하이퍼파라미터)을 생각합니다.")
    text("규모별 학습 설정을 정하려면 다음과 같이 합니다.")
    text("- 여러 작은 규모(예: 최대 1e24 FLOP)에서 실험해 손실을 구합니다.")
    text("- 스케일링 법칙을 적합해 목표 규모(예: 1e25 FLOP)에서 그 학습 설정의 손실을 예측합니다.")

    text("이렇게 하면 다음 두 가지를 할 수 있습니다.")
    text("1. 작은 규모의 실험을 이용해 더 큰 목표 규모에 맞게 학습 설정을 최적화합니다.")
    text("2. 목표 규모의 실험을 실행하기 전에 손실을 예측합니다.")
    text("스케일링 법칙이 저절로 성립하는 것은 아닙니다. 규모별 학습 설정을 신중하게 구성해야 합니다.")
    text("**하이퍼파라미터를 다른 규모로 이전**할 수 있도록 모델의 파라미터화를 정합니다. "), link(mup_2022)
    text("예측 가능성은 적어도 최적성만큼 중요합니다.")

    text("질문: 연산 예산이 C = 6 N D로 주어지면 더 큰 모델(N)을 쓸까요, 더 많은 토큰(D)으로 학습할까요?")
    text("대표적인 연산량 최적 스케일링 법칙: "), link(kaplan_scaling_laws_2020), link(chinchilla_2022)
    text("- IsoFLOP 곡선: 여러 작은 연산 예산에 대해 각각 최적인 N을 찾습니다.")
    text("- 이어서 스케일링 법칙을 적합해 더 큰 연산 예산으로 외삽합니다.")
    image("images/chinchilla-isoflop.png", width=800)
    text("요약하면 D = 20 N 부근이 최적입니다. 예를 들어 파라미터 700억 개 모델은 약 1.4조 토큰으로 학습합니다.")
    text("단, 작은 모델이 유리해지는 추론 비용은 이 계산에 포함되지 않습니다.")

    text("현재 진행 중인 Marin의 사례 "), post_link("https://x.com/percyliang/status/2034367256277533100")
    image("https://pbs.twimg.com/media/HDuErvvbsAAQ5Yt?format=jpg&name=4096x4096", width=600)
    text("이번 주에 학습이 끝날 예정입니다. 사전에 등록한 손실 예측값과 얼마나 잘 맞는지 살펴봅시다.")

    text("## 과제 3(스케일링 법칙)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment3-scaling"), link(title="2025년 봄 학기 PDF", url="https://github.com/stanford-cs336/assignment3-scaling/blob/master/cs336_spring2025_assignment3_scaling.pdf")
    text("- 과거 실행 결과를 바탕으로 학습 API를 정의합니다(하이퍼파라미터 → 손실).")
    text("- 정해진 FLOP 예산 안에서 ‘학습 작업’을 제출하고 관측값을 수집합니다.")
    text("- 관측값에 스케일링 법칙을 적합합니다.")
    text("- 외삽한 하이퍼파라미터와 예측 손실을 제출합니다.")
    text("- 순위표: 주어진 FLOP 예산에서 손실 최소화")


def data():
    text("질문: 모델에 어떤 능력이 있기를 원할까요?")
    text("여러 언어를 쓰는 능력일까요? 대화를 잘하는 능력일까요? 에이전트로서 코딩을 수행하는 능력일까요?")

    text("## 평가")
    text("평가의 목적은 무엇일까요?")
    text("1. 내부 평가: 모델 개발의 방향을 정합니다. 규모가 바뀔 때 지표가 매끄럽게 변하는지와 상대 성능이 중요합니다.")
    text("2. 외부 평가: 실제 사용 사례에서의 절대적인 품질을 측정합니다. 실제 환경을 얼마나 잘 반영하는지가 중요합니다.")
    text("평가 예:")
    text("1. 퍼플렉서티: 평가 데이터 오염을 피하려면 인터넷에 없는 비공개 문서로 측정하는 것이 이상적입니다.")
    text("2. 고급 사용 사례: GPQA, HLE, SWE-Bench, Terminal-Bench")
    text("언어 모델은 범용이므로 다양한 평가가 필요합니다.")

    text("## 데이터 선별")
    text("- 데이터는 저절로 얻어지지 않습니다.")
    text("- 출처: 인터넷에서 수집한 웹페이지, 책, arXiv 논문, GitHub 코드 등")
    image("https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png", width=600)
    text("- 저작권이 있는 데이터로 학습할 때 공정 이용을 근거로 삼을 수 있을까요? "), link("https://arxiv.org/pdf/2303.15715.pdf")
    text("- 데이터 사용 계약을 맺어야 할 수도 있습니다(예: Google의 Reddit 데이터 사용). "), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- 원본 데이터는 텍스트 대신 HTML, PDF, 디렉터리 형태로 들어오므로 처리가 필요합니다.")

    text("## 데이터 처리")
    text("- 변환: HTML과 PDF에서 본문을 추출해 텍스트로 바꿉니다.")
    text("- 필터링: 분류기를 활용해 품질이 높은 데이터를 남기고 유해한 내용을 제거합니다.")
    text("- 중복 제거: 연산을 절약하고 모델의 암기를 방지합니다. 블룸 필터나 MinHash를 사용합니다.")
    text("- 데이터 혼합: 각 출처의 비중을 얼마나 높이거나 낮출까요? "), link(regmix_2025), link(olmix_2026)
    text("- 재작성과 합성 데이터: 언어 모델로 실제 데이터를 증강해 후속 과제와 더 비슷하도록 만듭니다. "), link(wrap_2024)

    text("데이터 종류:")
    text("- 사전학습 데이터: 규모가 크고 다양합니다.")
    text("- 중간 학습 데이터: 긴 문맥을 포함하는 고품질 데이터입니다.")
    text("- 후학습 데이터: 지도 미세조정용 대화, 도구를 호출하는 에이전트의 실행 기록입니다.")

    text("## 과제 4(데이터)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment4-data"), link(title="2025년 봄 학기 PDF", url="https://github.com/stanford-cs336/assignment4-data/blob/spring2025/cs336_spring2025_assignment4_data.pdf")
    text("- Common Crawl의 HTML을 텍스트로 변환")
    text("- 품질과 유해한 내용을 기준으로 필터링할 분류기 학습")
    text("- MinHash로 중복 제거")
    text("- 순위표: 주어진 토큰 예산에서 퍼플렉서티 최소화")


def alignment():
    text("지금까지는 다음 토큰을 예측하는 완전한 지도 신호로 모델을 학습했습니다.")
    text("이제 모델이 어느 정도 작동하므로 **약한 지도 신호**로 더 개선할 수 있습니다.")
    text("왜 약한 지도 신호를 쓸까요? 답을 직접 생성하는 것보다 생성한 답을 평가하기가 더 쉬운 경우가 있기 때문입니다.")

    text("기본 절차:")
    text("1. 모델로 응답을 생성합니다.")
    text("2. 사람, 검증기 또는 언어 모델 판정자가 응답을 채점합니다.")
    text("3. 더 좋은 응답을 선호하도록 모델을 갱신합니다.")

    text("알고리즘:")
    text("- 강화학습의 근접 정책 최적화(Proximal Policy Optimization, PPO) "), link(ppo_2017), link(instruct_gpt_2022)
    text("- 직접 정책 최적화(Direct Policy Optimization, DPO): 선호 데이터를 쓰며 절차가 더 단순합니다. "), link(dpo_2023)
    text("- 그룹 상대 선호 최적화(Group Relative Preference Optimization, GRPO): 가치 함수를 없앱니다. "), link(grpo)

    text("어려운 점:")
    text("- 강화학습 알고리즘은 불안정하고 튜닝하기 어렵습니다.")
    text("- 규모가 커지면 비동기 롤아웃 추론을 비롯해 많은 인프라를 새로 구축해야 합니다.")
    text("- 시스템 효율과 현재 정책에서 직접 표본을 뽑는 정도 사이에서 계속 절충해야 합니다.")

    text("## 과제 5(정렬)")
    link(title="GitHub", url="https://github.com/stanford-cs336/assignment5-alignment"), link(title="2025년 봄 학기 PDF", url="https://github.com/stanford-cs336/assignment5-alignment/blob/spring2025/cs336_spring2025_assignment5_alignment.pdf")
    text("- 직접 선호 최적화(Direct Preference Optimization, DPO) 구현")
    text("- 그룹 상대 선호 최적화(Group Relative Preference Optimization, GRPO) 구현")


############################################################
# 토큰화

def tokenization():
    text("이 단원은 Andrej Karpathy의 토큰화 영상에서 아이디어를 얻었습니다. 영상을 참고하세요. "), video_link("https://www.youtube.com/watch?v=zduSFxRajkE")

    intro_to_tokenization()
    tokenization_examples()
    character_tokenizer()
    byte_tokenizer()
    word_tokenizer()
    bpe_tokenizer()

    text("요약:")
    text("- 토크나이저: 문자열 ↔ 토큰(인덱스)")
    text("- 문자, 바이트, 단어를 각각 기본 단위로 삼는 토큰화는 모두 효율이 많이 떨어집니다.")
    text("- BPE는 데이터에 따라 동작하는 효과적인 경험적 방법입니다.")
    text("- 현재 토큰화는 별도 단계이지만 언젠가는 바이트부터 전체 모델을 한꺼번에 학습할 수도 있습니다.")

    text("어떤 방법이든 다음 조건을 만족해야 합니다.")
    text("1. 모델(예: Transformer)이 텍스트, 영상, DNA 등 시퀀스의 일부를 묶은 추상적 단위를 처리해야 합니다.")
    text("2. 묶음의 크기는 달라질 수 있어야 합니다. 그래야 중요한 부분에 모델의 용량을 더 많이 배분할 수 있습니다.")


class CharacterTokenizer(Tokenizer):
    "문자열을 유니코드 코드 포인트의 시퀀스로 표현한다."
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    "문자열을 바이트 시퀀스로 표현한다."
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    "`indices`에서 `pair`가 나타나는 곳을 모두 `new_index`로 바꾼 결과를 반환한다."
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


@dataclass(frozen=True)
class BPETokenizerParams:
    "BPETokenizer를 지정하는 데 필요한 값."
    vocab: dict[int, bytes]     # 인덱스 -> 바이트열
    merges: dict[tuple[int, int], int]  # 인덱스1, 인덱스2 -> 새 인덱스



class BPETokenizer(Tokenizer):
    "병합 규칙과 어휘를 사용하는 BPE 토크나이저."
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # 참고: 이 구현은 매우 느리다.
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)  # @stepover
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


def get_compression_ratio(string: str, indices: list[int]) -> float:  # @inspect string indices
    "`string`을 `indices`로 토큰화했을 때 토큰당 UTF-8 바이트 수를 반환한다."
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


def get_gpt5_tokenizer():
    # 코드: https://github.com/openai/tiktoken
    return tiktoken.get_encoding("o200k_base")


def intro_to_tokenization():
    text("원시 텍스트는 일반적으로 유니코드 문자열로 표현합니다.")
    string = "Hello, 🌍! 你好!"

    text("언어 모델은 토큰 시퀀스에 확률 분포를 부여합니다. 토큰은 보통 정수 인덱스로 나타냅니다.")
    indices = [15496, 11, 995, 0]

    text("따라서 문자열을 토큰으로 *인코딩*하는 절차가 필요합니다.")
    text("토큰을 다시 문자열로 *디코딩*하는 절차도 필요합니다.")
    text(""), link(Tokenizer), text("는 encode와 decode 메서드를 구현하는 클래스입니다.")


def tokenization_examples():
    text("토크나이저의 동작을 살펴보려면 다음 사이트에서 직접 입력해 보세요. "), link(title="대화형 사이트", url="https://tiktokenizer.vercel.app/?encoder=gpt2")

    text("## 관찰한 점")
    text("- 단어와 그 앞의 공백이 같은 토큰에 포함됩니다(예: \" world\").")
    text("- 문장의 처음과 중간에 등장하는 같은 단어가 서로 다르게 표현됩니다(예: \"hello hello\").")
    text("- 숫자는 몇 자리씩 묶어 토큰화합니다.")

    text("OpenAI의 GPT-5 토크나이저(tiktoken)가 어떻게 동작하는지 봅시다.")
    tokenizer = get_gpt5_tokenizer()  # @stepover
    string = "Hello, 🌍! 你好!"  # @inspect string

    text("encode()를 거친 뒤 decode()를 적용하면 원래 문자열이 복원되는지 확인합니다.")
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    
    text("압축률: 토큰당 바이트 수")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    text("압축률이 높을수록 시퀀스가 짧아집니다. 어텐션 계산량은 시퀀스 길이의 제곱에 비례하므로 짧을수록 유리합니다.")
    text("**어휘 크기**를 늘리면 가능한 토큰 값이 많아져 압축률을 높일 수 있지만, 개별 토큰의 출현이 희소해집니다.")
    vocabulary_size = tokenizer.n_vocab  # @inspect vocabulary_size

    text("실제 어휘를 살펴봅시다. "), link(title="어휘", url=get_local_url("var/gpt5_tokenizer_vocab.txt"))
    output_tokenizer(tokenizer, "var/gpt5_tokenizer_vocab.txt")  # @stepover


def output_tokenizer(tokenizer, path: str):
    "`tokenizer`의 어휘를 `path`에 한 줄에 하나씩 쓴다."
    if not os.path.exists(path):
        vocab = [b.decode("utf-8", errors="replace") for b in tokenizer.token_byte_values()]
        with open(path, "w") as f:
            for token in vocab:
                f.write(token + "\n")


def character_tokenizer():
    text("유니코드 문자열은 유니코드 문자의 시퀀스입니다.")
    text("각 문자는 `ord`를 사용해 정수인 코드 포인트로 변환할 수 있습니다.")
    assert ord("a") == 97
    assert ord("🌍") == 127757
    text("`chr`로 다시 문자로 변환할 수 있습니다.")
    assert chr(97) == "a"
    assert chr(127757) == "🌍"

    text("이제 `Tokenizer`를 만들어 인코딩 후 디코딩하면 원문과 일치하는지 확인합시다.")
    tokenizer = CharacterTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # ord 호출 @inspect indices @stepover
    reconstructed_string = tokenizer.decode(indices)  # chr 호출 @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    text("유니코드 문자는 약 15만 개입니다. "), link(title="위키백과", url="https://en.wikipedia.org/wiki/List_of_Unicode_characters")
    vocabulary_size = max(indices) + 1  # 이 값은 하한이다. @inspect vocabulary_size
    text("문제 1. 어휘가 매우 큽니다.")
    text("문제 2. 🌍처럼 거의 나타나지 않는 문자가 많아 어휘를 비효율적으로 사용합니다.")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio @stepover
    text("이 토크나이저는 어휘가 크면서도 압축률이 낮다는 두 단점을 모두 갖습니다.")


def byte_tokenizer():
    text("유니코드 문자열은 바이트 시퀀스로 표현할 수 있으며, 각 바이트는 0부터 255까지의 정수로 나타낼 수 있습니다.")
    text("가장 널리 쓰이는 유니코드 인코딩은 "), link(title="UTF-8", url="https://en.wikipedia.org/wiki/UTF-8")

    text("유니코드 문자 중에는 한 바이트로 표현되는 문자도 있습니다.")
    assert bytes("a", encoding="utf-8") == b"a"
    text("여러 바이트가 필요한 문자도 있습니다.")
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"

    text("이제 `Tokenizer`를 만들어 인코딩과 디코딩으로 원문을 복원할 수 있는지 확인합시다.")
    tokenizer = ByteTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices @stepover
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    text("어휘는 작습니다. 한 바이트로 256가지 값을 표현할 수 있습니다.")
    vocabulary_size = 256  # @inspect vocabulary_size
    text("압축률은 어떨까요?")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio @stepover
    assert compression_ratio == 1
    text("압축률이 매우 낮아서 시퀀스가 지나치게 길어집니다.")
    text("어텐션의 계산량이 길이의 제곱에 비례해 Transformer의 문맥 길이가 제한된다는 점을 고려하면 좋지 않은 결과입니다.")


def word_tokenizer():
    text("전통적인 자연어 처리 방식에 가까운 또 다른 방법은 문자열을 단어로 나누는 것입니다.")
    string = "I'll say supercalifragilisticexpialidocious!"

    chunks = regex.findall(r"\w+|.", string)  # @inspect chunks
    text("이 정규식은 연속된 영숫자를 한데 묶어 단어로 처리합니다.")

    text("이것을 `Tokenizer`로 만들려면 각 묶음을 정수에 대응시켜야 합니다.")
    text("이제 각각의 묶음을 정수에 대응시키는 표를 만들 수 있습니다.")

    text("장점: 사람이 만든 단어가 단위이므로 각 토큰에 의미가 있습니다.")

    vocabulary_size = "Number of distinct chunks in the training data"
    compression_ratio = get_compression_ratio(string, chunks)  # @inspect compression_ratio @stepover
    text("압축률은 좋지만 어휘가 매우 커질 수 있습니다.")

    text("그 밖에도 다음과 같은 문제가 있습니다.")
    text("- 드문 단어가 많아 모델이 해당 단어를 충분히 학습하지 못합니다.")
    text("- 어휘 크기를 고정하는 방법이 분명하지 않습니다.")
    text("- 학습 중에 보지 못한 새 단어에는 특수 UNK 토큰을 부여합니다. 바람직하지 않은 방식이고 퍼플렉서티 계산에도 문제가 생길 수 있습니다.")


def bpe_tokenizer():
    text("## 바이트 쌍 인코딩(BPE)")
    text("Philip Gage가 1994년 데이터 압축을 위해 BPE 알고리즘을 소개했습니다. "), article_link("http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM")
    text("이후 신경망 기계 번역을 위해 자연어 처리에 맞게 응용했습니다. "), link(sennrich_2016)
    text("(그 전까지 논문에서는 단어 기반 토큰화를 사용했습니다.)")
    text("이후 GPT-2에도 BPE가 쓰였습니다. "), link(gpt2_2019)

    text("기본 원리: 원시 텍스트에서 토크나이저를 *학습*해 데이터에 맞는 어휘를 구성합니다.")
    text("자주 나오는 바이트 시퀀스는 하나의 토큰으로, 드문 시퀀스는 여러 토큰으로 표현합니다.")

    text("각 바이트를 토큰 하나로 놓고, 가장 자주 나오는 인접 토큰 쌍을 차례로 병합합니다.")

    text("## 토크나이저 학습")
    string = "the cat in the hat"  # @inspect string
    params = train_bpe(string, num_merges=3)

    text("## 토크나이저 사용")
    text("이제 새로운 텍스트가 주어지면 인코딩할 수 있습니다.")
    tokenizer = BPETokenizer(params)  # @stepover
    string = "the quick brown fox"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string @stepover
    assert string == reconstructed_string

    text("과제 1에서는 여기서 다음과 같이 더 발전시킵니다.")
    text("- 현재 encode()는 모든 병합 규칙을 순회합니다. 실제 적용할 규칙만 순회하세요.")
    text("- 특수 토큰(예: <|endoftext|>)을 찾아 그대로 보존하세요.")
    text("- 사전 토큰화를 사용하세요(예: GPT-2 토크나이저의 정규식).")
    text("- 구현을 가능한 한 빠르게 만드세요.")


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    text("`string`의 바이트 목록에서 시작합니다.")
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # 인덱스1, 인덱스2 => 병합한 인덱스
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # 인덱스 -> 바이트열

    for i in range(num_merges):
        # 각 토큰 쌍의 출현 횟수를 센다.
        counts = count_adjacent_pairs(indices)  # @inspect counts @stepover

        # 가장 자주 나타나는 쌍을 찾는다.
        pair = max(counts, key=counts.get)  # @inspect pair

        # 그 쌍을 병합한다.
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices @stepover

    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio

    return BPETokenizerParams(vocab=vocab, merges=merges)


def count_adjacent_pairs(indices: list[int]) -> dict[tuple[int, int], int]:
    "`indices`에서 서로 인접한 각 토큰 쌍에 출현 횟수를 대응시킨 딕셔너리를 반환한다."
    counts = defaultdict(int)
    for index1, index2 in zip(indices, indices[1:]):
        counts[(index1, index2)] += 1
    return counts


if __name__ == "__main__":
    main()
