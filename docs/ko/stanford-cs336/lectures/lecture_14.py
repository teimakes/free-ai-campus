from dataclasses import dataclass
import numpy as np
import itertools
import mmh3
from edtrace.file_util import download_file
from edtrace import text, image, link
from lecture_13 import the_pile
from lecture_util import article_link, post_link
from references import dolma_2024, the_pile_2020, dclm_2024

def main():
    text("## 14강. 데이터 II")
    text("지난 강의")
    text("- 운영 서비스(GitHub 등) → 덤프·크롤링(GitHub Archive 등) → 처리된 데이터(The Stack 등).")
    text("- 고려할 사항: 서비스 이용약관과 저작권(이용 허락 또는 공정 이용).")

    text("이번 강의")
    text("- 데이터 처리 과정: 변환, 필터링, 중복 제거, 혼합.")
    text("- 중간 학습과 지도 미세조정(SFT): 합성 데이터.")

    # 데이터 처리 과정
    transformation()
    filtering()
    deduplication()
    data_mixing()

    # 후학습 데이터
    post_training_data()

    text("정리")
    text("- 필터링: 원하는 데이터의 특성을 판별할 분류기를 학습한다(언어·품질·독성).")
    text("- 중복 제거: 해싱으로 큰 데이터셋에서도 유사한 항목을 찾는다.")
    text("- 혼합: 작은 규모에서 혼합 비율을 시험하고 대규모 학습에 쓸 최적 비율을 외삽한다.")
    text("- 적용 분야: 언어 식별, 품질 필터링, 독성 필터링.")
    text("- 후학습 데이터는 평가 과제와 형태가 비슷하며 합성 데이터를 쓰기도 한다.")
    text("- 실제 예제를 살펴보는 등 분야별로 필요한 작업이 많다.")


def transformation():
    text("원시 데이터가 곧바로 일반 텍스트 형태로 주어지는 것은 아니다.")
    text("HTML, PDF(arXiv), 디렉터리(코드 저장소) 등의 형태로 들어온다.")

    text("HTML을 텍스트로 변환하기(주로 다루는 경우)")
    text("- 탐색 메뉴와 광고 등 반복 요소를 제거하고 본문을 추출한다.")
    text("- 이미지와 표는 어떻게 처리할까?")
    text("- 일렬의 텍스트로 바꿔야 하므로 일부 정보는 사라진다.")
    text("- 규칙 기반 도구: trafilatura, resiliparse, jusText, lynx 등.")
    text("- 변환 정확도는 결과에 영향을 준다. "), link(dclm_2024)
    image("images/dclm-wet.png", width=300)

    text("FinePDFs "), post_link("https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog")
    image("https://huggingfacefw-finepdfsblog.hf.space/_astro/pdf-description.Cb49jXc6_Z17eX4E.webp", width=600)
    text("- 출처: Common Crawl.")
    text("- PDF는 크기가 커서 일부만 수집되기도 한다. 이런 PDF는 다시 수집한다.")
    text("- 시각 언어 모델 기반 OCR(RolmOCR) 또는 Docling을 사용하고 처리 속도를 높인다.")
    text("- 정제와 필터링을 많이 거친다.")
    text("- 원본의 지면 배치 정보 상당 부분이 사라진다.")


def filtering():
    text("알고리즘의 기본 문제")
    text("- **목표 데이터** T와 대량의 **원시 데이터** R이 주어졌을 때 T와 비슷한 R의 부분집합 T'를 찾는다.")
    image("images/raw-target-schema.png", width=600)

    text("적용 예")
    text("- 언어 식별(영어와 나머지 언어).")
    text("- 품질 필터링(고품질과 저품질).")
    text("- 독성 필터링(독성이 없는 자료와 있는 자료).")

    text("필터링 알고리즘에 필요한 성질")
    text("- 목표 데이터로부터 일반화해야 한다. T와 T'가 같은 자료일 필요는 없다.")
    text("- 방대한 R 전체에 적용해야 하므로 매우 빨라야 한다.")

    text("데이터 선택에 관한 조사 논문 "), link("https://arxiv.org/abs/2402.16827")

    text("일반적인 방법: T와 R이 주어지면 T와 유사한 R의 부분집합을 찾는다.")
    text("1. R과 T로 모델을 추정하고 점수 함수를 얻는다.")
    text("2. R의 예제를 점수에 따라 선별한다.")

    text("분류 방법")
    text("- T에 대한 생성 모델(KenLM): score(x) = p_T(x).")
    text("- 간단한 분류기(fastText): score(x) = p(T | x).")
    text("score(x) >= threshold인 예제 x를 남긴다. 확률적으로 선택할 수도 있다.")

    text("모델을 이용해 필터링할까?")
    text("- C4, Gopher, RefinedWeb, FineWeb, Dolma 등은 의도적으로 모델 기반 필터링을 사용하지 않는다.")
    text("- GPT-3, LLaMA, DCLM 등은 모델 기반 필터링을 사용한다. 이 방식이 일반화되고 있다.")

    text("언어 식별")
    text("- 목표는 영어 등 특정 언어의 텍스트를 찾는 것이다.")
    text("- fastText 언어 식별 "), article_link("https://fasttext.cc/docs/en/language-identification.html")
    text("- 이미 학습된 분류기를 사용할 수 있다.")
    text("- 176개 언어를 지원한다.")
    text("- 위키백과, 번역 사이트 Tatoeba, 동남유럽 뉴스 사이트 SETimes 등 다국어 자료로 학습했다.")
    text("- Dolma는 p(English) >= 0.5인 페이지를 남긴다. "), link(dolma_2024)

    text("OpenMathText "), link("https://arxiv.org/pdf/2310.06786")
    text("- 목표: Common Crawl에서 대규모 수학 텍스트 말뭉치를 선별한다.")
    text("- LaTeX 명령 포함 여부 등 규칙으로 필터링한다.")
    text("- ProofPile로 학습한 KenLM의 퍼플렉서티가 15,000 미만이면 남긴다.")
    text("- 수학 문서를 예측하는 fastText 분류기를 학습했다. 수식이 있으면 임계값 0.17, 없으면 0.8을 쓴다.")
    text("- 결과는 147억 토큰이다. 이것으로 학습한 14억 매개변수 모델은 20배 많은 데이터로 학습한 모델보다 성능이 높았다.")

    text("GPT-3 "), link("https://arxiv.org/pdf/2005.14165")  # 부록 A
    text("- 양성: 위키백과, WebText2, Books1, Books2에서 뽑은 표본.")
    text("- 음성: Common Crawl에서 뽑은 표본.")
    text("단어 특징을 사용하는 선형 분류기를 학습한다. "), article_link("https://spark.apache.org/docs/latest/ml-features#tokenizer")
    text("점수에 따라 문서를 확률적으로 남긴다.")
    def keep_document(score: float) -> bool:
        return np.random.pareto(9) > 1 - score

    text("LLaMA/RedPajama "), link("https://arxiv.org/pdf/2302.13971")
    text("- 양성: 위키백과가 **참고문헌으로 인용한 페이지**의 표본.")
    text("- 음성: Common Crawl에서 뽑은 표본.")
    text("- 양성으로 분류된 문서를 남긴다.")

    text("phi-1 "), link("https://arxiv.org/pdf/2306.11644")
    text("- 품질이 매우 높은 교과서 수준의 데이터로 작은 모델(15억 매개변수)을 학습한다는 방침이다.")
    text("- GPT-3.5(이후 GPT-4)의 합성 데이터와 필터링한 데이터를 함께 사용한다.")
    R = "Python subset of the Stack"   # 원시 데이터
    prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"
    T = "Use GPT-4 with this prompt to classify 100K subset of R to get positive examples"
    text("- 사전학습한 코드 생성 모델의 출력 임베딩을 이용해 T에서 랜덤 포리스트 분류기를 학습한다.")
    text("- 분류기가 양성으로 판정한 R의 데이터를 선택한다.")
    text("[HumanEval](https://huggingface.co/datasets/openai_humaneval) 결과")
    text("- The Stack의 Python 부분집합으로 13억 매개변수 언어 모델을 학습했다. 96,000스텝 후 성능은 12.19%였다.")
    text("- 새로 필터링한 부분집합으로 같은 규모 모델을 학습했다. 36,000스텝 만에 17.68%로 더 좋은 결과를 냈다.")

    text("Dolma의 독성 필터링 "), link(dolma_2024)
    text("- 데이터셋: Jigsaw Toxic Comments(2018). "), link(title="데이터셋", url="https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge")
    text("- 온라인에서 더 나은 토론을 할 수 있도록 돕는 프로젝트다. "), article_link("https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/46064")
    text("- 위키백과 토론 문서의 댓글에 {toxic, severe_toxic, obscene, threat, insult, identity_hate} 레이블을 붙였다.")

    text("학습 규모에 따른 필터링 효과")
    text("- 모든 경우에 적합한 단 하나의 임계값은 없다.")
    text("- 오래 학습할 때는 품질이 조금 낮더라도 더 많은 데이터가 필요하다.")
    text("- 짧게 학습할 때는 양이 적더라도 품질이 높은 데이터를 원한다.")
    image("images/data-filtering-scale.png", width=800)

    text("정리")
    text("- 좋은 모델을 만들려면 필터링이 중요하다.")
    text("- 원하는 품질의 목표 데이터를 정의하고 그 기준을 원시 데이터로 일반화한다.")
    

def deduplication():
    text("중복의 두 유형")
    text("- 완전히 같은 중복: 미러 사이트, GitHub 포크 등. "), link(title="프로젝트 구텐베르크 미러", url="https://www.gutenberg.org/MIRRORS.ALL")
    text("- 유사 중복: 토큰 몇 개만 다른 텍스트.")

    text("유사 중복의 예")
    text("- 서비스 약관과 라이선스. "), link(title="MIT 라이선스", url="https://opensource.org/license/mit")
    text("- 복사해서 붙이거나 템플릿으로 생성한 정형적인 글. "), image("https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png", width=600)
    text("- 복사 과정에서 서식만 조금 달라진 텍스트.")

    text("C4에서 61,036번 반복된 상품 설명")
    text("'“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding.  believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!")
    link(title="예시 페이지", url="https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG")

    text("학습 데이터의 중복 제거가 언어 모델을 개선한다는 연구 "), link("https://arxiv.org/pdf/2107.06499")
    text("- 토큰 수가 줄어 더 효율적으로 학습한다.")
    text("- 암기를 줄여 저작권·개인정보 문제를 완화할 수 있다.")

    text("설계 시 선택할 사항")
    text("1. 무엇을 항목 하나로 볼까? 문장, 문단, 문서 중 무엇인가?")
    text("2. 완전 일치, 일부 항목의 공통 여부, 공통 항목의 비율 중 어떤 기준으로 비교할까?")
    text("3. 일치하면 모두 제거할까, 하나만 남길까?")

    text("어려운 점")
    text("- 중복 제거는 항목을 다른 항목과 비교하는 작업이다.")
    text("- 큰 데이터셋에서는 선형 시간 알고리즘이 필요하다.")

    hash_functions()
    exact_deduplication()
    jaccard_minhash()
    locality_sensitive_hashing()


def hash_functions():
    text("- 해시 함수 h는 항목을 해시값(정수 또는 문자열)으로 대응시킨다.")
    text("- 해시값은 원래 항목보다 훨씬 작다.")
    text("- 해시 충돌: x ≠ y인데 h(x) = h(y)인 경우다.")

    text("효율과 충돌 저항성의 관계 "),  article_link("https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed")
    text("- SHA-256 같은 암호학적 해시 함수는 충돌 저항성이 높고 느리다. 비트코인 등에 사용한다.")
    text("- DJB2, MurmurHash, CityHash는 충돌 저항성을 제공하지 않지만 빠르다. 해시 테이블 등에 사용한다.")

    text("여기서는 MurmurHash를 사용한다.")
    h = mmh3.hash("hello")  # @inspect h


def exact_deduplication():
    text("**간단한 예**")
    text("1. 항목은 문자열이다.")
    text("2. 완전히 같은 항목을 찾는다.")
    text("3. 하나만 남기고 제거한다.")

    # 원래 항목
    items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]  # @inspect items

    # 해시를 계산하고 같은 해시값에 대응하는 항목 목록을 만든다.
    hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

    # 그룹마다 항목 하나를 남긴다.
    deduped_items = [next(group) for h, group in hash_items]  # @inspect deduped_items

    text("- 장점: 단순하고 판정 기준이 명확하며 정밀도가 높다.")
    text("- 단점: 유사 중복은 제거하지 않는다.")
    text("- 이 코드는 MapReduce 형식이므로 병렬화하고 규모를 늘리기 쉽다.")

    text("**C4** "), link("https://arxiv.org/pdf/1910.10683v4")
    text("1. 항목은 연속된 세 문장이다.")
    text("2. 완전히 일치하는 구간을 찾는다.")
    text("3. 하나만 남기고 제거한다.")
    text("문서 중간에서 세 문장을 제거하면 남은 부분의 문맥이 이어지지 않을 수 있다.")


def jaccard_minhash():
    text("이제 집합에 어떤 항목이 속하는지 근사적으로 판정하는 문제를 보자.")
    text("먼저 유사도 척도가 필요하다.")

    text("### 자카드 유사도")
    text("정의: Jaccard(A, B) = |A의 원소와 B의 원소의 교집합| / |A의 원소와 B의 원소의 합집합|")
    A = {"1", "2", "3", "4"}
    B = {"1", "2", "3", "5"}

    def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union
    jaccard = compute_jaccard(A, B)  # @inspect jaccard

    text("두 문서의 자카드 유사도가 임계값 이상이면 **유사 중복**으로 정의한다.")

    text("알고리즘의 과제는 선형 시간에 유사 중복을 찾는 것이다.")

    text("### MinHash")
    text("MinHash는 Pr[h(A) = h(B)] = Jaccard(A, B)가 성립하는 무작위 해시 함수 h다.")

    text("일반적으로는 서로 다른 항목에 서로 다른 해시값을 원한다.")
    text("여기서는 충돌 확률이 유사도에 따라 달라져야 한다.")

    def minhash(S: set[str], seed: int):
        return min(mmh3.hash(x, seed) for x in S)

    text("특성 행렬 표현")
    text("항목 | A | B", verbatim=True)
    text("1    | 1 | 1", verbatim=True)
    text("2    | 1 | 1", verbatim=True)
    text("3    | 1 | 1", verbatim=True)
    text("4    | 1 | 0", verbatim=True)
    text("5    | 0 | 1", verbatim=True)

    text("무작위 해시 함수가 항목의 순열을 정한다.")
    text("A와 B에서 각각 어떤 항목이 첫 번째가 되는지 본다.")
    text("각 항목이 첫 번째(최솟값)가 될 확률은 같다.")
    text("- 1, 2, 3 중 하나가 처음이면 A와 B의 첫 항목이 같다.")
    text("- 4, 5 중 하나가 처음이면 A와 B의 첫 항목이 다르다.")

    # MinHash가 자카드 유사도를 근사하는지 확인한다.
    n = 100  # 이 개수만큼 무작위 해시 함수를 생성한다.
    matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]  # @stepover
    estimated_jaccard = len([m for m in matches if m]) / len(matches)  # @inspect estimated_jaccard
    assert abs(estimated_jaccard - jaccard) < 0.01

    text("이제 항목을 해싱할 수 있다. 다만 한 번 충돌했다고 Jaccard(A, B) > threshold라고 할 수는 없다.")


def locality_sensitive_hashing():
    text("지역 민감 해싱(LSH) "), link(title="교재의 해당 장", url="http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf")

    text("MinHash 함수 하나만 사용해 해싱한다고 하자.")
    text("P[A와 B가 충돌] = Jaccard(A, B)")
    text("평균적으로는 더 비슷한 항목이 자주 충돌하지만 개별 결과에는 무작위성이 크다.")

    text("목표: Jaccard(A, B) > threshold일 때 A와 B가 충돌하도록 한다.")
    text("임계값 전후에서 충돌 확률이 더 급격히 달라지게 해야 한다.")

    text("방법: 해시 함수 n개를 쓴다.")
    text("각 밴드에 r개씩, b개 밴드로 나눈다(n = b * r).")

    n = 12      # 해시 함수 수
    b = 3       # 밴드 수
    r = 4       # 밴드당 해시 함수 수
    text("해시 함수")
    text("h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12", verbatim=True)

    text("*어느 한* 밴드에서 *모든* 해시 함수의 결과가 같으면 A와 B가 충돌한다.")
    text("밴드 안의 AND 조건과 밴드 사이의 OR 조건을 결합하면 임계값 전후가 더 뚜렷해진다.")

    text("Jaccard(A, B)가 주어졌을 때 A와 B의 충돌 확률은?")

    def get_prob_collision(sim, b, r):  # @inspect sim @inspect b @inspect r
        prob_match = sim ** r                        # 지정한 밴드에서 일치할 확률  @inspect prob_match
        prob_collision = 1 - (1 - prob_match) ** b   # 어느 한 밴드에서라도 일치할 확률  @inspect prob_collision
        return prob_collision

    text("**예**")
    prob_collision = get_prob_collision(sim=0.8, b=5, r=10)  # @inspect prob_collision
    image("https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png", width=600)


    sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    probs = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}  # @inspect probs @stepover

    text("r을 늘리면 곡선이 가팔라지고 오른쪽으로 이동해 일치하기 어려워진다.")
    probs = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}  # @inspect probs @stepover

    text("b를 늘리면 곡선이 왼쪽으로 이동해 일치하기 쉬워진다.")
    probs = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}  # @inspect probs @stepover
    image("https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png", width=600)

    text("설정 예 "), link("https://arxiv.org/pdf/2107.06499"), text(": n = 9000, b = 20, r = 450")
    b = 20
    r = 450
    text("변화가 급격해지는 임계값은 어디일까?")
    threshold = (1 / b) ** (1 / r)  # @inspect threshold

    text("지정한 밴드에서 일치할 확률")
    prob_match = (1 / b)  # @inspect prob_match
    text("A와 B의 충돌 확률이 상수(약 1-1/e)가 되는 지점")
    prob_collision = 1 - (1 - 1 / b) ** b  #  @inspect prob_collision


def billion(x):
    return x * 10**9

def trillion(x):
    return x * 10**12


def data_mixing():
    text("언어 모델은 여러 출처의 데이터로 학습한다.")

    text("Marin의 데이터셋 "), link(title="토큰 뷰어", url="https://huggingface.co/spaces/marin-community/token-count-viewer")
    image("images/marin-token-viewer.png", width=800)

    text("The Pile "), link(the_pile_2020)
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png", width=600)
    text("데이터 출처별로 어떤 확률 분포를 쓸까?")

    text("예")
    sources = {"Wikipedia", "CC", "GitHub"}
    p = {"Wikipedia": 0.3, "CC": 0.5, "GitHub": 0.2}  # 가능한 데이터 혼합 비율의 한 예

    text("비교 기준")
    text("- 직관: 경험에 따라 p(s)를 직접 설정한다. 흔히 쓰는 방법이다.")
    text("- 균등 샘플링: 출처를 같은 확률로 뽑는다(p(s) ∝ 1).")
    text("- 비례 혼합: 출처에 있는 토큰 수에 비례해 뽑는다(p(s) ∝ num_tokens(s)).")

    text("직관적으로는 품질이 높은 출처에 높은 가중치를 주고 싶다.")
    text("하지만 다음도 고려해야 한다.")
    text("1. 문학, 코드, 논문처럼 단순 비교하기 어려운 출처 간의 다양성을 확보해야 한다.")
    text("2. 출처별 데이터는 유한하다. 작은 출처에 가중치를 지나치게 높게 주면 같은 데이터를 여러 에폭 반복하게 된다.")
    
    text("두 번째 점은 중요하고 주의해서 살펴봐야 한다.")
    text("예")
    source_token_counts = {
        "low": trillion(10),  # 10조 토큰(풍부) @stepover
        "high": billion(10),  # 100억 토큰(희소) @stepover
    }
    p = {"low": 0.5, "high": 0.5}  # 단순한 데이터 혼합 비율
    train_tokens = trillion(1)  # 1조 토큰 학습 @stepover
    low_num_epochs = (p["low"] * train_tokens) / source_token_counts["low"]  # @inspect low_num_epochs
    high_num_epochs = (p["high"] * train_tokens) / source_token_counts["high"]  # @inspect high_num_epochs
    text("품질이 높은 데이터를 50에폭 반복하면 과적합할 수 있다.")

    text("UniMax "), link("https://arxiv.org/abs/2304.09151")
    text("- 문제: 다국어 모델에서 언어별 비중을 조정한다.")
    text("- 기존 방법은 균등 혼합과 비례 혼합의 중간이다. α ∈ [0, 1]에 대해 p(s) ∝ num_tokens(s)^α.")
    text("- 각 출처를 균등하게 뽑되, 어느 출처든 에폭 수에 엄격한 **상한** C를 둔다.")
    text("- 구체적으로 모든 출처 s에 대해 p(s) * num_training_tokens ≤ C다.")

    text("회귀를 이용한 혼합 비율 선택 "), link("https://arxiv.org/abs/2407.01492"), link("https://arxiv.org/pdf/2602.12237")
    image("images/regmix.png", width=700)
    text("- 혼합 비율 `p`에 대한 분포를 정의한다(예: 디리클레 분포). ")
    text("- 선형 회귀, 그래디언트 부스팅 트리 등 회귀 방법을 정한다.")
    text("- 하류 평가로 목표를 정의한다. 평가에 과적합하지 않도록 주의한다.")
    text("- 작은 규모와 큰 규모의 결과가 다를 수 있다. 비용과 정확도를 고려해야 한다.")
    image("images/data-mixing-methods.png", width=700)
    text("기대 1. 최솟값을 주는 지점에서 회귀 모델의 예측이 정확하다.")
    text("기대 2. 작은 규모의 최적 혼합 비율이 큰 규모에도 적용된다.")

    text("하지만 규모에 따라 달라지는 효과가 적어도 하나 있다.")
    source_token_counts = {
        "low": trillion(10),  # 10조 토큰(풍부) @stepover
        "high": billion(10),  # 100억 토큰(희소) @stepover
    }
    text("- 작은 모델을 적은 토큰으로 학습할 때는 다음과 같다.")
    p = {"low": 0.1, "high": 0.9}  # 품질이 높은 데이터의 확률 질량을 늘린다.
    text("- 큰 모델에 이 비율을 그대로 쓰면 고품질 데이터를 너무 많이 반복해 과적합한다.")

    text("여러 에폭 학습하는 효과 모사하기 "), link("https://arxiv.org/pdf/2501.11747")
    text("- 작은 규모의 실험 조건을 큰 규모와 비슷하게 만든다. 이 강의의 여러 주제에서 쓰인 접근이다.")
    text("- 각 출처에서 같은 비율로 데이터를 줄인다.")
    small_run_tokens = billion(10)  # @stepover
    large_run_tokens = trillion(1)  # @stepover
    ratio = small_run_tokens / large_run_tokens  # @inspect ratio
    downsampled_source_token_counts = {s: count * ratio for s, count in source_token_counts.items()}  # @inspect downsampled_source_token_counts
    text("- 이런 혼합 데이터에서는 같은 자료를 지나치게 반복하는 모델의 성능이 낮게 나온다.")
    text("- 따라서 최적의 혼합 비율이 더 균형 있게 정해진다.")
    p = {"low": 0.7, "high": 0.3}  # 품질이 높은 데이터의 확률 질량을 늘린다.

    text("정리")
    text("- 문제: 위키백과, 일반 웹 문서, 코드 등 데이터 출처의 가중치를 어떻게 정할까?")
    text("- 회귀 기반 혼합: 작은 규모에서 혼합 비율과 손실의 관계를 추정하고 최적화한다. 스케일링 법칙과 비슷하다.")
    text("- 에폭 수와 과적합을 고려한다. 에폭 수에 상한을 두거나 작은 규모에서 반복 효과를 모사할 수 있다.")


def post_training_data():
    text("데이터 수집 절차")
    text("1. 사용할 환경을 정한다.")
    text("2. 과제와 프롬프트 집합을 정한다.")
    text("3. 성능이 높은 교사 모델의 응답을 수집한다.")

    text("OpenThoughts "), link("https://arxiv.org/abs/2506.04178")
    text("- QwQ-32B를 교사로 써서 예제 120만 개를 만들었다.")
    text("- 질문은 Stack Exchange, NuminaMath, Chemistry 등 사람이 작성했거나 합성한 자료 27곳에서 가져왔다.")
    image("images/openthoughts-sources.png", width=500)
    text("- 프롬프트당 응답을 여러 개(16개) 샘플링하면 도움이 됐다.")
    text("- 성능이 높은 모델이 항상 더 좋은 교사는 아니다. 이 연구에서는 QwQ-32B가 DeepSeek-R1보다 더 좋은 교사였다.")
    text("- 정답을 기준으로 필터링하는 것은 도움이 되지 않았다.")
    text("- OpenMath-2-Math처럼 작고 품질이 높은 출처가 크고 다양한 출처보다 좋았다.")
    image("images/openthoughts-pipeline.png", width=600)

    text("SWE-smith "), link("https://arxiv.org/abs/2504.21798")
    image("images/swe-smith.png", width=500)
    text("- 저장소가 주어지면 언어 모델이 코드에 버그를 넣어 수정 과제를 만든다.")
    text("- GitHub 저장소 128개에서 과제 5만 개를 얻었다.")

    text("SWE-Zero "), link("https://arxiv.org/abs/2604.01496")
    text("- 소프트웨어 공학 과제는 수학·코딩 대회와 달리 의존성이 많다.")
    text("- Docker 이미지 수천 개를 준비하는 일은 인프라 부담이 크다.")
    text("- 성능이 높은 모델은 실행 결과를 돌려받지 않아도 많은 과제를 풀 수 있었다.")
    image("images/swezero-noexec.png", width=600)
    text("성능이 높은 모델은 코드의 의미와 동작을 예측하는 내부 ‘세계 모델’을 갖고 있다는 설명이다.")
    text("- SWE-Zero: 저장소별 실행이 필요 없는 에이전트 작업 기록 30만 개.")
    text("- GitHub PR 15만 개.")
    text("- OpenHands 실행 구조를 사용했다. 에이전트가 이후 커밋에서 답을 찾는 일을 막기 위해 미래 커밋을 제거했다.")
    image("images/swezero-prompt.png", width=600)
    text("- Qwen3-Coder-480B에서 증류하고, 가능하면 실제로 실행해 필터링했다.")
    text("- SWE-Hero: 실행 피드백이 필요한 작업 기록 1만 3,000개.")
    image("images/swezero-results.png", width=700)

    text("SWE-rebench "), link("https://arxiv.org/pdf/2505.20411")
    text("- GitHub 저장소 3,400개에서 만든 상호작용형 Python 소프트웨어 공학 과제 2만 1,000개.")
    text("- GitHub와 GitHub Archive에서 PR 45만 개를 수집했다.")
    text("- Qwen 2.5-72B-Instruct로 의존성을 설치하고 PR 품질을 평가했다.")
    image("images/swe-rebench.png", width=600)

    text("SWE-ZERO-12M-trajectories "), link(title="데이터", url="https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories")
    text("- SWE-Zero를 에이전트 작업 기록 1,200만 개로 확장했다.")
    text("- SWE-rebench-v2의 실행 가능한 과제 3만 2,000개와 실행할 수 없는 과제 12만 개를 사용했다.")
    text("- 매우 작은 모델 mini-coder-1.7b(50.4 pass@100)와 mini-swe-agent 실행 구조를 썼다.")
    text("- [예시](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories/viewer/default/train?row=5&conversation-viewer=0)")

    text("정리")
    text("- 프롬프트 생성 방식으로 완전 합성, 반합성(실제 환경과 합성 과제), 실제 사례(GitHub PR)가 있다.")
    text("- 응답은 성능이 높으면서 교사 역할에도 적합한 모델에서 수집한다.")
    text("- 코드 실행 환경을 마련하는 데 손이 많이 간다.")
    text("- 필터링을 비롯해 처리할 세부 사항이 많다.")


if __name__ == "__main__":
    main()
