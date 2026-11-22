from edtrace import text, image, link
from lecture_util import article_link
from references import dclm_2024, nemotron_cc_2024, olmo_2_2025, llama_3_2024, gpt2_2019, openwebtext_2019, gopher_2021, alpaca_2023


def main():
    text("## 13강. 데이터 I")
    text("지난 강의에서는 *데이터가 주어졌을 때* 모델을 학습하는 방법을 다뤘다.")
    text("이번 강의와 다음 강의에서는 *어떤 데이터*로 학습할지 살펴본다.")

    motivation()

    # 데이터의 기원
    raw_sources()    # 데이터는 어디에서 올까?
    copyright()      # 어떤 데이터를 쓸 수 있을까?

    # 데이터 출처
    common_crawl()   # 웹 크롤링
    wikipedia()      # 일반 지식
    github()         # 코드
    arxiv()          # 연구 논문

    # 여러 모델이 사용한 데이터
    bert()                # 위키백과와 도서(BERT 학습, 2019)
    gpt2_webtext()        # Reddit 링크가 가리키는 페이지(GPT-2 학습, 2019)
    ccnet()               # 위키백과를 기준으로 Common Crawl 필터링(2019)
    t5_c4()               # 규칙 기반 필터링(T5 학습, 2019)

    gpt3()                # Common Crawl, 위키백과, 도서(GPT-3 학습, 2020)
    the_pile()            # 다양한 출처(GPT-J, GPT-NeoX 등 학습, 2021)
    gopher_massivetext()  # 규칙 기반 필터링(Gopher 학습, 2021)
    llama()               # Common Crawl, CCNet, Stack Exchange 등(LLaMA 학습, 2022)
    refinedweb()          # Common Crawl(Falcon 학습에 사용, 2023)
    dolma()               # 다양한 출처(2024)
    dclm()                # 성능이 좋은 품질 분류기로 필터링(2024)
    nemotron_cc()         # 대량의 토큰(2024)
    the_stack()           # 코드 데이터셋
    common_pile()         # 사용 허가를 받은 데이터

    text("### 정리")
    text("- 데이터를 얻으려면 직접 수집하고 가공해야 한다.")
    text("- 운영 중인 서비스 → 원시 데이터 → 처리된 데이터(변환·필터링·중복 제거).")
    text("- 언어 모델 간 차이를 결정하는 중요한 요소는 데이터다.")
    text("- 저작권과 개인정보 등 법적·윤리적 문제가 있다.")
    text("- 이 과정은 경험적 규칙에 의존하는 부분이 많아 개선할 여지가 많다.")


def motivation():
    text("언어 모델을 학습할 때 가장 신경 써야 할 것은 **데이터**다.")

    text("기업들이 어떤 정보를 공개하는지 살펴보자.")
    text("가중치를 공개한 모델(예: Llama 3 "), link(llama_3_2024), text("은 모델 구조를 전부 공개한다.")
    text("학습 절차까지 공개하기도 한다.")
    text("반면 데이터에 대한 정보는 거의 없다.")
    image("images/llama3-data.png", width=700)
    
    text("데이터를 공개하지 않는 이유")
    text("1. 기업 간 경쟁.")
    text("2. 저작권에 대한 법적 책임.")

    text("- 기반 모델 이전에는 지도학습용 데이터에 사람이 많은 양의 정답을 붙이는 일이 데이터 작업의 대부분이었다.")
    text("- 지금은 주석 작업이 줄었지만 선별과 정제에 여전히 많은 일이 필요하다.")
    text("- 드물지만 종류가 많은 사례까지 다뤄야 하므로 데이터 작업량은 구조·시스템 개발과 달리 사람의 노력에 따라 증가한다.")

    text("학습 단계")
    text("1. 사전학습: 웹 문서 같은 원시 텍스트로 학습한다.")
    text("2. 중간 학습: 품질이 높은 데이터로 더 학습해 능력을 높인다.")
    text("3. 후학습: 대화 기록으로 학습하거나 강화학습을 한다.")
    text("실제로는 단계 간 구분이 뚜렷하지 않고 단계가 더 많을 수도 있다.")
    text("학습 과정을 전체적으로 보면")
    text("상대적으로 품질이 낮은 대량의 데이터에서")
    text("품질이 높은 소량의 데이터로 이동한다.")

    text("용어")
    text("- 기본 모델: 사전학습과 중간 학습을 마친 모델.")
    text("- 지시·대화 모델: 후학습까지 마친 모델.")
    text("기본 모델을 따로 공개하지 않는 경우도 늘고 있다. 예를 들어 Qwen3.5-397B-A17B는 지시 모델이다.")

    text("예: AI2의 OLMo "), link(olmo_2_2025)
    text("1. **사전학습**")
    image("images/olmo2-pretraining.png", width=600)
    text("2. **중간 학습**")
    image("images/olmo2-dolmino.png", width=600)
    text("3. **후학습** "), link("https://arxiv.org/pdf/2411.15124")
    image("images/tulu.png", width=600)

    text("이 데이터셋들은 무엇이며 어떻게 선택하고 처리할까?")


def raw_sources():
    text("‘언어 모델은 인터넷 전체를 학습한다’는 말을 흔히 듣는다.")
    text("조금 더 정확히 말하면 ~인터넷~ 공개된 월드 와이드 웹이다.")
    text("하지만 이 표현도 완전히 정확하지 않다.")

    text("먼저 웹은 접속할 수 있는 여러 서버로 이루어져 있다.")
    text("`$ curl https://cs336.stanford.edu/`")

    text("실행 중인 서버 그 자체를 학습 데이터로 쓸 수는 없다.")
    text("**크롤러**의 동작")
    text("- 초기 주소 집합에서 시작해 웹페이지를 찾는다.")
    text("- 찾은 웹페이지를 내려받는다.")

    text("하지만 모든 웹페이지를 내려받아 학습할 수는 없다.")

    text("동적 콘텐츠")
    text("- 오늘날 웹사이트 중 상당수는 앱이다.")
    text("- URL은 그대로일 수 있다.")
    text("- 내용을 보려면 버튼을 누르거나 양식을 제출해야 한다.")
    text("- 예: Discord, wandb.")

    text("인증")
    text("- 계정으로 로그인해야 하거나 요금을 내야 할 때가 있다.")
    text("- 예: Facebook, X, LinkedIn, 뉴욕타임스. 접근이 제한된 서비스 내부에 방대한 콘텐츠가 있다.")

    text("기술적 제한")
    text("- `robots.txt`에서 일부 콘텐츠의 수집을 허용하지 않을 수 있다([예](https://www.nytimes.com/robots.txt)). 준수는 자발적이다.")
    text("- Cloudflare 등으로 봇을 감지해 차단하거나 CAPTCHA를 표시하는 사이트가 있다.")
    text("- 특정 IP 주소나 국가의 접속을 차단하기도 한다.")
    text("- 요청 횟수 제한이 있을 수 있다.")
    
    text("법적 제한")
    text("- 서비스 이용약관에서 봇으로 다운로드하는 것을 금지할 수 있다.")
    text("- 학습을 위해 웹페이지를 복사할 권한이 없을 수 있다.")

    text("데이터 수집 동의의 감소 "), link("https://arxiv.org/abs/2407.14933")
    text("- C4, RefinedWeb, Dolma에 포함된 URL의 robots.txt와 이용약관 제한을 조사했다.")
    text("- 시간이 지나면서 제한이 늘어났다.")
    image("images/decline-consent.png", width=700)

    text("크롤러가 수집 규칙을 지키지 않는 경우")
    image("images/anthropic-crawling.png", width=500)
    text("- 이용약관, robots.txt, 서버 부하를 고려해야 한다. 서버 부하는 서비스를 느리게 하고 운영 비용을 높인다.")
    text("- 저작권 문제도 있다. 뒤에서 다룬다.")

    text("섀도 라이브러리 "), article_link("https://en.wikipedia.org/wiki/Shadow_library")
    text("- 기술적으로는 웹의 일부다.")
    text("- 예: Library Genesis(LibGen), Z-Library, Anna’s Archive, Sci-Hub.")
    text("- 저작권을 고려하지 않고 Elsevier 같은 출판사의 유료 접근 제한을 우회한다.")
    text("- 삭제 명령과 소송을 받았으며 여러 국가에서 차단됐다.")
    text("- 여러 국가에 서버를 두고 제한을 피해 운영하는 경우가 많다.")
    text("- 자유롭게 이용할 수 있어야 할 자료를 누구나 볼 수 있게 한 것이라는 주장도 있다.")
    text("- 법적인 관점에서는 무단 복제와 저작권 침해다.")
    text("- LibGen에는 약 400만 권의 책(2019년), Sci-Hub에는 약 8,800만 편의 논문(2022년)이 있다.")

    text("정리")
    text("- 인터넷에는 방대한 데이터가 있다.")
    text("- 실제로 접근할 수 있는 데이터에는 기술적·법적 제한이 많다.")


def copyright():
    text("학습에 사용할 수 있는 데이터는 무엇인가?")

    text("### 지식재산권법")
    text("- 목적은 지적 창작물을 만들 *유인*을 제공하는 것이다.")
    text("- 지식재산권에는 저작권, 특허, 상표, 영업 비밀이 있다.")

    text("**저작권법**")
    text("- 영국의 1709년 앤 여왕법(Statute of Anne)으로 거슬러 올라간다. 정부와 법원이 처음으로 규율했다. "), article_link("https://en.wikipedia.org/wiki/Statute_of_Anne")
    text("- 미국에서 다루는 법은 1976년 저작권법이다. "), article_link("https://en.wikipedia.org/wiki/Copyright_Act_of_1976")
    text("- 보호 대상은 *현재 알려져 있거나 이후 개발될 유형의 표현 매체에 고정된 독창적인 저작물로, 직접 또는 기계·장치의 도움으로 인지·복제하거나 달리 전달할 수 있는 것*이다.")

    text("- 전화번호부 같은 모음집에는 선택·배열에 창작성이 있어야 저작권이 인정된다.")
    text("- 저작권은 표현을 보호한다. 예를 들어 퀵소트라는 아이디어 자체는 보호 대상이 아니다.")

    text("- 보호 범위는 ‘출판된’ 저작물(1909)에서 ‘고정된’ 저작물(1976)로 확대됐다.")
    text("- 특허와 달리 저작권 보호에 등록이 필수는 아니다.")
    text("- 보호 기준은 매우 낮다. 개인 웹사이트도 저작권의 보호를 받는다는 예를 들 수 있다.")

    text("- 창작자가 저작권 침해 소송을 제기하려면 등록이 필요하다.")
    text("- 등록 비용은 65달러다. "), article_link("https://www.copyright.gov/about/fees.html")
    text("- 강의에서 든 보호 기간은 75년이다. 저작권이 만료되면 퍼블릭 도메인이 된다. 예로 셰익스피어·베토벤의 작품과 프로젝트 구텐베르크의 자료 대부분이 있다.")

    text("정리: *인터넷 자료는 거의 모두 저작권의 적용을 받는다.*")

    text("저작권이 있는 작품을 사용하는 방법")
    text("1. 이용 허락을 받는다.")
    text("2. 공정 이용 조항을 근거로 삼는다.")

    text("### 라이선스")
    text("- 계약법상 라이선스는 권리 허락자가 이용자에게 부여한다.")
    text("- 실질적으로 ‘소송을 제기하지 않겠다는 약속’으로 볼 수 있다.")

    text("- 크리에이티브 커먼즈 라이선스를 통해 저작권이 있는 저작물을 자유롭게 배포할 수 있다.")
    text("- 예: 위키백과, OpenCourseWare, Khan Academy, Free Music Archive, Flickr 사진 3억 700만 장, MusicBrainz 이미지 3,900만 장, YouTube 영상 1,000만 편.")
    text("- 퍼블릭 도메인과 기존 저작권 제도 사이를 연결하기 위해 Lessig와 Eldred가 2001년에 만들었다.")

    text("여러 모델 개발사가 기반 모델 학습을 위해 데이터 이용 계약을 맺는다.")
    text("- Google과 Reddit "), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- OpenAI와 Shutterstock "), article_link("https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year")
    text("- OpenAI와 Stack Exchange "), article_link("https://stackoverflow.co/company/press/archive/openai-partnership")

    text("**공정 이용(제107조)**")
    text("공정 이용 여부를 판단하는 네 가지 요소")
    text("1. 이용 목적과 성격. 상업적 이용보다 교육적 이용, 단순 복제보다 변형적 이용에 유리하다.")
    text("2. 원저작물의 성격. 허구보다 사실에 기반한 자료, 창작성이 높은 자료보다 낮은 자료의 이용에 유리하다.")
    text("3. 이용한 부분의 양과 중요성. 작품 전체보다 짧은 일부를 쓰는 데 유리하다.")
    text("4. 이용이 원작의 현재 또는 잠재 시장에 미치는 영향.")

    text("공정 이용 사례")
    text("- 영화를 보고 줄거리를 요약한다.")
    text("- 코드라는 표현을 복사하는 대신 알고리즘이라는 아이디어를 다시 구현한다.")
    text("- Google Books가 책을 색인하고 일부를 표시한 사례(Authors Guild 대 Google, 2002~2013).")

    text("저작권 문제는 글자 그대로 외웠는지에 그치지 않는다.")
    text("- 줄거리와 등장인물(예: 해리 포터)도 보호 대상이 될 수 있다.")
    text("- 풍자를 위해 모방하는 패러디는 공정 이용일 가능성이 있다.")
    text("저작권에서는 의미와 경제적 영향도 고려한다.")

    text("언어 모델에 관해 고려할 점")
    text("- 데이터를 복사하는 학습의 첫 단계부터, 그 뒤에 아무 작업도 하지 않더라도 침해 문제가 생긴다.")
    text("- 모델 학습은 단순 복사·붙여넣기와 거리가 멀기 때문에 변형적 이용으로 볼 수 있다는 주장이다.")
    text("- 모델은 마법사 같은 일반적 아이디어를 다뤄야 하고 해리 포터 같은 구체적 표현을 복제해서는 안 된다는 관점이다.")
    text("- 저작권과 별개로 언어 모델은 작가와 예술가의 시장에 영향을 줄 수 있다.")

    text("**서비스 이용약관**")
    text("- 이용 허락이 있거나 공정 이용을 주장할 수 있어도 서비스 약관이 추가 제한을 둘 수 있다.")
    text("- 예로 든 YouTube 약관에서는 영상에 크리에이티브 커먼즈 라이선스가 있어도 다운로드를 금지한다.")

    text("### 소송")
    text("뉴욕타임스 대 OpenAI(2023)")
    text("- 주장: 뉴욕타임스 기사로 학습하고 기사를 재현했다.")

    text("작가들(Bartz, Graeber 등) 대 Anthropic(2024)")
    text("- 주장: 책 수백만 권을 무단 복제했고 원고들의 책으로 학습했다.")
    text("- 약식 판결(2025): 이 사건에서 원고들의 작품으로 학습한 것은 공정 이용으로 판단했다.")
    text("- 그러나 무단 복제본을 취득하는 일은 학습에 쓰지 않았더라도 공정 이용이 아니라고 판단했다.")
    text("- Anthropic은 책을 구매해 스캔하기도 했다. 이는 공정 이용으로 보았지만 앞선 복제 이후였다.")
    text("- 결과: Anthropic은 작가들에게 15억 달러를 지급하는 데 합의했다.")

    text("작가들(Kadrey, Silverman 등) 대 Meta ")
    text("- 주장: Llama 논문에 나타난 것처럼 원고들의 책을 학습에 사용했다.")
    text("- 약식 판결(2025): 이 사건의 구체적 사정에서 책을 사용한 학습은 공정 이용으로 판단했다. "), article_link("https://techcrunch.com/2025/06/25/federal-judge-sides-with-meta-in-lawsuit-over-training-ai-models-on-copyrighted-books/")
    text("- 토렌트로 책을 전송했다는 주장은 강의 당시 계류 중이었다.")

    text("정리")
    text("- 강의에서 다룬 구체적인 사건에서는 학습을 공정 이용으로 판단했다. 일반적으로도 그러한지는 불확실하다.")
    text("- 서적의 불법 복제는 위법이다.")
    text("- 여전히 활발하게 변하는 분야다.")


def common_crawl():
    text("[Common Crawl](https://commoncrawl.org/)은 2007년 설립한 비영리 단체다.")

    text("통계")
    text("- 대략 한 달에 한 번 웹을 수집해 30억~50억 페이지를 추가한다.")
    text("- 회차 사이에 겹치는 페이지가 있지만 다양한 자료를 모으려 한다.")
    text("- 지금까지 수집한 페이지는 3,000억 개다.")

    text("- URL 수를 추정하기는 어렵지만 수십억 개 규모다.")
    text("- Google의 검색 색인은 적어도 100 PB다. "), article_link("https://www.google.com/search/howsearchworks/how-search-works/organizing-information/")
    text("- [2026년 4월 수집분](https://commoncrawl.org/blog/april-2026-crawl-archive-now-available)은 21억 9,000만 페이지(372.2 TB)다.")

    text("크롤링에는 Apache Nutch를 사용한다. "), article_link("https://blog.commoncrawl.org/blog/common-crawl-move-to-nutch")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/WebCrawlerArchitecture.svg/330px-WebCrawlerArchitecture.svg.png", width=400)
    text("- 적어도 수억 개의 초기 URL에서 시작한다. "), article_link("https://commoncrawl.org/blog/march-2018-crawl-archive-now-available")
    text("- 대기열에서 URL 하나를 꺼내 페이지를 내려받고 하이퍼링크를 대기열에 추가한다.")

    text("수집 정책 "), article_link("https://en.wikipedia.org/wiki/Web_crawler")
    text("- 선택 정책: 어떤 페이지를 내려받을까?")
    text("- 예의 정책: robots.txt를 지키고 서버에 과부하를 주지 않는다.")
    text("- 재방문 정책: 페이지 변경 여부를 얼마나 자주 확인할까?")
    text("- URL은 동적으로 달라지고 여러 URL이 사실상 같은 내용을 가리킬 수 있다.")

    text("두 가지 형식")
    text("- WARC: 원시 HTTP 응답(예: HTML).")
    text("- WET: 텍스트로 변환한 결과. 변환 중 일부 정보가 사라진다.")

    text("HTML을 텍스트로 변환하기")
    text("- 변환 도구: [trafilatura](https://trafilatura.readthedocs.io/en/latest/), [resiliparse](https://resiliparse.chatnoir.eu/en/stable/).")
    text("- 변환 방식은 최종 언어 모델의 하류 과제 정확도에도 영향을 준다. "), link(dclm_2024)
    image("images/dclm-wet.png", width=300)


def wikipedia():
    text("이제 더 전문적인 자료 출처를 살펴보자.")

    text("[위키백과](https://www.wikipedia.org/): 자유롭게 이용할 수 있는 온라인 백과사전.")
    text("- [임의 문서](https://en.wikipedia.org/wiki/Special:Random)")
    text("- 2001년에 설립했다.")
    text("- 2026년 5월 기준 361개 언어판에 문서 6,700만 개가 있다. 영어·스페인어·독일어·프랑스어가 가장 흔하다. "), article_link("https://meta.wikimedia.org/wiki/Wikipedia")

    text("어떤 내용을 다루는가?")
    text("- 독자적인 연구나 개인의 생각을 싣지 않는다. 의견, 홍보, 개인 웹페이지 등은 대상에서 제외한다. "), article_link("https://en.wikipedia.org/wiki/Wikipedia:What_Wikipedia_is_not")
    text("- 신뢰할 만한 출처에서 상당한 비중으로 다뤘는지를 기준으로 등재 여부를 판단한다. "), article_link("https://en.wikipedia.org/wiki/Wikipedia:Notability")

    text("누가 작성하는가?")
    text("- 인터넷 사용자 누구나 편집할 수 있고 관리자가 훼손된 내용을 되돌린다.")
    text("- 소수의 기여자가 편집 대부분을 맡는다. 예를 들어 Steven Pruit은 500만 번 편집했다. "), article_link("https://en.wikipedia.org/wiki/Steven_Pruitt")
    text("- 몇 주마다 [데이터 덤프](https://dumps.wikimedia.org/enwiki/)를 배포하므로 직접 크롤링할 필요가 없다.")

    text("데이터 오염 공격 "), link("https://arxiv.org/pdf/2302.10149")
    text("- 주기적으로 배포되는 덤프에는 되돌리기 전의 악의적인 편집 내용이 포함될 수 있다.")
    text("- 연구 사례에서는 오염된 예제로 학습한 모델이 특정 표현(예: iPhone)을 부정적인 감정과 연결했다. "), link("https://arxiv.org/pdf/2010.12563")
    text("- 품질이 높은 출처에도 부적절한 내용이 포함될 수 있다.")


def github():
    text("코드는 프로그래밍 과제에 도움이 된다. 추론에도 도움이 된다고 알려져 있다.")

    text("[GitHub](https://github.com/):")
    text("- 2008년 코드 저장소 호스팅 서비스로 설립했고 2018년에 Microsoft가 인수했다.")
    text("- 2026년 5월 기준 GitHub에 4억 2,000만 개가 넘는 저장소가 있고 그중 2,800만 개가 공개 상태다. "), article_link("https://en.wikipedia.org/wiki/GitHub")
    text("- 저장소에는 디렉터리 구조, 커밋 이력, 이슈, 풀 리퀘스트, 댓글 등이 있다.")
    text("- 복사한 코드와 포크 등으로 중복이 많다.")
    text("- MIT·Apache 같은 허용적 라이선스가 있는 공개 저장소는 학습에 사용할 수 있다.")
    
    text("데이터의 두 유형")
    text("- 저장소: GitHub 웹페이지를 수집하는 대신 Git 프로토콜로 내려받는다.")
    text("- 메타데이터: GitHub API가 이슈, 풀 리퀘스트, 댓글 등을 제공한다. 이벤트 스트림의 시간대별 스냅숏은 [GitHub Archive](https://info.arxiv.org/help/bulk_data_s3.html)에 있다.")

    text("[Software Heritage](https://www.softwareheritage.org/):")
    text("- 소프트웨어를 수집하고 보존하기 위해 2016년에 설립한 비영리 단체다.")
    text("- 이슈·댓글 같은 메타데이터보다 저장소 자체에 집중한다.")
    text("- GitHub, GitLab, Bitbucket, PyPI 등의 자료를 모은다.")
    text("- 2026년 5월 기준 소스 파일 2,880만 개가 있다.")


def arxiv():
    text("[arXiv](https://arxiv.org/):")
    text("- 1991년부터 연구자들이 무료로 논문을 공유하고 읽을 수 있는 웹사이트다.")
    text("- 물리학부터 시작해 수학, 컴퓨터과학, 통계학 등을 다룬다.")
    text("- 제출물은 약 300만 건이다. "), article_link("https://arxiv.org/stats/monthly_submissions")
    text("- 제출물은 메타데이터, PDF, 선택 사항인 LaTeX 소스로 구성된다.")
    text("- 간단한 승인 절차가 있다. 동료 심사는 아니다.")
    text("- 저자가 모든 권리를 보유하거나 CC BY 같은 크리에이티브 커먼즈 라이선스를 선택한다.")
    text("- 제목·초록 등 메타데이터에는 CC0를 적용한다.")
    text("- [Amazon S3](https://info.arxiv.org/help/bulk_data_s3.html)에서 일괄 다운로드할 수 있어 크롤링할 필요가 없다.")


def bert():
    link("https://arxiv.org/pdf/1810.04805")

    text("BERT의 학습 데이터")
    text("- 위키백과")
    text("- 도서")
    books_corpus()

    text("- 시퀀스의 단위가 단일 문장이 아니라 문서라는 점이 중요하다.")
    text("- 비교: One Billion Word Benchmark(Chelba 외, 2013)는 기계 번역 자료의 문장으로 구성했다.")


def books_corpus():
    text("[Smashwords](https://www.smashwords.com/)")
    text("- 2008년에 설립했으며 누구나 전자책을 직접 출판할 수 있다.")
    text("- 2024년 기준 작가 15만 명, 책 50만 권이다.")

    text("BooksCorpus "), link("https://arxiv.org/abs/1506.06724")
    text("- Smashwords에서 가격이 0달러인 자가출판 도서를 수집했다.")
    text("- 책 7,000권, 단어 9억 8,500만 개다.")
    text("- Smashwords 이용약관 위반으로 내려갔다. "), article_link("https://en.wikipedia.org/wiki/BookCorpus")


def gpt2_webtext():
    text("WebText: GPT-2를 학습한 데이터셋 "), link(gpt2_2019)
    text("- Reddit에서 카르마 점수가 3 이상인 게시물의 외부 링크 페이지를 모았다. 점수를 품질의 대리 지표로 썼다.")
    text("- 800만 페이지, 텍스트 40 GB다.")

    text("OpenWebTextCorpus: WebText의 공개 재현 "), link(openwebtext_2019)
    text("- Reddit 게시물 데이터셋에서 URL을 모두 추출했다.")
    text("- Facebook의 fastText 분류기로 영어가 아닌 자료를 걸러냈다.")
    text("- 유사 중복을 제거했다.")


def ccnet():
    text("CCNet "), link("https://arxiv.org/pdf/1911.00359")
    text("- 목표: 사전학습에 쓸 대규모 고품질 데이터셋을 자동으로 구축한다.")
    text("- 우르두어처럼 자료가 적은 언어의 데이터를 더 확보하는 데 특히 관심을 뒀다.")

    text("구성 요소")
    text("- 중복 제거: 간단히 정규화한 다음 중복 문단을 제거한다.")
    text("- 언어 식별: fastText 언어 분류기를 실행하고 목표 언어(예: 영어)만 남긴다.")
    text("- 품질 필터링: KenLM 5-그램 모델에서 위키백과와 비슷하게 평가되는 문서를 남긴다.")

    text("결과")
    text("- BERT 모델을 학습했을 때 Common Crawl을 CCNet으로 처리한 데이터가 위키백과보다 높은 성능을 냈다.")
    text("- CCNet은 공개 도구와 논문에서 발표한 데이터셋을 모두 가리킨다.")


def t5_c4():
    text("Colossal Clean Crawled corpus (C4) "), link("https://arxiv.org/pdf/1910.10683v4")

    text("이 논문은 모든 NLP 과제를 하나의 형식으로 바꾸는 Text-to-Text Transfer Transformer(T5)로 더 유명하지만")
    text("C4 데이터셋도 중요한 결과물이다.")

    text("관찰: Common Crawl 자료 대부분은 그대로 쓰기에 유용한 자연어가 아니다.")

    text("2019년 4월 Common Crawl 스냅숏 하나(1조 4,000억 토큰)에서 시작했다.")

    text("사람이 정한 규칙")
    text("- 구두점으로 끝나며 단어가 5개 이상인 줄만 남긴다.")
    text("- 문장이 3개 미만인 페이지를 제거한다.")
    text("- 금칙어가 하나라도 있는 페이지를 제거했다. "), article_link("https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en")
    text("- '{'(코드 제외), 'lorem ipsum', 'terms of use' 등이 포함된 페이지를 제거했다.")
    text("- langdetect로 영어 텍스트만 남겼다(영어일 확률 0.99).")

    text("최종 결과: 텍스트 806 GB(1,560억 토큰).")

    text("C4 분석 "), link("https://arxiv.org/pdf/2104.08758")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/c4-domains.png", width=700)

    text("추가 실험: WebText와 비슷한 데이터셋")
    text("- OpenWebText 링크가 가리키는 페이지로 한정했다(Reddit에서 카르마 3 이상인 게시물의 링크).")
    text("- 덤프 12개에서 텍스트 17 GB를 얻었다. WebText는 40 GB였으므로 Common Crawl이 모든 페이지를 포함하지는 않았음을 시사한다.")
    text("- GLUE, SQuAD 등 여러 NLP 벤치마크에서 성능이 개선됐다.")


def gpt3():
    text("GPT-3 데이터셋 "), link("https://arxiv.org/pdf/2005.14165")  # 2.2절
    text("- 전처리한 Common Crawl.")
    text("- WebText2: WebText에 링크를 더한 자료.")
    text("- 출처가 분명하지 않은 인터넷 도서 말뭉치(Books1, Books2).")
    text("- 위키백과")

    text("결과: 570 GB(4,000억 토큰).")

    text("Common Crawl 처리")
    text("- WebText, 위키백과, Books1, Books2를 나머지 자료와 구별하는 품질 분류기를 학습했다.")
    text("- WebText 및 벤치마크를 포함해 문서의 유사 중복을 제거했다.")


def the_pile():
    text("The Pile "), link("https://arxiv.org/pdf/2101.00027")

    text("- GPT-3에 대응해 공개 언어 모델을 만들려는 노력의 일부다.")
    text("- 다수의 자원봉사자가 Discord에서 기여하고 작업을 조율했다.")
    text("- 품질이 높은 분야 22개를 선별했다.")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png", width=600)

    text("- 텍스트 825 GB(약 2,750억 토큰)다.")
    text("- Pile-CC: Common Crawl의 WARC에서 jusText로 텍스트를 추출했다. WET보다 결과가 좋았다.")
    text("- PubMed Central: 논문 500만 편. 미국 국립보건원(NIH)의 지원을 받은 연구는 공개가 의무다.")
    text("- arXiv: 1991년부터 운영한 연구 논문 사전 공개 서비스. LaTeX를 사용한다.")
    text("- Enron 이메일: 2002년 Enron 수사 중 공개된 고위 경영진 이용자 150명의 이메일 50만 건이다. "), article_link("https://www.cs.cmu.edu/~enron/")

    project_gutenberg()
    books3()
    stackexchange()


def project_gutenberg():
    text("[Project Gutenberg](https://www.gutenberg.org/)")
    text("- 문학 작품에 대한 접근을 넓히려고 Michael Hart가 1971년 시작했다.")
    text("- 2025년에 책 약 7만 5,000권이 있으며 대부분 영어다.")
    text("- 저작권 확인을 마친 책만 포함한다. 대부분은 퍼블릭 도메인에 속한다.")

    text("PG-19: 2019년 이전 프로젝트 구텐베르크 도서 "), article_link("https://github.com/google-deepmind/pg19")


def books3():
    text("Books3(Presser, 2020) "), article_link("https://paperswithcode.com/dataset/books3")
    text("- 섀도 라이브러리 Bibliotik의 책 19만 6,000권이다."),
    text("- Stephen King, 이민진, Zadie Smith 등의 작품을 포함했다. "), article_link("https://www.wired.com/story/battle-over-books3/")
    text("- 저작권 침해 및 소송으로 내려갔다. "), article_link("https://huggingface.co/datasets/the_pile_books3")



def stackexchange():
    text("- 사용자가 질문하고 답변하는 웹사이트들의 모음이다.")
    text("- 2008년 Stack Overflow로 시작해 수학, 문학 등 다른 주제로 확장했다. "), link(title="사이트 목록", url="https://stackexchange.com/sites")
    text("- 평판 점수와 배지로 참여를 장려한다.")
    text("- [예시](https://ell.stackexchange.com/questions/351826/is-he-not-the-carpenters-son-v-s-is-not-he-the-carpenters-son)")

    text("- 질문·답변 형식이 지시 미세조정이나 실제 사용 사례와 비슷하다.")
    text("- 사용자, 투표, 댓글, 배지, 태그를 필터링에 활용할 수 있다.")
    text("- 메타데이터를 포함하는 익명화된 XML 덤프가 있다. "), link(title="링크", url="https://archive.org/details/stackexchange")



def gopher_massivetext():
    text("Gopher 학습에 사용한 MassiveText 데이터셋 "), link(gopher_2021)
    text("Gopher는 이후 Chinchilla로 대체됐다. 두 모델 모두 공개되지 않았지만 데이터에 대한 설명이 자세하다.")

    text("구성")
    text("- MassiveWeb: 뒤에서 더 다룬다.")
    text("- C4")
    text("- 도서: 세부 정보 없음.")
    text("- 뉴스: 세부 정보 없음.")
    text("- GitHub: 세부 정보 없음.")
    text("- 위키백과: 세부 정보 없음.")

    text("MassiveWeb 필터링 단계")
    text("- 영어만 남기고 중복과 학습·테스트 중첩을 처리한다.")
    text("- 분류기 대신 사람이 정한 규칙으로 품질을 거른다. 예: 단어의 80% 이상이 알파벳 문자를 하나 이상 포함한다.")
    text("- 독성 판별에는 단어 목록 대신 Google SafeSearch를 사용한다.")

    text("결과: 텍스트 10.5 TB. Gopher가 실제 학습한 양은 그중 12%인 3,000억 토큰이다.")


def llama():
    text("LLaMA 데이터셋 "), link("https://arxiv.org/pdf/2302.13971")
    text("- CCNet으로 Common Crawl을 처리하고 위키백과의 *참고문헌*에 해당하는지를 분류했다.")
    text("- C4: 규칙 기반으로 필터링했으며 더 다양하다.")
    text("- GitHub: 허용적 라이선스 자료를 남기고 사람이 정한 규칙으로 필터링했다.")
    text("- 위키백과: 2022년 6~8월의 20개 언어판에 수작업 규칙을 적용했다.")
    text("- 프로젝트 구텐베르크와 The Pile의 Books3.")
    text("- arXiv: 주석과 참고문헌을 제거하고 매크로를 본문에 전개했다.")
    text("- Stack Exchange: 가장 큰 사이트 28곳의 답변을 점수로 정렬했다.")
    text("결과: 1조 2,000억 토큰.")

    text("Together의 RedPajama v1이 이 작업을 재현했다. "), link("https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T")
    text("Cerebras의 [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama): MinHashLSH로 RedPajama v1의 중복을 제거해 얻은 6,270억 토큰 부분집합.")


def refinedweb():
    text("RefinedWeb "), link("https://arxiv.org/pdf/2306.01116") 
    text("- 주장: 웹 데이터만으로 충분하다.")
    text("- [예시](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)")
    text("- WET 대신 WARC에서 trafilatura로 HTML 본문을 추출해 텍스트로 변환한다.")
    text("- Gopher 규칙을 사용한다. 편향을 피하기 위해 기계학습 기반 필터링은 사용하지 않는다.")
    text("- 5-그램의 MinHash로 유사 중복을 제거한다.")
    text("전체 5조 토큰 중 6,000억 토큰을 공개했다.")

    text("FineWeb "), article_link("https://huggingface.co/datasets/HuggingFaceFW/fineweb")
    text("- RefinedWeb의 재현으로 시작해 처리 방법을 개선했다.")
    text("- Common Crawl 덤프 95개.")
    text("- URL과 언어를 필터링한다. p(en) > 0.65이면 남긴다.")
    text("- Gopher, C4 및 추가 수작업 규칙으로 필터링한다.")
    text("- MinHash로 유사 중복을 제거한다.")
    text("- 이메일 주소와 공인 IP 주소 등 개인 식별 정보(PII)를 익명화한다.")
    text("결과: 15조 토큰.")


def dolma():
    text("Dolma "), link("https://arxiv.org/pdf/2402.00159")
    image("https://miro.medium.com/v2/resize:fit:1400/1*-0Qqhvu7JD6Y9JgsfKJdxw.png", width=700)

    text("- Reddit: Pushshift 프로젝트(2005~2023)의 게시물과 댓글을 별개로 수록했다.")
    text("- PeS2o: Semantic Scholar의 학술 논문 4,000만 편.")
    text("- C4, 프로젝트 구텐베르크, 위키백과와 위키책.")

    text("Common Crawl 처리")
    text("- fastText 분류기로 언어를 식별해 영어를 남긴다.")
    text("- Gopher와 C4 규칙으로 품질을 걸러내고 모델 기반 필터링은 사용하지 않는다.")
    text("- 규칙과 Jigsaw 분류기로 독성을 필터링한다.")
    text("- 블룸 필터로 중복을 제거한다.")

    text("결과: 3조 토큰.")

def dclm():
    text("DataComp-LM "), link(dclm_2024)
    text("- 목표: 여러 데이터 처리 알고리즘을 실험할 표준 데이터셋을 정한다.")
    text("- Common Crawl을 처리해 240조 토큰의 DCLM-pool을 만들었다.")
    text("- DCLM-baseline은 품질 분류기로 DCLM-pool을 필터링한 결과다.")
    image("images/dclm-filter.png", width=800)

    text("### 모델 기반 필터링")
    text("양성 예제(20만 개)")
    text("- [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5): 대부분 GPT-4로 생성한 지시 데이터다([예시](https://huggingface.co/datasets/teknium/OpenHermes-2.5/viewer/default/train)).")
    text("- [ELI5](https://www.reddit.com/r/explainlikeimfive/): 궁금한 점을 묻고 답하는 서브레딧이다([예시](https://huggingface.co/datasets/sentence-transformers/eli5/viewer/pair/train)).")
    text("음성 예제(20만 개)")
    text("- [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)")
    text("결과: 3조 8,000억 토큰.")

    text("fastText 분류기를 학습해 DCLM-pool 전체에 적용했다.")
    text("이 품질 분류기는 다른 필터링 방법보다 높은 성능을 보였다.")
    image("images/dclm-quality.png", width=600)


def nemotron_cc():
    text("Nemotron-CC "), link(nemotron_cc_2024)
    text("- FineWebEdu와 DCLM은 데이터를 지나치게 많이 걸러낸다(90% 제거).")
    text("- 품질을 유지하면서 더 많은 토큰을 확보해야 한다.")
    text("- HTML을 텍스트로 바꿀 때는 더 많은 토큰을 추출한 jusText를 trafilatura 대신 썼다.")

    text("여러 분류기의 결합")
    text("- Nemotron-340B-instruct로 FineWeb 문서의 교육적 가치를 채점하고 더 빠른 모델로 증류한다.")
    text("- DCLM 분류기.")

    text("문장 재작성으로 합성 데이터 만들기")
    text("- 품질이 낮은 데이터는 언어 모델로 다시 작성한다.")
    text("- 품질이 높은 데이터로는 언어 모델이 질의응답 쌍이나 주요 정보 추출 등 과제를 만들게 한다.")

    text("결과: 6조 3,000억 토큰(고품질 부분집합은 1조 1,000억).")
    text("참고로 Llama 3는 15조, Qwen 3는 36조 토큰으로 학습했다.")
    image("images/nemotron-results.png", width=800)


def the_stack():
    text("The Stack "), link("https://arxiv.org/pdf/2211.15533")
    text("- GitHub Archive(2015~2022)에서 저장소 이름을 가져왔다.")
    text("- git clone으로 저장소 1억 3,700만 개를 받았다. 전체 파일은 510억 개이고 고유 파일은 50억 개다.")
    text("- go-license-detector로 MIT와 Apache 같은 허용적 라이선스의 코드만 남겼다.")
    text("- MinHash와 자카드 유사도로 유사 중복을 제거한다.")
    text("- 결과: 코드 3.1 TB.")

    text("Stack v2 "), link("https://arxiv.org/abs/2402.19173")
    text("- GitHub Archive의 이슈, 댓글, PR.")
    text("- Software Heritage의 저장소.")
    text("- PyPI, npm, devdocs.io 등 웹사이트를 수집한 문서.")
    text("- 바이너리 파일, 악성 코드, 봇 활동, 중복을 제거하고 개인 식별 정보를 가린 뒤 PR 일부를 표본으로 뽑는다.")
    text("- 소스 코드, 특히 Nim처럼 데이터가 적은 언어의 코드를 공통 저수준 중간 언어(LLVM)와 짝짓는다.")
    text("- 기존 데이터셋(GSM8K, 코드 경진대회, Stack Overflow, arXiv, 위키백과, OpenWebMath)도 포함한다.")

    text("풀 리퀘스트")
    text("- 구조화된 객체를 일렬의 토큰 시퀀스로 바꾼다.")
    text("- diff 주변 파일 내용 등 일부 문맥을 본문에 넣고 표본을 추린다.")
    image("images/stackv2-pr1.png", width=250), image("images/stackv2-pr2.png", width=400)


def common_pile():
    text("복습")
    text("- 인터넷 자료 거의 전부에 저작권이 있다.")
    text("- 그중 일부에는 허용적 라이선스가 있다.")
    text("- 저작권이 있는 자료를 학습에 쓰는 것이 공정 이용인지 일반적인 결론은 나지 않았다.")

    text("허용적 라이선스 데이터만으로 좋은 모델을 학습할 수 있을까?")

    text("CommonPile "), link("https://arxiv.org/pdf/2506.05209")
    image("images/commonpile.png", width=700)
    text("- 허용적 라이선스 데이터 8 TB를 수집했다.")

    text("세부 쟁점")
    text("- 라이선스 세탁: 저작권이 있는 자료를 허용적 라이선스로 바꿔 배포하는 경우다. 탐지하기 어렵다.")
    text("- 모음집에 적용되는 라이선스(예: Dolma의 ODC-By)가 개별 자료까지 포괄하지는 않는다.")
    text("- 이용 허락을 받지 않은 데이터로 학습한 언어 모델이 만든 합성 데이터의 지위는 불분명하다.")

    image("images/comma-results.png", width=700)
    text("- 어느 정도의 성능은 낼 수 있지만 토큰을 더 확보하지 않고 경쟁하기는 어렵다.")


if __name__ == "__main__":
    main()
