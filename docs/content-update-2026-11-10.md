# 신규 자료·한국어 가이드 확인 기록

확인일: **2026-11-10 (Asia/Seoul)**

이번 변경은 신규 자료 6개, 목적별 탐색, 자료별 한국어 학습 가이드입니다. 기존 전체 목록을 재검증한 기록은 아닙니다. 2026 학기 강좌와 계속 갱신되는 웹 강좌를 구분하며, 확인 날짜를 강좌 제작 연도로 사용하지 않습니다.

## 추가 자료와 근거

| 자료 | 직접 확인한 출처 | 반영 내용·한계 |
| --- | --- | --- |
| CMU 11-785 | [2026 Fall 홈페이지](https://deeplearning.cs.cmu.edu/F26/), [소개·선수지식](https://deeplearning.cs.cmu.edu/F26/pages/general_logistics.html), [공개 강의표](https://deeplearning.cs.cmu.edu/F26/pages/tables/lectures_table.html), [운영 안내](https://deeplearning.cs.cmu.edu/F26/pages/syllabus.html) | 학기·강사·Python/수학 선수지식, 차시별 YouTube·슬라이드 연결 확인. 11월 이후 일정과 미공개 영상 칸이 있어 진행 중·일부 공개로 표기. 과제용 Canvas·Gradescope·Hohum 접근은 수강생 안내와 구분. |
| Berkeley CS185/285 | [공식 홈페이지](https://rail.eecs.berkeley.edu/deeprlcourse/), [강의계획](https://rail.eecs.berkeley.edu/deeprlcourse/syllabus/), 홈페이지가 연결한 [2023 녹화](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps) | 2026 Spring 자료와 LLM RL 주제·과제, 공식 선수지식 확인. 홈페이지의 공개 녹화 연결은 2023 Fall이며 현재 학기 영상 안내는 bCourses. 2026 전체 녹화의 공개 여부는 확정하지 않고 영상·자료 학기를 분리. |
| Hugging Face LLM Course | [영어 소개·목차](https://huggingface.co/learn/llm-course/chapter1/1), [한국어 소개·목차](https://huggingface.co/learn/llm-course/ko/chapter1/1) | 무료 안내, Transformers·Datasets·Tokenizers, 영어판의 데이터 품질·미세조정·추론 모델 단원 확인. 한국어판 목차와 후반 단원 소개는 차이가 있어 일부 번역으로 표기. |
| Hugging Face Agents Course | [영어 안내](https://huggingface.co/learn/agents-course/en/unit0/introduction), [한국어 안내](https://huggingface.co/learn/agents-course/ko/unit0/introduction) | 도구 호출·프레임워크·RAG·평가 구성과 Python/LLM 선수지식 확인. 영어판 인증 안내는 마감 없음, 한국어판에는 2025-05-01 마감이 남아 있어 차이를 가이드에 표시. |
| Hugging Face MCP Course | [공식 소개·목차](https://huggingface.co/learn/mcp-course/unit0/introduction) | Anthropic 협력·무료 안내·MCP 기초와 앱 구현·배포 구성 확인. LLM/API·Python 또는 TypeScript 선수지식과 계정 안내 반영. 특정 SDK 버전 실행·한국어판 유무는 미검증. |
| Microsoft AI Agents for Beginners | [영어 README](https://github.com/microsoft/ai-agents-for-beginners/blob/main/README.md), [한국어 README](https://github.com/microsoft/ai-agents-for-beginners/blob/main/translations/ko/README.md) | 18개 레슨 목록, 일부 레슨의 영상 링크, Co-op Translator 번역, Microsoft Agent Framework·Foundry Agent Service V2·Azure 계정 요구 확인. 원문의 무료 공개와 실행 서비스 비용을 구분. |

## 확인본 식별자

GitHub 파일은 아래 blob SHA로 이번 확인본을 식별합니다. 운영자가 변경하는 외부 홈페이지에는 확인 날짜를 적용합니다.

| 파일 | 확인한 blob SHA |
| --- | --- |
| `CMU-IDeeL/CMU-IDeeL.github.io` · `F26/pages/tables/lectures_table.html` | `dded41245737b9dcd6a1c43d4bf9dee008e247ae` |
| `microsoft/ai-agents-for-beginners` · `README.md` | `58d295fa647db1c9bc30a9d370ed1e5fb6a6787a` |
| `microsoft/ai-agents-for-beginners` · `translations/ko/README.md` | `cec145f7e88a88ce97abe0368dc5e1ccf2d21b9e` |

## 편집 기준과 검증 범위

- 소속별 원래 목록과 대학 중심 from scratch 경로를 유지하고, 한국어 가이드에 학습 목적·권장 수준을 추가했습니다. 수준·추천 순서·작은 실습 목표는 편집자의 제안입니다.
- 가이드는 원문 전체 번역이나 대학의 공식 학습 경로가 아닙니다. 이미 제공되는 한국어판은 원문과 함께 연결했고 번역 본문·강의 영상·슬라이드는 복제하지 않았습니다.
- 자료가 공개되어 있다는 사실과 채점·인증·GPU·API 이용 조건은 별개입니다. 모든 실습이 무료 환경에서 실행된다고 보장하지 않습니다.
- 확인한 것은 공식 페이지의 본문·목차·제공자·학기·공개 링크와 일부 번역 안내입니다. 전체 영상 시청·지역별 재생·번역 전체 대조·과제 실행·인증 신청은 수행하지 않았습니다.
- 문서 내부 경로·앵커 65개와 새 문서의 표 열 수를 검사했습니다. 신규 문서의 외부 URL 16개 중 15개는 HTTP 200을 확인했습니다. YouTube 재생목록은 검색 도구에서 제목·공식 연결을 확인했으나 로컬 HTTP 요청은 프록시 403으로 차단되어 재생 검증으로 간주하지 않았습니다.
