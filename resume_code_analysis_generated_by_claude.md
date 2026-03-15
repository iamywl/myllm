# Resume 프로젝트 — 코드 동작 원리 분석

> `/Users/ywlee/sideproejct/resume` 코드베이스를 자동 분석한 문서입니다.

---

## 1. 프로젝트 개요

순수 HTML/JavaScript와 Tailwind CSS로 만든 **정적 단일 페이지 이력서 웹사이트**입니다. 프레임워크, 빌드 도구, 백엔드 없이 `index.html`을 브라우저에서 열기만 하면 동작합니다.

- **언어**: 한국어 (`lang="ko"`)
- **소유자**: 인프라 & 보안 엔지니어 (이용원 / iamywl)
- **호스팅**: GitHub Pages (`iamywl.github.io`)

---

## 2. 프로젝트 구조

```
resume/
├── index.html          # HTML 진입점 및 페이지 껍데기
├── data.js             # 이력서 콘텐츠 전체 (데이터 객체)
├── render.js           # 렌더링 로직 (데이터 읽기 → DOM 생성)
├── ebpf-icon.svg       # eBPF 기술 아이콘 (커스텀 SVG)
├── expressions.txt     # 보조 데이터 (생년월일 & 위치)
├── README.md           # 프로젝트 설명
└── js/
    ├── app.js          # (레거시/미사용) 대체 초기화 스크립트
    └── templates.js    # (레거시/미사용) 이전 템플릿 헬퍼
```

---

## 3. 기술 스택

| 계층     | 기술                        | 비고                          |
|----------|-----------------------------|-------------------------------|
| 마크업   | HTML5                       | 시맨틱 구조                   |
| 스타일   | Tailwind CSS (CDN)          | 커스텀 설정 없음              |
| 로직     | Vanilla JavaScript (ES6+)   | 프레임워크 의존성 없음        |
| 폰트     | 시스템 폰트 스택            | -apple-system, Roboto 등     |
| 호스팅   | GitHub Pages                | 정적 파일 서빙               |

**npm 의존성 제로. 빌드 단계 제로.**

---

## 4. 아키텍처 및 데이터 흐름

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  index.html  │────▶│   data.js    │────▶│  render.js   │
│  (껍데기)     │     │  (콘텐츠)     │     │  (렌더러)     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │  DOM 출력     │
                                          │  (스타일 UI)  │
                                          └──────────────┘
```

### 실행 순서

1. 브라우저가 `index.html`을 로드
2. `<script>` 태그가 순차적으로 로드: `data.js` → `render.js`
3. `data.js`가 모든 콘텐츠를 담은 전역 객체 `resumeData`를 선언
4. `DOMContentLoaded` 이벤트 발생 시 `render.js`가 `renderResume()` 호출
5. `renderResume()`가 전역 `resumeData`를 읽고, 템플릿 리터럴로 HTML 문자열을 만들어 `<div id="resume-app"></div>`에 주입
6. CDN에서 로드된 Tailwind CSS가 모든 스타일링을 적용

---

## 5. 데이터 계층 — `data.js`

모든 이력서 콘텐츠는 단일 전역 객체 `resumeData`에 저장됩니다.

```javascript
resumeData = {
  profile: {
    name,        // "이용원 (iamywl)"
    keyword,     // 태그라인 배지 텍스트
    title,       // "Infrastructure & Security Engineer"
    email,       // 연락처 이메일
    github,      // GitHub 사용자명
    summary,     // 짧은 요약 (현재 비어 있음)
    fullIntro    // 장문 철학 텍스트 (eBPF, 커널, 보안 관련)
  },
  projects: [    // 프로젝트 객체 배열
    {
      title,         // 프로젝트명
      period,        // "YYYY.MM - YYYY.MM"
      description,   // 한 줄 설명
      points,        // 성과 항목 문자열 배열
      link           // GitHub URL 또는 미공개 시 "#"
    }
  ],
  awards: [      // 수상/활동 객체 배열
    {
      date,      // "YYYY.MM.DD"
      title,     // 수상명
      org        // 수여 기관
    }
  ],
  techStack: [   // 기술 항목 배열
    {
      name,      // 기술명
      img        // 로고 URL (외부 CDN 또는 로컬 SVG)
    }
  ]
}
```

### 콘텐츠 요약

- **프로젝트 3개**: 컨테이너 보안 기법 연구(NSR), ROMA 룸메이트 매칭 인프라, 부동산 데이터 서비스
- **수상 6개**: CKA 자격증, 해커톤 수상, CNCF 기여 등
- **기술 6개**: Kubernetes, AWS, Ansible, Jenkins, eBPF, Docker

---

## 6. 렌더링 엔진 — `render.js`

### `renderResume()` 함수

유일한 렌더링 함수로 다음을 수행합니다:

1. `#resume-app` DOM 요소를 가져옴
2. `resumeData` 존재 여부 검증
3. ES6 템플릿 리터럴로 전체 HTML 문자열 구성
4. `.map().join('')`으로 배열(프로젝트, 수상, 기술스택) 순회
5. 결과를 `innerHTML`에 할당

### 주요 렌더링 동작

| 기능               | 구현 방식                                                          |
|--------------------|--------------------------------------------------------------------|
| 반응형 레이아웃     | Tailwind 브레이크포인트: `sm:`, `md:` 접두사 (모바일 우선)          |
| 프로젝트 링크       | 조건부: `link === '#'`이면 "Coming Soon" 배지 표시                  |
| 이미지 폴백         | `onerror` 속성으로 깨진 로고를 플레이스홀더로 교체                   |
| 호버 효과           | `hover:scale-105`, `hover:border-l-4 border-blue-500`             |
| 그리드 레이아웃     | 수상: 데스크톱 2열. 기술스택: 모바일 3열 → 데스크톱 6열              |

---

## 7. 페이지 섹션 (UI 구성요소)

컴포넌트 프레임워크가 없으므로, 각 섹션은 단일 함수 내 HTML 블록으로 렌더링됩니다:

### 7.1 헤더 / 프로필
- 키워드 배지가 있는 이름
- 직함, 이메일, GitHub 링크
- 그래디언트 배경과 둥근 카드 스타일

### 7.2 철학 / 상세 소개
- 어두운 배경의 큰 텍스트 블록 (`from-slate-800 to-slate-900`)
- 모서리에 장식용 SVG 요소
- `fullIntro` 포함 — eBPF와 보안 철학에 대한 상세 설명

### 7.3 주요 프로젝트
- 호버 시 왼쪽 테두리 강조가 있는 카드 목록
- 각 카드: 제목, 기간, 설명, 성과 항목, GitHub 링크
- 미공개 프로젝트에 "Coming Soon" 배지

### 7.4 활동 & 수상
- 2열 반응형 그리드
- 각 카드: 날짜, 제목, 기관

### 7.5 공식 기술 스택
- 아이콘 그리드 (3열 → 6열)
- 외부 로고 이미지와 onerror 폴백
- 호버 시 확대 애니메이션

---

## 8. 외부 리소스

| 리소스             | URL / 출처                                           | 용도               |
|--------------------|------------------------------------------------------|--------------------|
| Tailwind CSS       | `https://cdn.tailwindcss.com`                        | CSS 프레임워크     |
| Kubernetes 로고    | GitHub raw (kubernetes/kubernetes 저장소)             | 기술스택 아이콘    |
| AWS 로고           | Wikimedia Commons                                    | 기술스택 아이콘    |
| Ansible 로고       | Wikimedia Commons                                    | 기술스택 아이콘    |
| Jenkins 로고       | jenkins.io                                           | 기술스택 아이콘    |
| Docker 로고        | docker.com                                           | 기술스택 아이콘    |
| eBPF 아이콘        | 로컬 파일 `./ebpf-icon.svg`                          | 기술스택 아이콘    |

**API 호출 없음. 백엔드 없음. 데이터베이스 없음.**

---

## 9. 레거시 / 미사용 코드

`js/` 폴더의 두 파일은 `index.html`에서 로드하지 않습니다:

- **`js/app.js`** — 불완전한 대체 초기화 스크립트 (~14줄). 어디서도 참조되지 않음.
- **`js/templates.js`** — 이전 템플릿 헬퍼 시스템. `render.js`의 인라인 템플릿 방식으로 대체됨.

안전하게 삭제할 수 있습니다.

---

## 10. 수정 가이드

| 변경 대상            | 수정 파일        | 방법                                             |
|----------------------|------------------|-------------------------------------------------|
| 이력서 콘텐츠        | `data.js`        | `resumeData` 객체 업데이트                       |
| 레이아웃 / 스타일    | `render.js`      | 템플릿 문자열 내 Tailwind 클래스 수정             |
| 페이지 구조          | `render.js`      | `renderResume()`에서 HTML 섹션 추가/제거          |
| 페이지 메타데이터    | `index.html`     | `<title>`, `<meta>` 업데이트 또는 스크립트 추가   |
| 기술스택 아이콘      | `data.js`        | `techStack` 배열의 `img` URL 변경                 |

---

## 11. 핵심 설계 결정

1. **프레임워크 없음** — 의존성 제로, 즉시 로딩 가능
2. **데이터/뷰 분리** — 모든 콘텐츠가 `data.js`에 있어 렌더링 로직 수정 없이 업데이트 가능
3. **CDN 기반 Tailwind** — 빌드 도구 불필요 (초기 CSS 용량이 약간 큰 대신)
4. **모바일 우선 반응형** — 모바일에서 단일 열, 데스크톱에서 다중 열
5. **클라이언트 사이드 전용** — 서버 없음, SSR 없음, 하이드레이션 없음 — GitHub Pages로 정적 파일만 서빙
